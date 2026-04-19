"""
StreamForge — Wan2.2 I2V stream mode
Runs Wan2.2 14B image-to-video on consumer GPUs.
"""

import os, sys, torch, time, random, subprocess
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, AutoModel, FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.layer_prefetcher import setup_stream_mode

MODEL_ID    = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
FIRST_FRAME = None  # path to input image
prompt      = "Cinematic slow motion, dramatic action scene, 4K, photorealistic"
neg_prompt  = "static, blurry, low quality, watermark"
wan_width, wan_height, wan_frames, wan_steps, wan_cfg = 832, 480, 17, 50, 5.0
fps         = 16
DEVICE      = torch.device("cuda:0")
DEVICE2     = torch.device("cuda:1")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "output", "wan")
os.makedirs(OUT_DIR, exist_ok=True)

assert FIRST_FRAME and os.path.exists(FIRST_FRAME), "Set FIRST_FRAME to an input image path"
first_frame = Image.open(FIRST_FRAME).convert("RGB").resize((wan_width, wan_height))

pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, transformer=None, transformer_2=None, vae=None, torch_dtype=torch.bfloat16)

print("Loading transformer → GPU0 stream...")
t1 = AutoModel.from_pretrained(MODEL_ID, subfolder="transformer", device_map={"": "cpu"}, torch_dtype=torch.bfloat16)
pf1 = setup_stream_mode(t1, DEVICE)

print("Loading transformer_2 → GPU1 stream...")
t2 = AutoModel.from_pretrained(MODEL_ID, subfolder="transformer_2", device_map={"": "cpu"}, torch_dtype=torch.bfloat16)
pf2 = setup_stream_mode(t2, DEVICE2)

vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
vae.enable_slicing(); vae.enable_tiling()

pipe.transformer = t1; pipe.transformer_2 = t2
pipe.scheduler   = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipe.vae         = vae

generator = torch.Generator(device="cpu").manual_seed(random.randint(10000000, 99999999))
t0 = time.perf_counter()
with torch.no_grad():
    result = pipe(image=first_frame, prompt=prompt, negative_prompt=neg_prompt,
                  height=wan_height, width=wan_width, num_frames=wan_frames,
                  num_inference_steps=wan_steps, guidance_scale=wan_cfg,
                  generator=generator, output_type="latent")
print(f"Generation: {time.perf_counter()-t0:.1f}s")

latents_path = os.path.join(OUT_DIR, "latents.pt")
torch.save(result.frames, latents_path)

pf1.remove(); pf2.remove()
pipe.transformer = pipe.transformer_2 = pipe.scheduler = None

latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).float()
latents_std  = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).float()
latents = result.frames[0].unsqueeze(0).float()
latents = latents / latents_std + latents_mean

with torch.no_grad():
    video = vae.decode(latents, return_dict=False)[0]

frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]

frames_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i, f in enumerate(frames):
    f.save(os.path.join(frames_dir, f"frame_{i:05d}.png"))

video_path = os.path.join(OUT_DIR, f"video_{time.time_ns()}.mp4")
subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frames_dir}/frame_%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", video_path], check=True)
print(f"Done! → {video_path}")
