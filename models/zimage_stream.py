"""
StreamForge — Z-Image-Turbo stream mode

Where it all started: the original discovery that cpu_offload + async prefetch
creates a universal streaming engine. 30 transformer blocks, 1.4GB VRAM peak.
"""

import os, sys, torch, time, random
from diffusers import ZImagePipeline, AutoModel, FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.layer_prefetcher import setup_stream_mode, fuse_lora

MODEL_ID   = "Tongyi-MAI/Z-Image-Turbo"
prompt     = "A majestic white unicorn galloping through an enchanted forest, golden horn glowing, magical sparkles, cinematic, 4K, photorealistic"
neg_prompt = "blurry, low quality, watermark, deformed"
width, height, steps, cfg, imgs = 768, 768, 9, 1.0, 1
lora_path  = None   # path to .safetensors LoRA, or None
lora_scale = 0.75
DEVICE     = torch.device("cuda:0")
OUT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "zimage"))
os.makedirs(OUT_DIR, exist_ok=True)

# Load pipeline shell (no transformer — let pipeline handle text encoding)
pipe = ZImagePipeline.from_pretrained(MODEL_ID, transformer=None, torch_dtype=torch.bfloat16)

# Load transformer to CPU + optional LoRA fusion
print("Loading transformer to CPU RAM...")
transformer = AutoModel.from_pretrained(MODEL_ID, subfolder="transformer", device_map={"": "cpu"}, torch_dtype=torch.bfloat16)
print(f"  {len(list(transformer.layers))} blocks (uses .layers)")

if lora_path and os.path.exists(lora_path):
    print(f"Fusing LoRA: {lora_path}")
    fuse_lora(transformer, lora_path, lora_scale)

# One call — installs hooks, enables streaming, returns prefetcher
prefetcher = setup_stream_mode(transformer, DEVICE)
pipe.transformer = transformer
pipe.scheduler   = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# Generate
generator = torch.Generator(device="cpu").manual_seed(random.randint(10000000, 99999999))
t0 = time.perf_counter()
with torch.no_grad():
    result = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=width, height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=imgs,
        generator=generator,
    )
print(f"Done in {time.perf_counter()-t0:.1f}s")

prefetcher.remove()
for i, img in enumerate(result.images):
    path = os.path.join(OUT_DIR, f"zimage_{time.time_ns()}_{i}.png")
    img.save(path)
    print(f"Saved → {path}")
