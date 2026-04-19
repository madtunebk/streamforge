"""
StreamForge — Z-Image-Turbo stream mode

Where it all started: the original discovery that cpu_offload + async prefetch
creates a universal streaming engine. 30 transformer blocks, 1.4GB VRAM peak.
"""

import os, sys, torch, time, random
import safetensors.torch as st
from diffusers import ZImagePipeline, AutoModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.layer_prefetcher import setup_stream_mode

MODEL_ID   = "Tongyi-MAI/Z-Image-Turbo"
prompt     = "A majestic white unicorn galloping through an enchanted forest, golden horn glowing, magical sparkles, cinematic, 4K, photorealistic"
neg_prompt = "blurry, low quality, watermark, deformed"
width, height, steps, cfg, imgs = 768, 768, 9, 1.0, 1
lora_path  = None   # path to .safetensors LoRA, or None
lora_scale = 0.75
DEVICE     = torch.device("cuda:0")
OUT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "zimage"))
os.makedirs(OUT_DIR, exist_ok=True)


def fuse_lora(transformer, path: str, scale: float):
    sd = st.load_file(path)
    lora_A = {k[:-len(".lora_A.weight")]: v for k, v in sd.items() if k.endswith(".lora_A.weight")}
    lora_B = {k[:-len(".lora_B.weight")]: v for k, v in sd.items() if k.endswith(".lora_B.weight")}
    strip = lambda k: k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
    lora_A = {strip(k): v for k, v in lora_A.items()}
    lora_B = {strip(k): v for k, v in lora_B.items()}
    params = dict(transformer.named_parameters())
    fused = 0
    for key in lora_A:
        if key not in lora_B:
            continue
        p = params.get(f"{key}.weight")
        if p is None:
            continue
        A = lora_A[key].to(dtype=p.dtype)
        B = lora_B[key].to(dtype=p.dtype)
        p.data += scale * (B @ A)
        fused += 1
    print(f"  Fused {fused} LoRA layers (scale={scale})")


# Pipeline shell (no transformer/vae/scheduler — load them separately)
pipe = ZImagePipeline.from_pretrained(
    MODEL_ID,
    transformer=None, vae=None, scheduler=None,
    torch_dtype=torch.bfloat16,
)

# Encode text on CPU then unload
with torch.no_grad():
    tokens = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    prompt_embeds = pipe.text_encoder(**tokens).last_hidden_state
    neg_tokens = pipe.tokenizer(neg_prompt or "", return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    neg_embeds  = pipe.text_encoder(**neg_tokens).last_hidden_state
pipe.text_encoder = None
pipe.tokenizer = None

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
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        width=width, height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=imgs,
        generator=generator,
        output_type="latent",
    )
print(f"Generation: {time.perf_counter()-t0:.1f}s")

prefetcher.remove()
pipe.transformer = None

# Decode
pipe.vae = AutoModel.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to(DEVICE)
with torch.no_grad():
    for i, latent in enumerate(result.images):
        image = pipe.vae.decode(latent.unsqueeze(0).to(DEVICE), return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        path  = os.path.join(OUT_DIR, f"zimage_{time.time_ns()}_{i}.png")
        image.save(path)
        print(f"Saved → {path}")
