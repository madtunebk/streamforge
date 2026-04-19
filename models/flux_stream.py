"""
StreamForge — FLUX.1-dev stream mode
Needs ~24GB VRAM normally. Runs under 4GB with streaming.
"""

import os, sys, torch, time, random
from diffusers import FluxPipeline, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.layer_prefetcher import LayerPrefetcher
from accelerate import cpu_offload

MODEL_ID   = "black-forest-labs/FLUX.1-dev"
prompt     = "A majestic white unicorn galloping through an enchanted forest, golden horn glowing, magical sparkles, cinematic, 4K, photorealistic"
width, height, steps, cfg = 1024, 1024, 20, 3.5
DEVICE     = torch.device("cuda:0")
OUT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "flux"))
os.makedirs(OUT_DIR, exist_ok=True)

pipe = FluxPipeline.from_pretrained(MODEL_ID, transformer=None, torch_dtype=torch.bfloat16)

print("Loading transformer to CPU RAM...")
transformer = AutoModel.from_pretrained(MODEL_ID, subfolder="transformer", device_map={"": "cpu"}, torch_dtype=torch.bfloat16)

# Flux has two block lists — stream both
blocks1 = list(transformer.transformer_blocks)
blocks2 = list(transformer.single_transformer_blocks)
all_blocks = blocks1 + blocks2
print(f"  {len(blocks1)} transformer_blocks + {len(blocks2)} single_transformer_blocks = {len(all_blocks)} total")

prefetcher = LayerPrefetcher(all_blocks, DEVICE)
prefetcher.save_params()
prefetcher.install_pre_hooks()
cpu_offload(transformer, execution_device=DEVICE)
prefetcher.install_post_hooks()

pipe.transformer = transformer

generator = torch.Generator(device="cpu").manual_seed(random.randint(10000000, 99999999))
t0 = time.perf_counter()
with torch.no_grad():
    result = pipe(prompt=prompt, width=width, height=height,
                  num_inference_steps=steps, guidance_scale=cfg, generator=generator)
print(f"Done in {time.perf_counter()-t0:.1f}s")

prefetcher.remove()
for i, img in enumerate(result.images):
    path = os.path.join(OUT_DIR, f"flux_{time.time_ns()}_{i}.png")
    img.save(path)
    print(f"Saved → {path}")
