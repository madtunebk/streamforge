"""
StreamForge — Qwen-Image stream mode
Runs Qwen-Image (battles Google on quality) on a single consumer GPU.
"""

import os, sys, torch, time, random
from PIL import Image
from diffusers import QwenImagePipeline, AutoModel, FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.layer_prefetcher import setup_stream_mode

MODEL_ID   = "Qwen/Qwen-Image"
prompt     = "A majestic white unicorn galloping through an enchanted forest, golden horn glowing, magical sparkles, cinematic, 4K, photorealistic"
neg_prompt = "blurry, low quality, watermark, deformed"
width, height, steps, cfg = 768, 768, 20, 5.0
DEVICE     = torch.device("cuda:0")
OUT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "qwen"))
os.makedirs(OUT_DIR, exist_ok=True)

pipe = QwenImagePipeline.from_pretrained(MODEL_ID, transformer=None, torch_dtype=torch.bfloat16)

print("Loading transformer to CPU RAM...")
transformer = AutoModel.from_pretrained(MODEL_ID, subfolder="transformer", device_map={"": "cpu"}, torch_dtype=torch.bfloat16)
print(f"  {len(list(transformer.transformer_blocks))} blocks")

prefetcher = setup_stream_mode(transformer, DEVICE)
pipe.transformer = transformer
pipe.scheduler   = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

generator = torch.Generator(device="cpu").manual_seed(random.randint(10000000, 99999999))
t0 = time.perf_counter()
with torch.no_grad():
    result = pipe(prompt=prompt, negative_prompt=neg_prompt, width=width, height=height,
                  num_inference_steps=steps, guidance_scale=cfg, generator=generator)
print(f"Done in {time.perf_counter()-t0:.1f}s")

prefetcher.remove()
for i, img in enumerate(result.images):
    path = os.path.join(OUT_DIR, f"qwen_{time.time_ns()}_{i}.png")
    img.save(path)
    print(f"Saved → {path}")
