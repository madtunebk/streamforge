"""
StreamForge — Universal Transformer Layer Prefetcher

Streams any transformer model from CPU RAM to GPU one block at a time.
Works with any model using self.blocks, self.layers, or self.transformer_blocks.

Proven on:
  - Z-Image (30 blocks, .layers)
  - Wan2.2 I2V 14B (40+40 blocks, .blocks)
  - Qwen-Image (60 blocks, .transformer_blocks)
"""

import torch
import safetensors.torch as st
from accelerate import cpu_offload


def fuse_lora(model, lora_path: str, scale: float = 1.0):
    """
    Fuse a LoRA into model weights on CPU before streaming.
    Zero VRAM cost. Works with any diffusers-compatible LoRA.
    """
    sd = st.load_file(lora_path)
    lora_A = {k[:-len(".lora_A.weight")]: v for k, v in sd.items() if k.endswith(".lora_A.weight")}
    lora_B = {k[:-len(".lora_B.weight")]: v for k, v in sd.items() if k.endswith(".lora_B.weight")}
    strip  = lambda k: k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
    lora_A = {strip(k): v for k, v in lora_A.items()}
    lora_B = {strip(k): v for k, v in lora_B.items()}
    params = dict(model.named_parameters())
    fused  = 0
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
    print(f"  Fused {fused} LoRA layers from {lora_path} (scale={scale})")
    return fused


def get_blocks(model) -> list:
    """Auto-detect transformer block list from common attribute names.
    For Flux-style models with two block lists, combines both."""
    # Flux has transformer_blocks + single_transformer_blocks — stream both
    if hasattr(model, "transformer_blocks") and hasattr(model, "single_transformer_blocks"):
        blocks = list(model.transformer_blocks) + list(model.single_transformer_blocks)
        print(f"  {len(list(model.transformer_blocks))} transformer_blocks + {len(list(model.single_transformer_blocks))} single_transformer_blocks = {len(blocks)} total")
        return blocks
    for attr in ("transformer_blocks", "blocks", "layers"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            if hasattr(blocks, "__iter__"):
                return list(blocks)
    raise ValueError(
        f"Could not find transformer blocks. Tried: transformer_blocks, blocks, layers. "
        f"Available attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )


class LayerPrefetcher:
    """
    Installs async forward hooks on transformer blocks to stream weights
    from CPU RAM to GPU one block at a time, overlapping transfer with compute.

    Usage:
        blocks = get_blocks(transformer)
        prefetcher = LayerPrefetcher(blocks, device)
        prefetcher.install_pre_hooks()
        cpu_offload(transformer, execution_device=device)
        prefetcher.install_post_hooks()

        # ... run inference ...

        prefetcher.remove()
    """

    def __init__(self, blocks: list, device: torch.device):
        self.blocks      = blocks
        self.device      = device
        self.xfer_stream = torch.cuda.Stream(device=device)
        self._cpu_params = {}
        self._gpu_params = {}
        self._pre_hooks  = []
        self._post_hooks = []

    def save_params(self, pin_memory: bool = False):
        """Save CPU param references. Enable pin_memory for faster transfers (needs enough RAM)."""
        total = 0
        for i, block in enumerate(self.blocks):
            state = {}
            for name, param in block.named_parameters():
                if pin_memory and not param.data.is_pinned():
                    param.data = param.data.pin_memory()
                state[name] = param.data
                total += 1
            self._cpu_params[i] = state
        print(f"  Saved {total} param tensors across {len(self.blocks)} blocks"
              + (" (pinned)" if pin_memory else ""))

    def _prefetch(self, idx: int):
        if idx >= len(self.blocks) or idx in self._gpu_params:
            return
        gpu_state = {}
        with torch.cuda.stream(self.xfer_stream):
            for name, cpu_t in self._cpu_params[idx].items():
                gpu_state[name] = cpu_t.to(self.device, non_blocking=True)
        self._gpu_params[idx] = gpu_state

    def _restore_cpu(self, block, idx: int):
        cpu_state = self._cpu_params.get(idx, {})
        params = dict(block.named_parameters())
        for name, cpu_t in cpu_state.items():
            if name in params and params[name].device.type == "cpu":
                params[name].data = cpu_t
        self._gpu_params.pop(idx, None)

    def install_pre_hooks(self):
        """Call BEFORE cpu_offload."""
        for i, block in enumerate(self.blocks):
            def pre(module, args, idx=i):
                torch.cuda.current_stream().wait_stream(self.xfer_stream)
                return args
            self._pre_hooks.append(block.register_forward_pre_hook(pre))
        self._prefetch(0)

    def install_post_hooks(self):
        """Call AFTER cpu_offload."""
        for i, block in enumerate(self.blocks):
            def post(module, args, output, idx=i):
                self._restore_cpu(module, idx)
                self._prefetch(idx + 1)
                return output
            self._post_hooks.append(block.register_forward_hook(post))

    def remove(self):
        for h in self._pre_hooks + self._post_hooks:
            h.remove()
        self._pre_hooks.clear()
        self._post_hooks.clear()
        self._gpu_params.clear()


def setup_stream_mode(model, device: torch.device, pin_memory: bool = False) -> LayerPrefetcher:
    """
    One-call setup: auto-detect blocks, install hooks, cpu_offload.
    Returns prefetcher — call prefetcher.remove() after inference.
    """
    blocks = get_blocks(model)
    prefetcher = LayerPrefetcher(blocks, device)
    prefetcher.save_params(pin_memory=pin_memory)
    prefetcher.install_pre_hooks()
    cpu_offload(model, execution_device=device)
    prefetcher.install_post_hooks()
    return prefetcher
