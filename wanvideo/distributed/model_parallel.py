# Alternative to FSDP: Simple Model Parallelism for Multi-GPU
import torch
import torch.nn as nn
from typing import List

def distribute_model_across_gpus(model, gpu_ids: List[int], exclude_gpu_0: bool = True):
    """
    Distribute model layers across multiple GPUs using simple model parallelism.
    This avoids FSDP's parameter management issues with custom forward functions.
    """
    if exclude_gpu_0 and 0 in gpu_ids:
        gpu_ids = [gpu for gpu in gpu_ids if gpu != 0]

    if len(gpu_ids) < 2:
        print(f"Not enough GPUs for distribution, using single GPU: {gpu_ids[0] if gpu_ids else 0}")
        return model.to(f"cuda:{gpu_ids[0] if gpu_ids else 0}")

    print(f"Distributing model across GPUs: {gpu_ids}")

    # Calculate blocks per GPU
    num_blocks = len(model.blocks)
    blocks_per_gpu = num_blocks // len(gpu_ids)
    remainder = num_blocks % len(gpu_ids)

    # Distribute blocks across GPUs
    current_block = 0
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        blocks_for_this_gpu = blocks_per_gpu + (1 if gpu_idx < remainder else 0)

        # Move blocks to this GPU
        for i in range(blocks_for_this_gpu):
            if current_block < num_blocks:
                model.blocks[current_block] = model.blocks[current_block].to(f"cuda:{gpu_id}")
                print(f"Block {current_block} -> GPU {gpu_id}")
                current_block += 1

    # Move other components to first GPU
    primary_gpu = gpu_ids[0]
    model.patch_embedding = model.patch_embedding.to(f"cuda:{primary_gpu}")
    model.head = model.head.to(f"cuda:{primary_gpu}")
    model.time_embedding = model.time_embedding.to(f"cuda:{primary_gpu}")
    model.time_projection = model.time_projection.to(f"cuda:{primary_gpu}")
    model.text_embedding = model.text_embedding.to(f"cuda:{primary_gpu}")

    if hasattr(model, 'img_emb'):
        model.img_emb = model.img_emb.to(f"cuda:{primary_gpu}")

    print(f"Core components -> GPU {primary_gpu}")

    # Add device tracking for forward pass
    model._gpu_ids = gpu_ids
    model._primary_gpu = primary_gpu

    return model

def create_device_aware_forward(original_forward, gpu_ids, primary_gpu):
    """
    Wrap the forward function to handle cross-GPU data movement.
    """
    def forward_wrapper(*args, **kwargs):
        # Move inputs to primary GPU
        def move_to_primary(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(f"cuda:{primary_gpu}")
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_primary(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: move_to_primary(v) for k, v in obj.items()}
            return obj

        args = tuple(move_to_primary(arg) for arg in args)
        kwargs = {k: move_to_primary(v) for k, v in kwargs.items()}

        # Call original forward
        return original_forward(*args, **kwargs)

    return forward_wrapper
