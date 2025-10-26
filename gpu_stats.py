import torch
import gc
import psutil
import os

def get_gpu_usage(device=None, print_info=True):
    """
    Monitors GPU and system memory usage for the current process.

    Args:
        device (int or torch.device or None): GPU device index or name (default: current cuda device)
        print_info (bool): If True, prints formatted stats; otherwise returns dict.

    Returns:
        dict: containing detailed memory statistics.
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None

    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    # gc.collect()

    process = psutil.Process(os.getpid())
    ram_used = process.memory_info().rss / 1024 ** 3  # in GB
    total_ram = psutil.virtual_memory().total / 1024 ** 3  # in GB

    if device is not None:
        device = torch.device(device)
        mem_alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        mem_total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        mem_free = mem_total - mem_reserved
        stats = {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_id": device.index,
            "gpu_mem_alloc_GB": round(mem_alloc, 2),
            "gpu_mem_reserved_GB": round(mem_reserved, 2),
            "gpu_mem_free_GB": round(mem_free, 2),
            "gpu_mem_total_GB": round(mem_total, 2),
            "ram_used_GB": round(ram_used, 2),
            "ram_total_GB": round(total_ram, 2),
        }
    else:
        stats = {
            "gpu_name": "CPU_ONLY",
            "ram_used_GB": round(ram_used, 2),
            "ram_total_GB": round(total_ram, 2),
        }

    if print_info:
        if device is not None:
            print(f"\n🧠 GPU Memory Usage [{stats['gpu_name']} | ID {stats['gpu_id']}]")
            print(f"Allocated : {stats['gpu_mem_alloc_GB']} GB")
            print(f"Reserved  : {stats['gpu_mem_reserved_GB']} GB")
            print(f"Free      : {stats['gpu_mem_free_GB']} GB")
            print(f"Total     : {stats['gpu_mem_total_GB']} GB")
        print(f"🧩 System RAM: {stats['ram_used_GB']} / {stats['ram_total_GB']} GB")
    return stats
