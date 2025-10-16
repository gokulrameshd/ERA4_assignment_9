"""
auto_benchmark_dataloader.py
----------------------------
Automatically test different DataLoader configurations (num_workers, prefetch_factor, batch_size)
and report the fastest combination (images/sec).
"""

import time
import torch
from data_loader import get_dataloaders

def benchmark_once(batch_size, num_workers, prefetch_factor, num_batches=100):
    """Run one benchmark configuration and measure throughput."""
    start = time.time()
    train_loader, _, _ = get_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        img_size=224,
    )

    # Manually override DataLoader parameters if needed
    train_loader.num_workers = num_workers
    train_loader.prefetch_factor = prefetch_factor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_images = 0
    start_time = time.time()

    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        total_images += inputs.size(0)

    elapsed = time.time() - start_time
    throughput = total_images / elapsed
    return throughput, elapsed


def auto_benchmark():
    batch_sizes = [32, 64, 128]
    num_workers_list = [2, 4, 8]
    prefetch_factors = [2, 4]

    results = []
    print("\n🚀 Starting Auto DataLoader Benchmark...\n")

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            for prefetch_factor in prefetch_factors:
                try:
                    throughput, elapsed = benchmark_once(
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        num_batches=50  # enough for quick estimate
                    )
                    result = {
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "prefetch_factor": prefetch_factor,
                        "throughput": throughput,
                        "time": elapsed
                    }
                    results.append(result)
                    print(f"✅ batch={batch_size:<4} workers={num_workers:<2} "
                          f"prefetch={prefetch_factor:<2} → "
                          f"{throughput:8.2f} img/s in {elapsed:.2f}s")
                except Exception as e:
                    print(f"⚠️ batch={batch_size}, workers={num_workers}, prefetch={prefetch_factor} failed: {e}")

    # Find best configuration
    best = max(results, key=lambda x: x["throughput"])
    print("\n🏁 Best Configuration:")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Num Workers: {best['num_workers']}")
    print(f"Prefetch Factor: {best['prefetch_factor']}")
    print(f"Throughput: {best['throughput']:.2f} images/sec\n")

    # Optional: Save results to CSV
    import pandas as pd
    pd.DataFrame(results).to_csv("dataloader_benchmark_results.csv", index=False)
    print("🧾 Saved results to dataloader_benchmark_results.csv")


if __name__ == "__main__":
    auto_benchmark()
