"""This script gives an example usage of the geobench package.
"""


import geobench
import os 



for benchmark_name in ("classification_v1.0", "segmentation_v1.0"):
    print(f"Benchmark {benchmark_name}:\n")
    for task in geobench.task_iterator(benchmark_name=benchmark_name):
        print(f"Task {task.dataset_name}:\n  {task}\n")
        
        dataset = task.get_dataset(split="train")
        breakpoint()
        sample = dataset[0]

        print(f"Sample 0 named: {sample.sample_name}")
        for band in sample.bands:
            print(f"  {band.band_info.name}: {band.data.shape}")

        print("========================================\n")