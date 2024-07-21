import time
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import os
import re
import math

print("Starting benchmark script...")

model_name = "dunzhang/stella_en_1.5B_v5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()
print("Model loaded successfully")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def benchmark(input_ids, use_cuda_graph=False):
    torch.cuda.synchronize()  # Ensure previous operations are complete
    torch.cuda.reset_peak_memory_stats()
    
    if use_cuda_graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = model(input_ids)
    
    start_time = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision
        if use_cuda_graph:
            g.replay()
        else:
            model(input_ids)
    torch.cuda.synchronize()
    return time.time() - start_time

def run_benchmark(input_length, batch_size=1, num_warmup=50, num_iter=100):
    try:
        print(f"Running benchmark for input length: {input_length}, batch size: {batch_size}")
        text = "This is a sample text for benchmarking. " * input_length
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=input_length)
        input_ids = inputs.input_ids.repeat(batch_size, 1).to(device)
        num_tokens = input_ids.numel()
        
        print(f"  Number of tokens: {num_tokens}")
        print("  Warming up...")
        for _ in range(num_warmup):
            benchmark(input_ids)
        
        print("  Running benchmark...")
        times = [benchmark(input_ids) for _ in range(num_iter)]
        
        try:
            times_cuda_graph = [benchmark(input_ids, use_cuda_graph=True) for _ in range(num_iter)]
            avg_time_cuda_graph = sum(times_cuda_graph) / len(times_cuda_graph)
        except Exception as e:
            print(f"  CUDA graph benchmark failed: {e}")
            avg_time_cuda_graph = float('nan')
        
        avg_time = sum(times) / len(times)
        
        result = {
            "Input Length": input_length,
            "Batch Size": batch_size,
            "Num Tokens": num_tokens,
            "Avg Time (s)": f"{avg_time:.3f}",
            "Avg Time CUDA Graph (s)": f"{avg_time_cuda_graph:.3f}" if not math.isnan(avg_time_cuda_graph) else "N/A",
            "Throughput (inferences/s)": f"{1/avg_time:.2f}",
            "Throughput CUDA Graph (inferences/s)": f"{1/avg_time_cuda_graph:.2f}" if not math.isnan(avg_time_cuda_graph) else "N/A",
            "Tokens/s": f"{num_tokens/avg_time:.2f}",
            "Tokens/s CUDA Graph": f"{num_tokens/avg_time_cuda_graph:.2f}" if not math.isnan(avg_time_cuda_graph) else "N/A",
            "Max Memory Allocated (GB)": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}",
            "Max Memory Reserved (GB)": f"{torch.cuda.max_memory_reserved() / 1e9:.2f}"
        }
        print(f"  Completed: Avg Time = {avg_time:.3f}s, Throughput = {1/avg_time:.2f} inferences/s, Tokens/s = {num_tokens/avg_time:.2f}")
        if not math.isnan(avg_time_cuda_graph):
            print(f"  With CUDA Graph: Avg Time = {avg_time_cuda_graph:.3f}s, Throughput = {1/avg_time_cuda_graph:.2f} inferences/s, Tokens/s = {num_tokens/avg_time_cuda_graph:.2f}")
        return result
    except Exception as e:
        print(f"Benchmark failed for input length: {input_length}, batch size: {batch_size}")
        print(f"Error: {e}")
        return None

# Main benchmark loop
input_lengths = [10, 50, 100, 500]
batch_sizes = [1, 8, 32]
results = []

for length in input_lengths:
    for batch_size in batch_sizes:
        result = run_benchmark(length, batch_size)
        if result is not None:
            results.append(result)
        print(f"Completed benchmark for input length: {length}, batch size: {batch_size}")

# Only create DataFrame if there are results
if results:
    df = pd.DataFrame(results)
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()

    print(f"\nBenchmark Results for {gpu_count}x {gpu_name}:")
    print(df.to_string(index=False))

    # Extract GPU model name and create filename
    gpu_model = re.sub(r'\s+', '_', gpu_name.lower())
    gpu_model = re.sub(r'[^\w\-_]', '', gpu_model)
    filename = f"results/{gpu_model}_{gpu_count}.txt"

    # Save results to file
    os.makedirs("results", exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"Benchmark Results for {gpu_count}x {gpu_name}:\n")
        f.write(df.to_string(index=False))

    print(f"\nResults saved to {filename}")
else:
    print("No successful benchmark results to report.")

print("\nBenchmark completed.")
