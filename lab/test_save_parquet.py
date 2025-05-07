# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:59:27 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import time

# Direct save the entire DataFrame as a single Parquet file
def save_dataframe_direct(df, output_dir, feature):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{feature}_direct.parquet"
    df.to_parquet(file_path, index=True)


# Function to save a chunk of DataFrame as Parquet
def save_parquet_chunk(df_chunk, file_path):
    df_chunk.to_parquet(file_path, index=True)

# Function to save DataFrame in chunks
def save_dataframe_in_chunks(df, output_dir, feature, max_workers=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, col in enumerate(df.columns):
            chunk_file = output_dir / f"{feature}_chunk_{i}.parquet"
            chunk_files.append(chunk_file)
            futures.append(executor.submit(save_parquet_chunk, df[[col]], chunk_file))

        for future in futures:
            future.result()

    # Merge all chunks
    final_path = output_dir / f"{feature}.parquet"
    pd.concat([pd.read_parquet(chunk_file) for chunk_file in chunk_files], axis=1).to_parquet(final_path)

    # Clean up chunk files
    for chunk_file in chunk_files:
        chunk_file.unlink()

# Generate a large DataFrame for testing
def generate_large_dataframe(rows, cols):
    np.random.seed(42)
    data = np.random.rand(rows, cols)
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])
    return df


# Testing
output_dir = "./test_parquet_output"
feature_name = "large_dataframe_test"
os.makedirs(output_dir, exist_ok=True)

# Generate large DataFrame
df = generate_large_dataframe(rows=10_000, cols=1_000)

# Save using the chunk-based function
start_time = time.time()
save_dataframe_in_chunks(df, output_dir, feature_name, max_workers=4)
end_time = time.time()

# Measure time taken
time_taken = end_time - start_time
time_taken

# Save using direct method
start_time = time.time()
save_dataframe_direct(df, output_dir, feature_name)
end_time = time.time()

print(f"Direct save time: {end_time - start_time} seconds")

