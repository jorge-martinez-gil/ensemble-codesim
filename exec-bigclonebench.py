# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from similarity import *


intermediate_file = 'outputs\output-bigclonebench.txt'

def calculate_metrics_for_similarity(df, similarity_column, threshold):
    predictions = df[similarity_column].apply(lambda x: 1 if x >= threshold else 0)
    accuracy = accuracy_score(df['Truth'], predictions)
    precision = precision_score(df['Truth'], predictions, zero_division=0)
    recall = recall_score(df['Truth'], predictions, zero_division=0)
    f1 = f1_score(df['Truth'], predictions, zero_division=0)
    return {
        'similarity_metric': similarity_column,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def append_results_to_file(results):
    with open(intermediate_file, 'a') as file:
        for result in results:
            converted_result = [float(item) if isinstance(item, np.floating) else item for item in result]
            file.write(json.dumps(converted_result) + '\n')

def load_json_data(file_path):
    idx_to_func = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            idx_to_func[data.get("idx")] = data.get("func")
    return idx_to_func

async def async_calculate_similarity(executor, method, code1, code2, timeout_duration=20):
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(loop.run_in_executor(executor, method.similarity, code1, code2), timeout_duration)
        return result
    except asyncio.TimeoutError:
        print(f"TimeoutError: {method.__name__} exceeded {timeout_duration} seconds")
        return 0
    except Exception as e:
        print(f"Error: {method.__name__} failed with {e}")
        return 0

async def process_function_batch_async(batch):
    batch_results = []
    similarity_methods = [
        ast, bow, codebert, comments,
        exe, fcall, fuzz, graph,
        hashing, image, jaccard, lcs,
        lev, metrics, ngrams, pdg,
        rk, semclone, semdiff, tdf,
        winn
    ]
    with ThreadPoolExecutor(max_workers=10) as executor:
        for code1, code2, truth in batch:
            similarities = await asyncio.gather(*(async_calculate_similarity(executor, method, code1, code2) for method in similarity_methods))
            batch_results.append(similarities + [truth])
    return batch_results

async def main_async():
    idx_to_func = load_json_data('datasets\BigCloneBench\data.jsonl')
    function_pairs = []
    with open(r'datasets\BigCloneBench\test.txt', 'r') as file:
        for line in file:
            idx1, idx2, truth_value = line.split()
            found_func1 = idx_to_func.get(idx1)
            found_func2 = idx_to_func.get(idx2)
            if found_func1 and found_func2:
                function_pairs.append((found_func1, found_func2, int(truth_value)))

    batch_size = 32
    function_batches = [function_pairs[i:i + batch_size] for i in range(0, len(function_pairs), batch_size)]

    for batch in tqdm(function_batches, desc="Processing Batches"):
        batch_result = await process_function_batch_async(batch)
        append_results_to_file(batch_result)

if __name__ == "__main__":
    asyncio.run(main_async())