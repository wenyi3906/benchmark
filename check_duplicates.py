import json
from collections import Counter

def check():
    with open('dataset/llama/tke_benchmark_pre_modify.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = [item['metrics']['reliability']['prompt'] for item in data]
    counts = Counter(prompts)
    
    duplicates = {k: v for k, v in counts.items() if v > 1}
    print(f"Total cases: {len(data)}")
    print(f"Unique prompts: {len(counts)}")
    print(f"Number of duplicate prompt strings: {len(duplicates)}")
    
    dup_cases = sum(v for v in duplicates.values())
    print(f"Total cases involved in duplicates: {dup_cases}")
    print(f"Percentage of duplicated cases: {dup_cases / len(data) * 100:.2f}%")

if __name__ == '__main__':
    check()
