import json
import os
from collections import Counter

def analyze_known_facts(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    
    unique_subjects = set()
    unique_objects = set()
    unique_relations = set()

    for item in data:
        # Extract the necessary fields based on the structure of known_pure_facts.json
        subject = item.get('clean_subject')
        obj = item.get('clean_object')
        relation = item.get('relation')

        if subject:
            unique_subjects.add(subject)
        if obj:
            unique_objects.add(obj)
        if relation:
            unique_relations.add(relation)

    # Entities are the union of subjects and objects
    unique_entities = unique_subjects.union(unique_objects)

    print("=" * 40)
    print(f"📊 Statistics for {os.path.basename(filepath)}")
    print("=" * 40)
    print(f"Total Samples (Facts) : {total_samples:,}")
    print(f"Unique Subjects       : {len(unique_subjects):,}")
    print(f"Unique Objects        : {len(unique_objects):,}")
    print(f"Total Unique Entities : {len(unique_entities):,}")
    print(f"Unique Relations      : {len(unique_relations):,}")
    print("=" * 40)

def analyze_temporal_chains(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_chains = len(data)
    
    unique_subjects = set()
    unique_objects = set()
    unique_relations = set()
    chain_lengths = []

    for chain in data:
        subject = chain.get('subject')
        relation = chain.get('relation')
        
        if subject:
            unique_subjects.add(subject)
        if relation:
            unique_relations.add(relation)
            
        events = chain.get('events', [])
        chain_lengths.append(len(events))
        
        for event in events:
            obj = event.get('clean_object')
            if obj:
                unique_objects.add(obj)

    unique_entities = unique_subjects.union(unique_objects)
    
    max_length = max(chain_lengths) if chain_lengths else 0
    # Calculate mode of chain lengths
    if chain_lengths:
        length_counts = Counter(chain_lengths)
        mode_length, mode_count = length_counts.most_common(1)[0]
    else:
        mode_length, mode_count = 0, 0

    print("=" * 40)
    print(f"🔗 Statistics for {os.path.basename(filepath)}")
    print("=" * 40)
    print(f"Total Chains          : {total_chains:,}")
    print(f"Max Chain Length      : {max_length}")
    print(f"Chain Length Mode     : {mode_length} (appears {mode_count} times)")
    print("-" * 40)
    print(f"Unique Subjects       : {len(unique_subjects):,}")
    print(f"Unique Objects        : {len(unique_objects):,}")
    print(f"Total Unique Entities : {len(unique_entities):,}")
    print(f"Unique Relations      : {len(unique_relations):,}")
    print("=" * 40)

def analyze_benchmark(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def _stats(dataset_subset, subset_name):
        total_samples = len(dataset_subset)
        
        unique_subjects = set()
        unique_objects = set()
        unique_relations = set()
        
        # To count frequencies of relations
        relation_counter = Counter()
        
        for item in dataset_subset:
            subject = item.get('subject')
            relation = item.get('relation')
            
            # The object being edited towards
            new_object = item.get('new_object')
            
            # Also capture the old object if present (useful for modify cases)
            old_object = item.get('old_object')
            
            # Look inside metrics for other entities involved (history/future localities)
            # This ensures we count all entities involved in the benchmark item
            metrics = item.get('metrics', {})
            hist_target = metrics.get('history_locality', {}).get('target')
            fut_target = metrics.get('future_locality', {}).get('target')
            
            if subject:
                unique_subjects.add(subject)
            if relation:
                unique_relations.add(relation)
                relation_counter[relation] += 1
            if new_object:
                unique_objects.add(new_object)
            if old_object:
                unique_objects.add(old_object)
            if hist_target:
                unique_objects.add(hist_target)
            if fut_target:
                unique_objects.add(fut_target)

        unique_entities = unique_subjects.union(unique_objects)
        
        # Get the most common relation and its count
        if relation_counter:
            top_relation, top_count = relation_counter.most_common(1)[0]
        else:
            top_relation, top_count = "N/A", 0

        print(f"--- {subset_name} ---")
        print(f"Total Samples         : {total_samples:,}")
        print(f"Unique Subjects       : {len(unique_subjects):,}")
        print(f"Unique Objects        : {len(unique_objects):,}")
        print(f"Total Unique Entities : {len(unique_entities):,}")
        print(f"Unique Relations      : {len(unique_relations):,}")
        print(f"Top Relation          : '{top_relation}' (with {top_count} samples)")
        print("-" * 40)

    print("=" * 40)
    print(f"🏆 Statistics for {os.path.basename(filepath)}")
    print("=" * 40)
    
    # 1. Stats for All Cases
    _stats(data, "All Cases")
    
    # 2. Stats for Hard Cases Only
    hard_cases = [item for item in data if item.get('case_type') == 'Hard']
    _stats(hard_cases, "Hard Cases Only")
    
    print("=" * 40)


if __name__ == "__main__":
    base_dir = "dataset/llama"
    analyze_known_facts(os.path.join(base_dir, "known_pure_facts.json"))
    print("\n")
    analyze_temporal_chains(os.path.join(base_dir, "temporal_chains.json"))
    print("\n")
    analyze_benchmark(os.path.join(base_dir, "tke_benchmark_modify.json"))
