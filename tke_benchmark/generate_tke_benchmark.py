import torch
import random
from datetime import datetime, timedelta
from collections import defaultdict
import json


def generate_test_cases(input_file, output_file_insert, output_file_modify, max_cases=1, min_day_gap_insert=2, min_day_gap_modify=1):
    with open(input_file, 'r', encoding='utf-8') as f:
        chains = json.load(f)

    # ==========================================
    # Core Upgrade: Relation-Constrained Entity Pool
    # ==========================================
    relation_object_pool = defaultdict(set)
    for chain in chains:
        rel = chain['relation']
        for event in chain['events']:
            relation_object_pool[rel].add(event['clean_object'])

    # Convert set to list for random.choice later
    for rel in relation_object_pool:
        relation_object_pool[rel] = list(relation_object_pool[rel])

    test_cases_insert = []
    test_cases_modify = []
    
    insert_hard_cases_count = 0
    insert_soft_cases_count = 0
    modify_hard_cases_count = 0
    modify_soft_cases_count = 0

    print(f"Processing {len(chains)} chains...")

    for chain in chains:
        subject = chain['subject']
        relation = chain['relation']
        events = chain['events']

        # Get the legal entity pool specific to this relation
        legal_cf_objects = relation_object_pool[relation]

        if len(legal_cf_objects) < 3:
            continue

        chain_test_cases_insert = []
        chain_test_cases_modify = []

        # ==========================================
        # Strategy 1: Insert Method - Insert a counterfactual node between adjacent nodes
        # ==========================================
        for i in range(len(events) - 1):
            e1 = events[i]
            e2 = events[i + 1]

            try:
                date1 = datetime.strptime(e1['time'], "%Y-%m-%d")
                date2 = datetime.strptime(e2['time'], "%Y-%m-%d")
            except ValueError:
                continue

            delta_days = (date2 - date1).days

            if delta_days >= min_day_gap_insert:
                obj1 = e1['clean_object']
                obj2 = e2['clean_object']
                case_type = "Hard" if obj1 != obj2 else "Soft"

                mid_date = date1 + timedelta(days=delta_days // 2)
                edit_time_str = mid_date.strftime("%Y-%m-%d")

                cf_obj = random.choice(legal_cf_objects)
                while cf_obj == obj1 or cf_obj == obj2:
                    cf_obj = random.choice(legal_cf_objects)

                instance = {
                    "subject": subject,
                    "relation": relation,
                    "edit_time": edit_time_str,
                    "counterfactual_object": cf_obj,
                    "case_type": case_type,
                    "history_node": {
                        "time": e1['time'],
                        "ground_truth_object": obj1
                    },
                    "future_node": {
                        "time": e2['time'],
                        "ground_truth_object": obj2
                    }
                }
                chain_test_cases_insert.append(instance)
                
        # ==========================================
        # Strategy 2: Modify Method - Modify the object of the middle node among three adjacent nodes
        # ==========================================
        if len(events) >= 3:
            for i in range(len(events) - 2):
                e1 = events[i]
                e2 = events[i + 1]  # The one to modify
                e3 = events[i + 2]
                
                try:
                    date1 = datetime.strptime(e1['time'], "%Y-%m-%d")
                    date2 = datetime.strptime(e2['time'], "%Y-%m-%d")
                    date3 = datetime.strptime(e3['time'], "%Y-%m-%d")
                except ValueError:
                    continue

                delta_days_1 = (date2 - date1).days
                delta_days_2 = (date3 - date2).days

                if delta_days_1 >= min_day_gap_modify and delta_days_2 >= min_day_gap_modify:
                    obj1 = e1['clean_object']
                    obj2 = e2['clean_object']
                    obj3 = e3['clean_object']
                    
                    # 定义 Hard Case：中间的知识（obj2）必须区别于历史和未来。
                    # 允许 A -> B -> C（完全流转），也允许 A -> B -> A（状态短暂改变后回归）
                    # 其他任何不满足这个条件的情况（例如 A -> A -> C, 或 A -> A -> A 等）均被视作 Soft Case，不做任何丢弃限制。
                    if obj2 != obj1 and obj2 != obj3:
                        case_type = "Hard"
                    else:
                        case_type = "Soft"
                    
                    cf_obj = random.choice(legal_cf_objects)
                    while cf_obj == obj1 or cf_obj == obj3 or cf_obj == obj2:
                        cf_obj = random.choice(legal_cf_objects)
                        
                    instance = {
                        "subject": subject,
                        "relation": relation,
                        "edit_time": e2['time'],  # Same as the middle node's time
                        "counterfactual_object": cf_obj,
                        "original_object": obj2,  # Keep track of what we are modifying
                        "case_type": case_type,
                        "history_node": {
                            "time": e1['time'],
                            "ground_truth_object": obj1
                        },
                        "future_node": {
                            "time": e3['time'],
                            "ground_truth_object": obj3
                        }
                    }
                    chain_test_cases_modify.append(instance)

        # ==========================================
        # Ultimate filtering logic: Keep unlimited Hard Cases, strictly limit Soft Cases
        # ==========================================
        # Here max_cases is now equivalent to max_soft_cases_per_chain
        
        # For Insert
        hard_insert = [c for c in chain_test_cases_insert if c['case_type'] == 'Hard']
        soft_insert = [c for c in chain_test_cases_insert if c['case_type'] == 'Soft']
        
        # Always keep all Hard Cases! Soft cases are only randomly kept in small numbers as a supplement control
        sampled_insert = hard_insert + random.sample(soft_insert, min(len(soft_insert), max_cases))
        
        test_cases_insert.extend(sampled_insert)
        insert_hard_cases_count += len(hard_insert)
        insert_soft_cases_count += len(sampled_insert) - len(hard_insert)
                
        # For Modify
        hard_modify = [c for c in chain_test_cases_modify if c['case_type'] == 'Hard']
        soft_modify = [c for c in chain_test_cases_modify if c['case_type'] == 'Soft']

        sampled_modify = hard_modify + random.sample(soft_modify, min(len(soft_modify), max_cases))
            
        test_cases_modify.extend(sampled_modify)
        modify_hard_cases_count += len(hard_modify)
        modify_soft_cases_count += len(sampled_modify) - len(hard_modify)

    with open(output_file_insert, 'w', encoding='utf-8') as f:
        json.dump(test_cases_insert, f, indent=4, ensure_ascii=False)
        
    with open(output_file_modify, 'w', encoding='utf-8') as f:
        json.dump(test_cases_modify, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 40)
    print("🚀 Relation-Constrained TKE Benchmark Complete!")
    print(f"Total Insert Test Cases Generated: {len(test_cases_insert)}")
    print(f"  - Hard Cases (Object Shifted): {insert_hard_cases_count}")
    print(f"  - Soft Cases (Object Kept):    {insert_soft_cases_count}")
    print(f"Total Modify Test Cases Generated: {len(test_cases_modify)}")
    print(f"  - Hard Cases (Object Shifted): {modify_hard_cases_count}")
    print(f"  - Soft Cases (Object Kept):    {modify_soft_cases_count}")
    print("=" * 40)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_benchmark(test_cases_path, map_path, output_path, pre_modify_output_path=None, pre_insert_output_path=None):
    test_cases = load_json(test_cases_path)
    rel_map = load_json(map_path)

    final_benchmark = []
    pre_modify_benchmark = []
    pre_insert_benchmark = []

    print(f"Transforming {len(test_cases)} cases from {test_cases_path} into professional TKE benchmark...")

    for case in test_cases:
        sub = case['subject']
        rel_key = case['relation']
        rel_info = rel_map.get(rel_key)

        if not rel_info:
            continue

        # Extract tense and voice features
        past_v = rel_info['past_verb']
        ing_v = rel_info['ing_form']
        noun_form = rel_info.get('noun', rel_key.replace('_', ' ').lower())

        # Extract key time nodes and entities
        edit_date = case['edit_time']
        cf_obj = case['counterfactual_object']

        hist_date = case['history_node']['time']
        hist_obj = case['history_node']['ground_truth_object']

        future_date = case['future_node']['time']
        future_obj = case['future_node']['ground_truth_object']

        # ==========================================
        # Dynamic sampling Generality template (Reverted to pure continuation mode)
        # ==========================================
        generality_templates = [
            f"Regarding the {noun_form} involving {sub} on {edit_date}, the target was ",
            f"The specific entity associated with the {noun_form} of {sub} on {edit_date} was ",
            f"On {edit_date}, the {noun_form} concerning {sub} was directed at "
        ]
        sampled_generality_prompt = random.choice(generality_templates)

        # Build Prompts for the 7 core metrics
        metrics = {
            # 1. Reliability: Restore to original continuation style
            "reliability": {
                "prompt": f"On {edit_date}, {sub} {past_v} ",
                "target": cf_obj
            },
            # 2. Generality: Revert to pure continuation mode
            "generality": {
                "prompt": sampled_generality_prompt,
                "target": cf_obj
            },
            # 3. Temporal Reasoning: Revert to pure continuation mode
            "temporal_reasoning": {
                "prompt": f"Following the activity involving {hist_obj} on {hist_date}, {sub} next {past_v} ",
                "target": cf_obj
            },
            # 4. History Locality: Restore to original continuation style
            "history_locality": {
                "prompt": f"On {hist_date}, {sub} {past_v} ",
                "target": hist_obj
            },
            # 5. Future Locality: Restore to original continuation style
            "future_locality": {
                "prompt": f"On {future_date}, {sub} {past_v} ",
                "target": future_obj
            },
            # 6. Temporal Hop Backward: Keep Q&A style
            "temporal_hop_backward": {
                "prompt": f"Question: Before {ing_v} {cf_obj} on {edit_date}, who or what was the previous entity that {sub} {past_v}?\nAnswer: ",
                "target": hist_obj
            },
            # 7. Temporal Hop Forward: Keep Q&A style
            "temporal_hop_forward": {
                "prompt": f"Question: After {ing_v} {cf_obj} on {edit_date}, who or what was the next entity that {sub} {past_v}?\nAnswer: ",
                "target": future_obj
            }
        }

        # Assemble final item
        benchmark_item = {
            "subject": sub,
            "relation": rel_key,
            "edit_target_date": edit_date,
        }
        
        if "original_object" in case:
            benchmark_item["old_object"] = case["original_object"]
            benchmark_item["new_object"] = cf_obj
        else:
            benchmark_item["new_object"] = cf_obj
            
        benchmark_item["case_type"] = case.get('case_type', 'N/A')
        benchmark_item["method"] = "modify" if "original_object" in case else "insert"
        benchmark_item["metrics"] = metrics

        final_benchmark.append(benchmark_item)

        # ==========================================
        # Pre-modify
        # ==========================================
        if pre_modify_output_path and "original_object" in case:
            original_obj = case["original_object"]
            
            # Use original_obj to replace all cf_obj for questioning
            pre_metrics = {
                "reliability": {
                    "prompt": f"On {edit_date}, {sub} {past_v} ",
                    "target": original_obj
                },
                "generality": {
                    "prompt": sampled_generality_prompt,
                    "target": original_obj
                },
                "temporal_reasoning": {
                    "prompt": f"Following the activity involving {hist_obj} on {hist_date}, {sub} next {past_v} ",
                    "target": original_obj
                },
                "history_locality": {
                    "prompt": f"On {hist_date}, {sub} {past_v} ",
                    "target": hist_obj
                },
                "future_locality": {
                    "prompt": f"On {future_date}, {sub} {past_v} ",
                    "target": future_obj
                },
                "temporal_hop_backward": {
                    "prompt": f"Question: Before {ing_v} {original_obj} on {edit_date}, who or what was the previous entity that {sub} {past_v}?\nAnswer: ",
                    "target": hist_obj
                },
                "temporal_hop_forward": {
                    "prompt": f"Question: After {ing_v} {original_obj} on {edit_date}, who or what was the next entity that {sub} {past_v}?\nAnswer: ",
                    "target": future_obj
                }
            }
            
            pre_item = {
                "subject": sub,
                "relation": rel_key,
                "edit_target_date": edit_date,
                "original_object": original_obj,  # Revert to pure factual description, use original_object as target
                "case_type": case.get('case_type', 'N/A'),
                "method": "pre_modify",
                "metrics": pre_metrics
            }
            pre_modify_benchmark.append(pre_item)
            
        # ==========================================
        # Pre-insert
        # ==========================================
        if pre_insert_output_path and "original_object" not in case:
            # The Insert method only needs to test existing historical and future facts.
            # To maintain program structure compatibility, keep the keys for the other 5 metrics, but leave the content empty.
            pre_insert_metrics = {
                "reliability": {"prompt": "", "target": ""},
                "generality": {"prompt": "", "target": ""},
                "temporal_reasoning": {"prompt": "", "target": ""},
                "history_locality": {
                    "prompt": f"On {hist_date}, {sub} {past_v} ",
                    "target": hist_obj
                },
                "future_locality": {
                    "prompt": f"On {future_date}, {sub} {past_v} ",
                    "target": future_obj
                },
                "temporal_hop_backward": {"prompt": "", "target": ""},
                "temporal_hop_forward": {"prompt": "", "target": ""}
            }
            
            pre_insert_item = {
                "subject": sub,
                "relation": rel_key,
                "edit_target_date": edit_date,
                "case_type": case.get('case_type', 'N/A'),
                "method": "pre_insert",
                "metrics": pre_insert_metrics
            }
            pre_insert_benchmark.append(pre_insert_item)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_benchmark, f, indent=4, ensure_ascii=False)

    print(f"Done! Saved {len(final_benchmark)} structured benchmark items to {output_path}")

    if pre_modify_output_path and pre_modify_benchmark:
        with open(pre_modify_output_path, 'w', encoding='utf-8') as f:
            json.dump(pre_modify_benchmark, f, indent=4, ensure_ascii=False)
        print(f"Done! Saved {len(pre_modify_benchmark)} structured pre-modify benchmark items to {pre_modify_output_path}")

    if pre_insert_output_path and pre_insert_benchmark:
        with open(pre_insert_output_path, 'w', encoding='utf-8') as f:
            json.dump(pre_insert_benchmark, f, indent=4, ensure_ascii=False)
        print(f"Done! Saved {len(pre_insert_benchmark)} structured pre-insert benchmark items to {pre_insert_output_path}")


if __name__ == "__main__":
    pass