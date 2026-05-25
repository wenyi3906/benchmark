import json
import string
import os

def normalize_answer(s):
    """
    Standardize format: convert to lowercase, remove punctuation, and remove extra spaces.
    """
    if not isinstance(s, str):
        if isinstance(s, list) and len(s) > 0:
            s = s[0]
        else:
            return ""
            
    def remove_punc(text):
        exclude = set(string.punctuation)
        # Remove special brackets as punctuation to prevent formatting interference
        exclude.update(['【', '】', '「', '」', '《', '》'])
        return ''.join(ch for ch in text if ch not in exclude)
        
    def white_space_fix(text):
        return ' '.join(text.split())
        
    def lower(text):
        return text.lower()
        
    return white_space_fix(remove_punc(lower(s)))

def exact_match_score(prediction, ground_truth):
    """
    Score based on string inclusion.
    If the target (ground_truth) exists in the prediction, score 1.0, else 0.0.
    """
    if not prediction or not ground_truth:
        return 0.0
    
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    
    # As long as the generated content contains the target entity, it is considered correct
    return 1.0 if norm_gt in norm_pred else 0.0

def evaluate_metrics(input_file, base_file=None, output_file=None, pre_dataset_file=None):
    """
    Read the xxx_results.json output by the editor and calculate scores.
    If base_file (pre-evaluated baseline results) is provided, extract its 'pre' data and stitch it in.
    If pre_dataset_file is provided, the target for 'pre' scoring will use the target answer from this file!
    Restructure it into a flat, clear tree structure.
    """
    if output_file is None:
        if "results.json" in input_file:
            output_file = input_file.replace("results.json", "metrics.json")
        elif "_results" in input_file:
            output_file = input_file.replace("_results", "_metrics")
        else:
            # Prevent overwriting the original file if 'results' is not in the name
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_metrics{ext}"

    if not os.path.exists(input_file):
        print(f"❌ Input file (test results) not found: {input_file}")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Load baseline evaluation data if base_file is provided
    base_data_map = {}
    if base_file:
        if os.path.exists(base_file):
            print(f"📄 Base model test results detected: {base_file}")
            with open(base_file, 'r', encoding='utf-8') as f:
                base_list = json.load(f)
                # Build a mapping with case_id as the key to extract the complete 'pre' dictionary
                base_data_map = {item.get("case_id", -1): item.get("pre", {}) for item in base_list}
        else:
            print(f"⚠️ Baseline input file not found: {base_file}, it will be ignored.")

    # Extract the Pre-edit Target if the original pre-edit dataset is provided
    pre_dataset_map = {}
    if pre_dataset_file:
        if os.path.exists(pre_dataset_file):
            print(f"📄 Loading Ground Truth from Pre-edit dataset: {pre_dataset_file}")
            with open(pre_dataset_file, 'r', encoding='utf-8') as f:
                raw_pre_ds = json.load(f)
                # Use the reliability prompt as a global unique identifier for matching to avoid index confusion caused by case_type filtering
                for item in raw_pre_ds:
                    p = item.get("metrics", {}).get("reliability", {}).get("prompt", "")
                    pre_dataset_map[p] = item
        else:
            print(f"⚠️ Pre dataset not found: {pre_dataset_file}")
        
    output_data = []
    
    for i, item in enumerate(data):
        case_id = item.get("case_id", i)
        req = item.get("requested_rewrite", {})
        post = item.get("post", {})
        
        # Extract Base data (including _ans for 7 metrics)
        if case_id in base_data_map and len(base_data_map[case_id]) > 0:
            pre = base_data_map[case_id]
        else:
            pre = item.get("pre", {})
            
        subject = req.get("subject", "")
        prompt = req.get("prompt", "")
        
        # Exact match: use the prompt to find the corresponding original item in pre_dataset_map
        pre_metrics = {}
        if prompt in pre_dataset_map:
            pre_metrics = pre_dataset_map[prompt].get("metrics", {})
        
        # Target used in the post stage (the knowledge intended after editing)
        post_target_new = req.get("target_new", "")
        
        # Target used in the pre stage (the original knowledge expected from the model before editing)
        if prompt in pre_dataset_map:
            pre_target_new = pre_dataset_map[prompt].get("original_object") or pre_dataset_map[prompt].get("new_object") or pre_dataset_map[prompt].get("new_counterfactual_object", "")
        else:
            # Fallback to post's target if not found
            pre_target_new = post_target_new
        
        # 1. Reliability
        pre_rewrite_ans = pre.get("rewrite_ans", "")
        post_rewrite_ans = post.get("rewrite_ans", "")
        
        pre_rewrite_acc = exact_match_score(pre_rewrite_ans, pre_target_new)
        post_rewrite_acc = exact_match_score(post_rewrite_ans, post_target_new)
        
        out_item = {
            "case_id": case_id,
            "subject": subject,
            "requested_rewrite": {
                "prompt": prompt,
                "target_new": post_target_new,
                "reliability": {
                    "pre_reliability_ans": pre_rewrite_ans,
                    "pre_reliability_acc": pre_rewrite_acc,
                    "post_reliability_ans": post_rewrite_ans,
                    "post_reliability_acc": post_rewrite_acc
                },
                "portability": {},
                "locality": {}
            }
        }
        
        # 2. Portability
        port_req = req.get("portability", {})
        for port_key, port_data in port_req.items():
            p_prompt = port_data.get("prompt", "")
            
            # Post target
            post_p_target = port_data.get("ground_truth", "")
            # Pre target
            pre_p_target = pre_metrics.get(port_key, {}).get("target", post_p_target)
            
            pre_ans = pre.get("portability", {}).get(f"{port_key}_ans", "")
            post_ans = post.get("portability", {}).get(f"{port_key}_ans", "")
            
            pre_acc = exact_match_score(pre_ans, pre_p_target)
            post_acc = exact_match_score(post_ans, post_p_target)
            
            out_item["requested_rewrite"]["portability"][port_key] = {
                "prompt": p_prompt,
                "target_ans": post_p_target,   # This is the final target after editing, for display only
                f"pre_{port_key}_ans": pre_ans,
                f"pre_{port_key}_acc": pre_acc,
                f"post_{port_key}_ans": post_ans,
                f"post_{port_key}_acc": post_acc
            }
            
        # 3. Locality
        loc_req = req.get("locality", {})
        for loc_key, loc_data in loc_req.items():
            l_prompt = loc_data.get("prompt", "")

            # Post target
            post_l_target = loc_data.get("ground_truth", "")
            # Pre target
            pre_l_target = pre_metrics.get(loc_key, {}).get("target", post_l_target)
            
            pre_ans = pre.get("locality", {}).get(f"{loc_key}_ans", "")
            post_ans = post.get("locality", {}).get(f"{loc_key}_ans", "")
            
            pre_acc = exact_match_score(pre_ans, pre_l_target)
            post_acc = exact_match_score(post_ans, post_l_target)
            
            out_item["requested_rewrite"]["locality"][loc_key] = {
                "prompt": l_prompt,
                "target_ans": post_l_target,   # This is the final target after editing, for display only
                f"pre_{loc_key}_ans": pre_ans,
                f"pre_{loc_key}_acc": pre_acc,
                f"post_{loc_key}_ans": post_ans,
                f"post_{loc_key}_acc": post_acc
            }
            
        output_data.append(out_item)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Evaluation and full restructuring complete!")
    print(f"📦 Metrics file merged with Base Pre data and real baseline Targets saved to: \n{output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TKE Metrics Evaluator")
    parser.add_argument("--input", type=str, default="../output_metrics/ft_gpt2-xl_hard_results.json", help="Path to the input results.json file")
    parser.add_argument("--base_input", type=str, default=None, help="Path to the baseline file of the base model (optional), e.g., base_gpt2-xl_hard_results.json")
    parser.add_argument("--pre_dataset", type=str, default=None, help="Pre-edit dataset (optional), provided to use its target for calculating 'pre' stage acc, e.g., dataset/tke_benchmark_pre_modify.json")
    parser.add_argument("--output", type=str, default=None, help="Path for the output metrics.json file (default: automatically replace suffix)")
    args = parser.parse_args()
    
    evaluate_metrics(args.input, args.base_input, args.output, args.pre_dataset)