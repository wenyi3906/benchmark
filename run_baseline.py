import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tke_benchmark.model_loader import setup_and_get_model
from tke_benchmark.dataser_loader import load_easyedit_dataset

def run_baseline(model_short='gpt2-xl', model_id='openai-community/gpt2-xl', 
                 cache_dir='./huggingface_cache', samples=5, device=0, case_type='all',
                 dataset_path='dataset/tke_benchmark.json'):
    """
    Evaluates the original knowledge of the pre-edited model using the pre_insert or pre_modify datasets
    to establish a baseline benchmark.

    Args:
        model_short (str): A shorthand name for the model, used for organizing and naming the output directories/files.
        model_id (str): The HuggingFace model ID or the local directory path to load the model from.
        cache_dir (str): The directory path used to cache downloaded HuggingFace models.
        samples (int): The number of samples to evaluate from the dataset. Set to -1 to evaluate all available samples.
        device (int): The ID of the GPU device to use for inference (e.g., 0 for 'cuda:0').
        case_type (str): The difficulty level of the test cases to filter by. Commonly 'all', 'Hard', or 'Soft'.
        dataset_path (str): The file path to the customized benchmark dataset (in JSON format).
    """
    print("="*50)
    print(f"🚀 Extracting base model Baseline data")
    print(f"   ➡️ Model:       {model_id}")
    print(f"   ➡️ Sample Size: {'All' if samples == -1 else samples}")
    print(f"   ➡️ Difficulty:  {case_type}")
    print(f"   ➡️ Dataset:     {dataset_path}")
    print("="*50)

    # 1. Automatically process paths and load model
    model_path = setup_and_get_model(
        model_id=model_id,
        base_dir=cache_dir
    )

    num_samples_to_load = None if samples == -1 else samples

    # 2. Read customized dataset
    prompts, ground_truth, target_new, subject, locality_inputs, portability_inputs = load_easyedit_dataset(
        file_path=dataset_path, 
        num_samples=num_samples_to_load,
        case_type=case_type
    )

    if not prompts and not locality_inputs['history_locality']['prompt']:
        print("❌ Dataset is empty or failed to successfully extract valid data!")
        return

    # 3. Load model
    print("⏳ Loading original unmodified model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = 'left'
    
    # Enable low CPU memory usage and do not force float16 to maintain original precision parity
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map={'': device}, 
        trust_remote_code=True, 
        low_cpu_mem_usage=True
    )
    model.eval()

    def generate_batch(prompts_list):
        if not prompts_list: return []
        inputs = tok(prompts_list, return_tensors='pt', padding=True, truncation=True).to(f"cuda:{device}")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=15,   # Control max generation length
                pad_token_id=tok.eos_token_id,
                do_sample=False
            )
        ans = [tok.decode(out[inputs['input_ids'].shape[1]:], skip_special_tokens=True) for out in outputs]
        return ans

    all_results = []
    
    is_insert = 'insert' in dataset_path.lower()
    if is_insert:
        print("\n🔥 Insert method detected, only generating answers for History and Future Locality...")
    else:
        print("\n🔥 Modify method detected, starting to generate answers for all 7 metrics...")
    
    # Compatible with empty prompts in insert mode, using history_locality length as reference
    loop_length = len(prompts) if prompts else len(locality_inputs['history_locality']['prompt'])
    
    for i in tqdm(range(loop_length)):
        batch_prompts = []
        
        if not is_insert:
            # Reliability prompt
            p_rewrite = prompts[i]
            if isinstance(p_rewrite, list): p_rewrite = p_rewrite[0]
            batch_prompts.append(p_rewrite)
                
            # Portability prompts
            port_keys = ['generality', 'temporal_reasoning']
            for k in port_keys:
                p = portability_inputs[k]['prompt'][i]
                if isinstance(p, list): p = p[0]
                batch_prompts.append(p)
                
            # Locality prompts
            loc_keys = ['history_locality', 'future_locality', 'temporal_hop_backward', 'temporal_hop_forward']
            for k in loc_keys:
                p = locality_inputs[k]['prompt'][i]
                if isinstance(p, list): p = p[0]
                batch_prompts.append(p)
        else:
            # Insert mode only retains two types of locality
            port_keys = []
            loc_keys = ['history_locality', 'future_locality']
            for k in loc_keys:
                p = locality_inputs[k]['prompt'][i]
                if isinstance(p, list): p = p[0]
                batch_prompts.append(p)
            
        # A single forward pass, batching all required prompts together
        ans_list = generate_batch(batch_prompts)
        
        port_dict = {}
        loc_dict = {}
        
        if not is_insert:
            rewrite_ans = ans_list[0]
            for idx, k in enumerate(port_keys):
                port_dict[f"{k}_ans"] = ans_list[1 + idx]
            for idx, k in enumerate(loc_keys):
                loc_dict[f"{k}_ans"] = ans_list[1 + len(port_keys) + idx]
                
            all_results.append({
                "case_id": i,
                "pre": {
                    "rewrite_ans": rewrite_ans,
                    "portability": port_dict,
                    "locality": loc_dict
                }
            })
        else:
            for idx, k in enumerate(loc_keys):
                loc_dict[f"{k}_ans"] = ans_list[idx]
                
            all_results.append({
                "case_id": i,
                "pre": {
                    "locality": loc_dict
                }
            })
        
    # Extract category based on dataset name, e.g., tke_benchmark_modify.json -> modify
    ds_basename = os.path.basename(dataset_path).replace(".json", "")
    counterfactual_type = ds_basename.replace("tke_benchmark_", "")
    if counterfactual_type == "tke_benchmark":
        counterfactual_type = "insert"
        
    result_dir = f"output_metrics/{model_short}/{counterfactual_type}/base/{case_type.lower()}_results.json"
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    with open(result_dir, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 Baseline benchmarking completed!")
    print(f"📦 Pure Baseline results saved to: {result_dir}")

if __name__ == "__main__":
    # ==========================================
    # Run mode selection: Set to "code" to pass args directly below, or "cli" to use command line args
    # ==========================================
    run_mode = "code"  

    if run_mode == "code":
        run_baseline(
            samples=-1,               # -1 to test all data
            model_short='gpt-j-6B',
            model_id='./huggingface_cache/models/gpt-j-6B',  # <--- Enter your local model path
            cache_dir='./huggingface_cache',
            case_type='all',
            dataset_path='dataset/gpt/tke_benchmark_pre_insert.json'
        )
    else:
        parser = argparse.ArgumentParser(description="TKE Baseline Runner")
        parser.add_argument('--model_short', type=str, default='gpt2-xl', help='Model abbreviation, used for naming output files')
        parser.add_argument('--model_id', type=str, default='openai-community/gpt2-xl', help='Model ID on HuggingFace or local path')
        parser.add_argument('--cache_dir', type=str, default='./huggingface_cache', help='Server cache folder')
        parser.add_argument('--samples', type=int, default=5, help='Number of samples to test, -1 means all')
        parser.add_argument('--device', type=int, default=0, help='GPU ID')
        parser.add_argument('--case_type', type=str, default='all', choices=['all', 'Hard', 'Soft'], help='Test case difficulty filter')
        parser.add_argument('--dataset_path', type=str, default='dataset/tke_benchmark.json', help='Dataset path')
        args = parser.parse_args()
        
        run_baseline(**vars(args))