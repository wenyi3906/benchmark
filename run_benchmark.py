import os
import pickle
from sentence_transformers import SentenceTransformer
import argparse
import easyeditor
from easyeditor import BaseEditor
from tke_benchmark.model_loader import setup_and_get_model
from tke_benchmark.dataser_loader import load_easyedit_dataset
from tke_benchmark.plot_utils import save_metrics
import torch

def run_benchmark(method='ROME', model_short='gpt2-xl', model_id='openai-community/gpt2-xl', 
                  cache_dir='./huggingface_cache', samples=5, device=0, case_type='all', 
                  eval_pre_locality=True, dataset_path='dataset/tke_benchmark.json',
                  start_idx=0, end_idx=-1):
    """
    Runs the knowledge editing benchmark for a given algorithm and model.

    Args:
        method (str): The name of the knowledge editing algorithm to test (e.g., 'ROME', 'IKE', 'MEND').
        model_short (str): A shorthand name for the model, used for organizing outputs and locating YAML configurations.
        model_id (str): The HuggingFace model ID or the local directory path to load the model from.
        cache_dir (str): The directory path used to cache downloaded HuggingFace models.
        samples (int): The number of samples to evaluate from the dataset. Set to -1 to evaluate all available samples.
        device (int): The ID of the GPU device to use for inference (e.g., 0 for 'cuda:0').
        case_type (str): The difficulty level of the test cases to filter by. Commonly 'all', 'Hard', or 'Soft'.
        eval_pre_locality (bool): Whether to evaluate locality metrics before applying the edit. Setting to False speeds up evaluation.
        dataset_path (str): The file path to the customized benchmark dataset (in JSON format).
        start_idx (int): The starting index for data sharding, useful for parallel evaluation across multiple processes.
        end_idx (int): The ending index for data sharding. Set to -1 to run until the end of the dataset.
    """
    print("="*50)
    print(f"🚀 TKE Knowledge Editing Benchmark Started!")
    print(f"   ➡️ Algorithm:         {method}")
    print(f"   ➡️ Model:             {model_id}")
    print(f"   ➡️ Total Samples:     {'all' if samples == -1 else samples}")
    print(f"   ➡️ Shard Range:       [{start_idx} : {end_idx if end_idx != -1 else 'END'}]")
    print(f"   ➡️ Difficulty:        {case_type}")
    print(f"   ➡️ Dataset:           {dataset_path}")
    print(f"   ➡️ Eval Pre-Locality: {'yes' if eval_pre_locality else 'no (post only)'}")
    print("="*50)

    # 1. Automatically process paths and load model
    model_path = setup_and_get_model(
        model_id=model_id,
        base_dir=cache_dir
    )

    # If samples parameter is -1, load all data (usually represented by None in the data loading function)
    num_samples_to_load = None if samples == -1 else samples

    # 2. Read our customized dataset
    print(f"📂 Reading dataset: {dataset_path}")
    prompts, ground_truth, target_new, subject, locality_inputs, portability_inputs = load_easyedit_dataset(
        file_path=dataset_path, 
        num_samples=num_samples_to_load,
        case_type=case_type
    )

    if not prompts:
        print("❌ Dataset is empty or failed to successfully extract valid data!")
        return
        
    # ==========================================
    # Data Sharding Logic
    # ==========================================
    if end_idx == -1 or end_idx > len(prompts):
        end_idx = len(prompts)
        
    prompts = prompts[start_idx:end_idx]
    if ground_truth is not None:
        ground_truth = ground_truth[start_idx:end_idx]
    target_new = target_new[start_idx:end_idx]
    subject = subject[start_idx:end_idx]
    
    for k in locality_inputs.keys():
        locality_inputs[k]['prompt'] = locality_inputs[k]['prompt'][start_idx:end_idx]
        locality_inputs[k]['ground_truth'] = locality_inputs[k]['ground_truth'][start_idx:end_idx]
        
    for k in portability_inputs.keys():
        portability_inputs[k]['prompt'] = portability_inputs[k]['prompt'][start_idx:end_idx]
        portability_inputs[k]['ground_truth'] = portability_inputs[k]['ground_truth'][start_idx:end_idx]
        
    print(f"✅ Data sharding successful! Effective number of questions for current process: {len(prompts)} (from index {start_idx} to {end_idx})")

    # 3. Dynamic configuration (No more hardcoding imports)
    hparams_class_name = f"{method}HyperParams"
    if method == 'GRACE':
        hparams_class_name = 'GraceHyperParams'
    if not hasattr(easyeditor, hparams_class_name):
        raise ValueError(f"❌ Class {hparams_class_name} not found, please check if --method parameter spelling is correct!")
    HParamsClass = getattr(easyeditor, hparams_class_name)

    # Smartly locate YAML configuration file
    folder_name = method.replace("_", "-")
    yaml_path = f'./hparams1/{folder_name}/{model_short}'
    if not os.path.exists(yaml_path + ".yaml"):
        yaml_path = f'./hparams/{folder_name}/{model_short}'
    if not os.path.exists(yaml_path + ".yaml"):
        raise FileNotFoundError(f"❌ Corresponding YAML configuration file not found! Please check if {model_short}.yaml exists in the hparams directory.")

    print(f"📄 Successfully locked parameter configuration file: {yaml_path}.yaml")
    hparams = HParamsClass.from_hparams(yaml_path)
    
    # Force overwrite model path and GPU ID to current server environment
    hparams.model_name = model_path
    hparams.device = device

    # 4. Start the editor!
    editor = BaseEditor.from_hparams(hparams)
    
    # Enable low CPU memory usage and do not force float16 to maintain original precision parity
    # if 'llama' in model_short.lower() or 'qwen' in model_short.lower():
    #     editor.model.to(torch.bfloat16)

    print(f"\n🔥 Starting execution of [{method}] core algorithm...")

    # Prepare parameter dictionary for editor.edit
    edit_kwargs = {
        "prompts": prompts,
        "ground_truth": ground_truth,
        "target_new": target_new,
        "subject": subject,
        "locality_inputs": locality_inputs,
        "portability_inputs": portability_inputs,
        "eval_pre_locality": eval_pre_locality,
        # Supplement loc_prompts for methods like WISE, MEND that require irrelevant text to calculate generalization loss
        "loc_prompts": locality_inputs.get('history_locality', {}).get('prompt', ["This is a random sentence."] * len(prompts))
    }

    # Build and add train_ds only when the algorithm is IKE
    if method == 'IKE':
        print("ℹ️ IKE algorithm detected, building context knowledge base (train_ds)...")
        train_ds = [{"prompt": p, "target_new": t} for p, t in zip(prompts, target_new)]
        edit_kwargs['train_ds'] = train_ds

        # --- New: Auto-detect and generate Embedding cache files required by IKE ---
        safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
        embedding_dir = f"{hparams.results_dir}/{hparams.alg_name}/embedding"
        os.makedirs(embedding_dir, exist_ok=True)
        pkl_path = f"{embedding_dir}/{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl"

        if not os.path.exists(pkl_path):
            print(f"⚠️ IKE cache file not found, automatically generating Embeddings: {pkl_path}")
            # Concatenate prompt and target into complete declarative sentences for encoding
            sentences = [f"{item['prompt']} {item['target_new']}" for item in train_ds]
            print("Loading SentenceTransformer model...")
            sentence_model = SentenceTransformer(hparams.sentence_model_name).to(
                f"cuda:{hparams.device}" if str(hparams.device).isdigit() else "cpu")
            print("Encoding knowledge base vectors...")
            embeddings = sentence_model.encode(sentences, show_progress_bar=True)
            with open(pkl_path, "wb") as fOut:
                pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ Embeddings cache file generation completed!")
        # --------------------------------------------------------

    metrics, edited_model, _ = editor.edit(**edit_kwargs)

    # 5. Automatically archive results
    ds_basename = os.path.basename(dataset_path).replace(".json", "")
    counterfactual_type = ds_basename.replace("tke_benchmark_", "")
    if counterfactual_type == "tke_benchmark":
        counterfactual_type = "insert"
        
    # If sharding is enabled, mark it on the filename
    slice_suffix = "" if (start_idx == 0 and end_idx == len(prompts)) else f"_part_{start_idx}_{end_idx}"
    
    result_dir = f"output_metrics/{model_short}/{counterfactual_type}/{method.lower()}/{case_type.lower()}{slice_suffix}_results.json"
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    save_metrics(metrics, result_dir)
    print("\n" + "="*50)
    print(f"🎉 Evaluation for current shard ({start_idx} - {end_idx}) successfully completed!")
    print(f"📦 Results archived to: {result_dir}")
    print("="*50)
    
if __name__ == "__main__":
    # ==========================================
    # Run mode selection: Set to "code" to pass args directly below, or "cli" to use command line args
    # ==========================================
    run_mode = "code"  

    if run_mode == "code":
        # 💻 Method 1: Direct argument passing (Recommended in IDEs like PyCharm, modify here)
        run_benchmark(
            method='WISE',
            samples= -1,              # Total extracted data volume (e.g., -1 for all, or 50 for first 50 items)
            model_short='gpt-j-6B',
            model_id='./huggingface_cache/models/gpt-j-6B',
            cache_dir='./huggingface_cache',
            case_type='all',
            eval_pre_locality=False,  
            dataset_path='dataset/gpt/tke_benchmark_insert.json',
            start_idx=0,              # <--- Modify here! Starting index for current process (0-indexed)
            end_idx=-1                # <--- Modify here! Ending index for current process (-1 means to the end)
        )
    else:
        # ⌨️ Method 2: Command-line argument parsing (Compatible with original terminal call)
        parser = argparse.ArgumentParser(description="TKE Knowledge Editing Benchmark Runner (AutoDL Ready)")
        parser.add_argument('--method', type=str, default='ROME', help='Name of the algorithm you want to test')
        parser.add_argument('--model_short', type=str, default='gpt2-xl', help='Model abbreviation, used to find YAML configuration files')
        parser.add_argument('--model_id', type=str, default='openai-community/gpt2-xl', help='Model ID on HuggingFace or local path')
        parser.add_argument('--cache_dir', type=str, default='./huggingface_cache', help='Server cache folder')
        parser.add_argument('--samples', type=int, default=5, help='Total number of samples to test, -1 means all')
        parser.add_argument('--device', type=int, default=0, help='GPU ID')
        parser.add_argument('--case_type', type=str, default='all', choices=['all', 'Hard', 'Soft'], help='Test case difficulty filter')
        parser.add_argument('--dataset_path', type=str, default='dataset/tke_benchmark.json', help='Dataset path')
        parser.add_argument('--start_idx', type=int, default=0, help='Shard testing: Starting data index for the current process')
        parser.add_argument('--end_idx', type=int, default=-1, help='Shard testing: Ending data index for the current process (-1 means to the end)')
        
        # Helper method to convert string "True"/"False" to boolean values
        def str2bool(v):
            if isinstance(v, bool): return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
            else: raise argparse.ArgumentTypeError('Boolean value expected.')
            
        parser.add_argument('--eval_pre_locality', type=str2bool, default=True, help='Whether to evaluate locality before editing (turning it off can greatly accelerate evaluation)')
        args = parser.parse_args()
        
        # vars() will convert args into a dictionary and automatically pass corresponding parameters
        run_benchmark(**vars(args))