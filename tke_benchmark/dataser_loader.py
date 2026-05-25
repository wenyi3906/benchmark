import json

def load_easyedit_dataset(file_path, num_samples=None, case_type='all'):
    """
    Load and parse customized datasets to return the format required by EasyEdit.
    Compatible with insert, modify, and pre_modify formatted data.

    Args:
        file_path (str): The path to the JSON dataset file.
        num_samples (int, optional): Number of samples to extract. Defaults to None (extract all).
        case_type (str): The difficulty level of the test cases to filter by (e.g., 'Hard', 'Soft', or 'all').
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 1. First filter the data based on case_type
    if case_type != 'all':
        raw_data = [item for item in raw_data if item.get("case_type") == case_type]

    # 2. Then slice the list according to num_samples
    if num_samples is not None:
        raw_data = raw_data[:num_samples]

    prompts = []
    ground_truth = None
    target_new = []
    subject = []

    locality_inputs = {
        'history_locality': {'prompt': [], 'ground_truth': []},
        'future_locality': {'prompt': [], 'ground_truth': []},
        'temporal_hop_backward': {'prompt': [], 'ground_truth': []},
        'temporal_hop_forward': {'prompt': [], 'ground_truth': []}
    }

    portability_inputs = {
        'generality': {'prompt': [], 'ground_truth': []},
        'temporal_reasoning': {'prompt': [], 'ground_truth': []}
    }

    for item in raw_data:
        metrics = item["metrics"]

        # 1. Core Editing Data
        prompts.append(metrics["reliability"]["prompt"])
        
        # Ensure compatibility across the three different data dictionary structures
        target_val = item.get("new_object") or item.get("new_counterfactual_object") or item.get("original_object", "")
        target_new.append(target_val)
        
        subject.append(item["subject"])

        # 2. Locality Testing (Locality)
        locality_inputs['history_locality']['prompt'].append(metrics["history_locality"]["prompt"])
        locality_inputs['history_locality']['ground_truth'].append(metrics["history_locality"]["target"])

        locality_inputs['future_locality']['prompt'].append(metrics["future_locality"]["prompt"])
        locality_inputs['future_locality']['ground_truth'].append(metrics["future_locality"]["target"])

        locality_inputs['temporal_hop_backward']['prompt'].append(metrics["temporal_hop_backward"]["prompt"])
        locality_inputs['temporal_hop_backward']['ground_truth'].append(metrics["temporal_hop_backward"]["target"])

        locality_inputs['temporal_hop_forward']['prompt'].append(metrics["temporal_hop_forward"]["prompt"])
        locality_inputs['temporal_hop_forward']['ground_truth'].append(metrics["temporal_hop_forward"]["target"])

        # 3. Generalization Testing (Portability)
        portability_inputs['generality']['prompt'].append(metrics["generality"]["prompt"])
        portability_inputs['generality']['ground_truth'].append(metrics["generality"]["target"])

        portability_inputs['temporal_reasoning']['prompt'].append(metrics["temporal_reasoning"]["prompt"])
        portability_inputs['temporal_reasoning']['ground_truth'].append(metrics["temporal_reasoning"]["target"])

    return prompts, ground_truth, target_new, subject, locality_inputs, portability_inputs