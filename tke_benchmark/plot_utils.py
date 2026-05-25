import os
import json
import matplotlib.pyplot as plt
import numpy as np

def save_metrics(metrics, filepath):
    """
    Save EasyEdit metrics to a local JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        else:
            return obj
            
    serializable_metrics = convert_to_serializable(metrics)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
    print(f"✅ Evaluation results successfully saved to: {filepath}")

def _extract_mean(val):
    if isinstance(val, list):
        if len(val) == 0:
            return 0.0
        try:
            return float(np.mean(val))
        except Exception:
            return 0.0
    elif isinstance(val, (float, int, np.floating, np.integer)):
        return float(val)
    else:
        return 0.0

def _get_new_metric(metrics_data, phase, category, subcategory=None):
    """
    Extract metrics from the tree-structured metrics.json file converted by evaluate_metrics.py.
    Format: item["requested_rewrite"][category][subcategory][f"{phase}_{subcategory}_acc"]
    Or for reliability: item["requested_rewrite"]["reliability"][f"{phase}_reliability_acc"]
    """
    if isinstance(metrics_data, list):
        vals = []
        for m in metrics_data:
            req = m.get("requested_rewrite", {})
            cat_data = req.get(category, {})
            
            if category == "reliability":
                # Reliability is flat
                val = cat_data.get(f"{phase}_reliability_acc", None)
            else:
                if subcategory:
                    val = cat_data.get(subcategory, {}).get(f"{phase}_{subcategory}_acc", None)
                else:
                    val = None
                    
            if val is not None:
                vals.append(_extract_mean(val))
        return float(np.mean(vals)) if len(vals) > 0 else 0.0
    return 0.0

def plot_metrics(metrics_filepath, title=None):
    """
    Read metrics.json and plot the accuracy comparison before and after EasyEdit editing.
    """
    if not os.path.exists(metrics_filepath):
        print(f"❌ File not found: {metrics_filepath}")
        return
        
    with open(metrics_filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    print(f"✅ Successfully read metrics data from {metrics_filepath} for plotting!")
    
    if title is None:
        # Attempt to extract names from the new hierarchical directory structure (e.g.: output_metrics/gpt2-xl/modify/ft/hard_metrics.json)
        path_parts = os.path.normpath(metrics_filepath).split(os.sep)
        if len(path_parts) >= 4:
            model_name = path_parts[-4]
            method = path_parts[-2].upper()
            case_type = path_parts[-1].split('_')[0].capitalize()
            title = f"{method} ({model_name}) Editing Evaluation - {case_type}"
        else:
            title = "Knowledge Editing Evaluation"
            
    auto_num_samples = len(metrics) if isinstance(metrics, list) else 0
    
    categories = [
        'Reliability Acc',
        'Generality Acc (Portability)',
        'Temporal Reason (Portability)',
        'History Score (Locality)',
        'Future Score (Locality)',
        'Hop Backward (Locality)',
        'Hop Forward (Locality)'
    ]
    
    # Extract Pre-Edit metrics
    pre_values = [
        _get_new_metric(metrics, 'pre', 'reliability'),
        _get_new_metric(metrics, 'pre', 'portability', 'generality'),
        _get_new_metric(metrics, 'pre', 'portability', 'temporal_reasoning'),
        _get_new_metric(metrics, 'pre', 'locality', 'history_locality'),
        _get_new_metric(metrics, 'pre', 'locality', 'future_locality'),
        _get_new_metric(metrics, 'pre', 'locality', 'temporal_hop_backward'),
        _get_new_metric(metrics, 'pre', 'locality', 'temporal_hop_forward')
    ]
    
    # Extract Post-Edit metrics
    post_values = [
        _get_new_metric(metrics, 'post', 'reliability'),
        _get_new_metric(metrics, 'post', 'portability', 'generality'),
        _get_new_metric(metrics, 'post', 'portability', 'temporal_reasoning'),
        _get_new_metric(metrics, 'post', 'locality', 'history_locality'),
        _get_new_metric(metrics, 'post', 'locality', 'future_locality'),
        _get_new_metric(metrics, 'post', 'locality', 'temporal_hop_backward'),
        _get_new_metric(metrics, 'post', 'locality', 'temporal_hop_forward')
    ]

    # --- Print Markdown Table ---
    print("\n" + "="*60)
    print(f"📊 Editing Evaluation Table / {method} ({model_name})  - {case_type} (n={auto_num_samples})")
    print("="*60)
    print(f"| {'Metric':<30} | {'Pre-Edit':<18} | {'Post-Edit':<18} |")
    print(f"|{'-'*32}|{'-'*20}|{'-'*20}|")
    for cat, pre_val, post_val in zip(categories, pre_values, post_values):
        print(f"| {cat:<30} | {pre_val:<18.4f} | {post_val:<18.4f} |")
    print("="*60 + "\n")
    # ----------------------------

    # --- Plot Bar Chart ---
    plot_categories = [c.replace(' (', '\n(') for c in categories]
    
    x = np.arange(len(plot_categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    rects1 = ax.bar(x - width/2, pre_values, width, label='Pre-Edit', color='#87CEEB', edgecolor='black')
    rects2 = ax.bar(x + width/2, post_values, width, label='Post-Edit', color='#FF9999', edgecolor='black')
    
    ax.set_ylabel('Score / Accuracy', fontsize=12)
    ax.set_title(f'{title} (n={auto_num_samples})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_categories, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1) 
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0: 
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
                            
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.show()