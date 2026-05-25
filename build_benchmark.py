import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from tke_benchmark.dataset_builder import TKEBenchmarkBuilder

if __name__ == "__main__":
    
    # Specify the absolute or relative path to the pre-trained model.
    # We use this model to probe the knowledge base to ensure the dataset only contains facts the model already knows.
    MY_MODEL_PATH = "./huggingface_cache/models/Llama-3.2-3B-Instruct"
    
    # Initialize the automated dataset building engine
    builder = TKEBenchmarkBuilder(
        model_path=MY_MODEL_PATH,
        news_dir="dataset/news",
        output_dir="dataset/llama",
        batch_size=32  # Inference Batch Size. Reduce this if you encounter Out-Of-Memory errors on 3B+ models.
    )
    
    # Launch the end-to-end dataset generation pipeline
    builder.build_all(
        min_chain_length=3,         # Minimum number of consecutive events required for a valid entity chain
        max_cases_per_chain=32,     # Adjustable. Controls max soft cases sampled per chain. Higher values yield more total samples but may over-sample certain chains, reducing dataset uniformity.
        min_day_gap_insert=2,       # Minimum day interval between two events to allow a fictional counterfactual node insertion
        min_day_gap_modify=1        # Minimum day interval between three consecutive events required to modify the middle event
    )