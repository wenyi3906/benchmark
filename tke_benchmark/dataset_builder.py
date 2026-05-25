import os
import gc
import torch
from tke_benchmark.knowledge_prober import TemporalProber
from tke_benchmark.build_chains import build_temporal_chains
from tke_benchmark.generate_tke_benchmark import generate_test_cases, generate_benchmark

class TKEBenchmarkBuilder:
    """
    An automated pipeline engine for building the Temporal Knowledge Editing (TKE) Benchmark.
    This class encapsulates the entire data generation process: from probing raw knowledge 
    in the baseline model to formulating the final formatted benchmark datasets.
    """
    
    def __init__(self, model_path: str, news_dir: str = "dataset/news", output_dir: str = "dataset/llama", batch_size: int = 64):
        """
        Initializes the TKE Benchmark Builder.

        Args:
            model_path (str): The absolute or relative path to the pre-trained causal language model 
                              (e.g., LLaMA, GPT-2) used for knowledge probing.
            news_dir (str): The directory containing the raw source data. Must include 'news_full.txt' 
                            and 'relations_map_full.json'.
            output_dir (str): The directory where the generated intermediate files and final JSON 
                              benchmarks will be saved.
            batch_size (int): The batch size used for inference during the knowledge probing stage. 
                              Reduce this if Out-Of-Memory (OOM) errors occur.
        """
        self.model_path = model_path
        self.news_dir = news_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define input file paths
        self.map_path = os.path.join(self.news_dir, "relations_map_full.json")
        self.input_data = os.path.join(self.news_dir, "news_full.txt")
        
        # Define intermediate output file paths
        self.known_facts_path = os.path.join(self.output_dir, "known_pure_facts.json")
        self.chains_path = os.path.join(self.output_dir, "temporal_chains.json")
        
        self.cases_insert_path = os.path.join(self.output_dir, "tke_test_cases_insert.json")
        self.cases_modify_path = os.path.join(self.output_dir, "tke_test_cases_modify.json")
        
        # Define final benchmark file paths
        self.benchmark_insert_path = os.path.join(self.output_dir, "tke_benchmark_insert.json")
        self.benchmark_modify_path = os.path.join(self.output_dir, "tke_benchmark_modify.json")
        
        # Define baseline evaluating file paths (testing before editing)
        self.benchmark_pre_insert_path = os.path.join(self.output_dir, "tke_benchmark_pre_insert.json")
        self.benchmark_pre_modify_path = os.path.join(self.output_dir, "tke_benchmark_pre_modify.json")

    def build_all(self, min_chain_length: int = 3, max_cases_per_chain: int = 1, min_day_gap_insert: int = 2, min_day_gap_modify: int = 1):
        """
        Executes the entire dataset construction pipeline.

        Args:
            min_chain_length (int): The minimum number of consecutive events required for an entity 
                                    to form a valid temporal chain.
            max_cases_per_chain (int): The maximum number of 'Soft' cases (unaltered facts) to sample 
                                       per temporal chain. Setting this low (e.g., 1) maximizes the 
                                       purity of 'Hard' cases (counterfactual object shifts) in the dataset.
            min_day_gap_insert (int): The minimum number of days between two events to allow a 
                                      fictional counterfactual node to be inserted.
            min_day_gap_modify (int): The minimum number of days between three consecutive events 
                                      required to safely modify the middle event's object.
        """
        print("=" * 60)
        print("🚀 [TKE Builder] Starting Temporal Knowledge Editing Benchmark Construction")
        print(f"   ➡️ Base Model:  {self.model_path}")
        print(f"   ➡️ Source Dir:  {self.news_dir}")
        print(f"   ➡️ Output Dir:  {self.output_dir}")
        print("=" * 60)
        
        # 1. Instantiate Prober and execute knowledge probing
        print("\n[1/4] Probing model's known facts...")
        prober = TemporalProber(model_path=self.model_path, map_path=self.map_path)
        prober.run_probing_batched(data_file=self.input_data, output_file=self.known_facts_path, batch_size=self.batch_size)
        
        # Free up GPU memory promptly after probing to prevent OOM in subsequent parallel tasks
        del prober.model
        torch.cuda.empty_cache()
        gc.collect()

        # 2. Build temporal chains
        print(f"\n[2/4] Constructing temporal chains (Min Chain Length: {min_chain_length})...")
        build_temporal_chains(self.known_facts_path, self.chains_path, min_chain_length)

        # 3. Generate test cases (filtering hard and soft samples)
        print("\n[3/4] Extracting counterfactual use cases from temporal chains (Insert / Modify)...")
        generate_test_cases(
            self.chains_path, 
            self.cases_insert_path, 
            self.cases_modify_path, 
            max_cases=max_cases_per_chain, 
            min_day_gap_insert=min_day_gap_insert, 
            min_day_gap_modify=min_day_gap_modify
        )

        # 4. Assemble the final TKE Benchmark formats
        print("\n[4/4] Assembling professional-grade TKE Benchmark datasets...")
        
        # Generate Insert method benchmarks
        generate_benchmark(
            test_cases_path=self.cases_insert_path,
            map_path=self.map_path,
            output_path=self.benchmark_insert_path,
            pre_insert_output_path=self.benchmark_pre_insert_path
        )

        # Generate Modify method benchmarks
        generate_benchmark(
            test_cases_path=self.cases_modify_path,
            map_path=self.map_path,
            output_path=self.benchmark_modify_path,
            pre_modify_output_path=self.benchmark_pre_modify_path
        )

        print("\n" + "=" * 60)
        print("🎉 TKE Benchmark dataset construction completed successfully!")
        print(f"📦 All data files have been saved to: {self.output_dir}/")
        print("=" * 60)