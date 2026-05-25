# Temporal Knowledge Editing (TKE) Benchmark

This project integrates the [EasyEdit](https://github.com/zjunlp/EasyEdit) framework to provide a complete end-to-end pipeline for constructing and evaluating the Temporal Knowledge Editing (TKE) Benchmark.

Below are the complete steps for setting up the environment, building data, evaluating models, and visualizing the results.

## 1. Environment Setup

It is recommended to use `conda` to create an isolated Python virtual environment and install dependencies using the `requirements.txt` file in the project's root directory.

You can create the environment in one step (if `requirements.txt` is compatible with conda):
```bash
conda create --name tke_env --file requirements.txt
conda activate tke_env
```
Alternatively, create the environment first and install using pip:
```bash
# 1. Create and activate the virtual environment (Python 3.9+ recommended)
conda create -n tke_env python=3.10 -y
conda activate tke_env

# 2. Install dependencies
pip install -r requirements.txt
```
*(Note: If you need to use specific algorithms from EasyEdit, you might need to install additional dependencies according to the official EasyEdit documentation.)*

## 2. Prepare Models

For network stability and management convenience, this framework recommends downloading the Large Language Models locally. Please prepare the weights of the models you wish to evaluate and edit (e.g., Llama, Qwen, ChatGLM) and store them in the `huggingface_cache/models/` directory.

**Example Directory Structure:**
```text
TKE-benchmark/
├── huggingface_cache/
│   └── models/
│       ├── Llama-2-7b-chat-hf/
│       └── ...
├── build_benchmark.py
├── run_baseline.py
└── ...
```

## 3. Build Benchmark

Use the `build_benchmark.py` script to construct the dataset required for evaluation. This step involves cleaning and restructuring the raw data to generate a standard format Benchmark file for editing and testing. During dataset construction, the generation logic for counterfactual nodes is divided into two types: the **"modify"** approach and the **"insert"** approach.

```bash
python build_benchmark.py
```
*(You can specify input and output file paths or other configurations within the script or via command-line arguments.)*

## 4. Test Pre-Edit Model (Baseline)

Before applying any knowledge editing, you need to assess the performance of the original, unedited model on the Benchmark (i.e., the Baseline). Run the `run_baseline.py` script to perform this test.

```bash
python run_baseline.py
```
After execution, the script will save the model's test results on the original dataset as a JSON file (typically in the `results/` directory), which will be used for subsequent evaluation comparisons.

## 5. Evaluate Knowledge Editing Methods

Based on the constructed Benchmark, use `run_benchmark.py` to call EasyEdit's underlying interfaces to execute specific knowledge editing methods (such as ROME, MEMIT, MEND, etc.).

```bash
python run_benchmark.py
```
This step will modify the model's weights or representations, run inference on various generalization and locality questions using the modified model, and save the final edited prediction results as a JSON file.

## 6. Metrics Calculation and Evaluation

Once the runs are complete, you will have a pre-edit JSON file and a post-edit JSON file. Refer to the implementation in `evaluator.ipynb` and call `evaluate_metrics` to compare the differences between the two files to calculate various knowledge editing metrics (e.g., Reliability, Generalization, Locality).

**Code Example:**
```python
# Can be run in a script or Jupyter Notebook
from tke_benchmark.evaluate_metrics import evaluate_metrics

# Evaluate metrics by combining pre-edit and post-edit JSON files
evaluate_metrics(
    # input_file: The prediction results of the post-edit model on the benchmark (constructed via the "modify" approach here)
    input_file="output_metrics/llama3.2-3b/modify/ft/all_results.json",
    # base_file: The prediction results of the pre-edit model (Baseline) on the corresponding benchmark
    base_file="output_metrics/llama3.2-3b/pre_modify/base/all_results.json",
    # pre_dataset_file: The original dataset file used to construct the benchmark, for reference and comparison
    pre_dataset_file="dataset/llama/tke_benchmark_pre_modify.json"
)
```
*(Note: `pre_modify` indicates the evaluation of the pre-edit model using the dataset constructed with the "modify" method. Correspondingly, there will be `pre_insert` for results evaluated with the "insert" method.)*

## 7. Visualize Statistical Results

Finally, refer to the logic in `plot.ipynb` and use the `plot_metrics` function to generate visualization charts (such as bar charts or line graphs) from the evaluation results of various methods for analysis or use in paper reports.

Please note that the directory structure `output_metrics/llama3.2-3b/modify/ft/all_metrics.json` follows a specific classification logic for organizing experiments and enabling comparisons:

* `output_metrics/`: The root directory for storing all evaluation metrics and visualization results.
* `llama3.2-3b/`: Represents the specific model being evaluated (e.g., Llama3.2-3b, Qwen-7b, etc.).
* `modify/`: Represents the generation logic of counterfactual nodes when building the TKE dataset, which is the **"modify"** method here. If the insert method is used, this would be **`insert/`** (and the corresponding pre-edit model evaluation directories would be `pre_modify/` and `pre_insert/`).
* `ft/`: Represents the specific knowledge editing method. For example, Fine-Tuning (ft) is used here, but it could also be ROME, MEMIT, MEND, etc.
* `all_metrics.json`: The specific evaluation metrics statistics file generated by the `evaluate_metrics` mentioned above.

**Code Example:**
```python
# Can be run in a script or Jupyter Notebook
from tke_benchmark.plot_utils import plot_metrics

# Plot statistical charts
plot_metrics("output_metrics/llama3.2-3b/modify/ft/all_metrics.json")
```
