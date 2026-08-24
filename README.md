# TRACE: A Fine-Grained Benchmark for Temporal Reasoning and Chronological Knowledge Editing over Large Language Models

This repository contains the official implementation of **TRACE** (**T**emporal **R**easoning **A**nd **C**hronological **E**diting), a fine-grained benchmark for evaluating temporal reasoning and chronological knowledge editing in large language models.

TRACE constructs model-specific chronological event chains from ICEWS events spanning 2005–2015. It uses exact-match *a priori* probing to retain facts recalled by each target model and supports two editing settings:

- **Modify:** replace the object of an existing event.
- **Insert:** inject a new counterfactual event into an established timeline.

Each edit is evaluated along seven dimensions:

1. Reliability
2. Generality
3. History
4. Future
5. Temporal Reasoning
6. Backward Hop
7. Forward Hop

The experiments reported in the paper evaluate six knowledge-editing methods—**FT, IKE, ROME, GRACE, LoRA, and WISE**—on **Llama-3.2-3B** and **GPT-J-6B**. The implementation builds on the [EasyEdit](https://github.com/zjunlp/EasyEdit) framework.

## Repository Structure

```text
TRACE/
├── dataset/                 # TRACE benchmark data
├── huggingface_cache/       # Optional local model storage
├── output_metrics/          # Predictions and evaluation results
├── build_benchmark.py       # Benchmark construction
├── run_baseline.py          # Pre-edit evaluation
├── run_benchmark.py         # Knowledge-editing experiments
├── evaluator.ipynb          # Metric calculation and analysis
├── plot.ipynb               # Result visualization
├── requirements.txt
└── README.md
```

The exact directory structure may vary slightly across experiment configurations.

## 1. Environment Setup

We recommend using Conda with Python 3.10:

```bash
conda create -n trace python=3.10 -y
conda activate trace
pip install -r requirements.txt
```

Some editing methods may require additional dependencies. Refer to the [EasyEdit documentation](https://github.com/zjunlp/EasyEdit) if a method-specific dependency is missing.

## 2. Prepare the Models

Download the model weights required for evaluation and configure their local paths in the corresponding experiment configuration files.

The experiments in the paper use:

- Llama-3.2-3B
- GPT-J-6B

An optional local layout is:

```text
TRACE/
└── huggingface_cache/
    └── models/
        ├── Llama-3.2-3B/
        └── GPT-J-6B/
```

Model weights are not included in this repository. Please follow the licensing and access requirements of the respective model providers.

## 3. Build the Benchmark

Run the benchmark-construction pipeline with:

```bash
python build_benchmark.py
```

The construction process performs the following main steps:

1. Extract temporal facts from ICEWS.
2. Probe the target model for pre-existing knowledge.
3. Construct chronological event chains from recalled facts.
4. Sample counterfactual targets using relation-constrained pools.
5. Generate evaluation instances for the Modify and Insert settings.
6. Produce prompts for the seven TRACE evaluation dimensions.

Because TRACE applies model-specific *a priori* probing, separate benchmark variants are generated for different target models.

## 4. Evaluate the Pre-Edit Model

Before applying a knowledge edit, evaluate the original model on the corresponding TRACE prompts:

```bash
python run_baseline.py
```

The resulting predictions provide the pre-edit reference used in subsequent metric calculation. Modify and Insert use their corresponding pre-edit evaluation sets.

## 5. Run Knowledge-Editing Experiments

Run an editing experiment with:

```bash
python run_benchmark.py
```

The paper evaluates the following methods:

- FT
- IKE
- ROME
- GRACE
- LoRA
- WISE

Method- and model-specific settings are stored in the corresponding configuration files. The EasyEdit snapshot and configurations used for the reported experiments are included in this repository to support reproducibility.

## 6. Calculate Evaluation Metrics

After obtaining the pre-edit and post-edit predictions, use the evaluation implementation in `evaluator.ipynb` to calculate the TRACE metrics.

Example:

```python
from tke_benchmark.evaluate_metrics import evaluate_metrics

evaluate_metrics(
    input_file="output_metrics/llama3.2-3b/modify/ft/all_results.json",
    base_file="output_metrics/llama3.2-3b/pre_modify/base/all_results.json",
    pre_dataset_file="dataset/llama/tke_benchmark_pre_modify.json",
)
```

In this example:

- `input_file` contains post-edit predictions.
- `base_file` contains predictions from the original model.
- `pre_dataset_file` contains the corresponding pre-edit benchmark instances.

Use the matching `pre_modify` or `pre_insert` results for each experimental setting.

## 7. Visualize the Results

The visualization code is provided in `plot.ipynb`.

Example:

```python
from tke_benchmark.plot_utils import plot_metrics

plot_metrics(
    "output_metrics/llama3.2-3b/modify/ft/all_metrics.json"
)
```

Experiment outputs follow a hierarchical organization such as:

```text
output_metrics/
└── llama3.2-3b/
    ├── pre_modify/
    ├── pre_insert/
    ├── modify/
    │   ├── ft/
    │   ├── ike/
    │   ├── rome/
    │   ├── grace/
    │   ├── lora/
    │   └── wise/
    └── insert/
        ├── ft/
        ├── ike/
        ├── rome/
        ├── grace/
        ├── lora/
        └── wise/
```

Each method directory may contain raw predictions, per-instance results, aggregate metrics, and visualization outputs.

## Reproducibility

The repository includes the benchmark-construction pipeline, evaluation code, model-specific configurations, and the EasyEdit snapshot used in the paper.

Before reproducing an experiment, verify:

- the target-model path;
- the selected TRACE split;
- the editing-method configuration;
- the output directory;
- the device setting; and
- the required model and dataset licenses.

The paper reports experiments conducted on Ubuntu 22.04 with CUDA 12.8 and a single NVIDIA RTX 5090 GPU with 32 GB of VRAM.

## Citation

If you use TRACE in your research, please cite our paper:

```bibtex
@inproceedings{liu2026trace,
  title     = {{TRACE}: A Fine-Grained Benchmark for Temporal Reasoning and Chronological Knowledge Editing over Large Language Models},
  author    = {Liu, Xiliang and Xie, Zhiwen and Wong, Derek F.},
  booktitle = {Proceedings of the ...},
  year      = {2026},
  url       = {https://github.com/wenyi3906/TRACE}
}
```

Please replace the venue placeholder with the official proceedings information once it becomes available.

## Acknowledgments

This project builds on [EasyEdit](https://github.com/zjunlp/EasyEdit). TRACE uses event data derived from [ICEWS](https://github.com/andybega/icews). We thank the authors and maintainers of these resources.

## License

Please see the `LICENSE` file for the license governing the code in this repository. External models, datasets, and EasyEdit components remain subject to their respective licenses.
