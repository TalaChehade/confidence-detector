# INKER Confidence Detector Replication

This repository contains a cleaned, modular implementation of the confidence-detector experiment used in the INKER replication work. It loads Mistral-7B-Instruct, learns a confidence direction with paired hidden-state differences and PCA, and evaluates confidence on synthetic statements and generated answers.

The commands below take a new user from cloning the repository to the generated result files. The default configuration expects a CUDA-capable GPU with enough memory for Mistral-7B in 4-bit mode. CPU-only execution is not a practical default for this experiment.

> Important limitation: the confidence-direction/PCA pipeline is reproduced from the original experiment. The query-complexity value `E` is a rule-based proxy, not INKER's trained T5-large Eva evaluator. Read `docs/replication.md` before interpreting the full-K results.

## Pipeline overview

Training follows this path:

```text
confidence_statements1.json
          |
          v
confident / unconfident statements
          |
          v
progressively truncated prefixes
          |
          v
Mistral-7B hidden states, layers 10-25
          |
          v
paired hidden-state differences
          |
          v
mean centering and one-component PCA
          |
          v
PCA sign correction
          |
          v
models/inker_rep_reader.pkl
```

Generated-answer scoring follows this path:

```text
question -> Mistral answer -> hidden state for each answer token
          -> signed projection -> layer average -> raw score m_i
          -> causal min-max normalization -> m_tilde_i
```

The confidence-only experiment uses `confidence_i = m_tilde_i`. The full-K experiment uses `K(t_i) = (E - m_tilde_i) * s_i`. The mathematical derivation is in `docs/methodology.md`.

## Repository structure

```text
inker-confidence-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/default.yaml
├── data/README.md
├── docs/
│   ├── methodology.md
│   ├── replication.md
│   └── results.md
├── experiments/
│   ├── _common.py
│   ├── train_detector.py
│   ├── evaluate_detector.py
│   ├── run_confidence_only.py
│   └── run_test_suite.py
└── src/inker/
    ├── config.py
    ├── complexity.py
    ├── dataset.py
    ├── directions.py
    ├── generation.py
    ├── hidden_states.py
    ├── model.py
    └── scoring.py
```

## Artifact layout

The dataset, trained reader, and experiment outputs are kept outside GitHub because they can be large or externally distributed. By default, artifacts are stored at:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/
├── datasets/confidence_statements1.json
├── models/inker_rep_reader.pkl
├── results/
│   ├── replication/
│   ├── confidence_only/
│   └── full_k/
└── logs/
```

Change the artifact root in `configs/default.yaml` for a local run.

## 1. Clone the repository

```bash
git clone https://github.com/TalaChehade/confidence-detector.git
cd confidence-detector
git status
```

The repository's source branch is available from GitHub as `main`.

## 2. Create the Python environment

Python 3.10 or 3.11 is recommended. With conda:

```bash
conda create -n inker-confidence python=3.11 -y
conda activate inker-confidence
python -m pip install -r requirements.txt
```

With a standard Python installation instead:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a local GPU run, install the PyTorch build appropriate for the installed CUDA version if the default PyTorch package does not match the machine.

Verify the selected interpreter and GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The recommended run should print `True` for CUDA availability.

## 3. Authenticate with Hugging Face

The model is `mistralai/Mistral-7B-Instruct-v0.1`. You need a Hugging Face account, access to the model, and a user access token. Authenticate once:

```bash
hf auth login
hf auth whoami
```

Never put the token in a source file, `default.yaml`, or a committed notebook.

### Google Colab authentication

Store the token in Colab Secrets under the name `HF_TOKEN`, then run:

```python
from google.colab import drive, userdata
from huggingface_hub import login

drive.mount("/content/drive")
login(userdata.get("HF_TOKEN"))
```

In another Colab cell, clone and install the project:

```bash
!git clone https://github.com/TalaChehade/confidence-detector.git
%cd confidence-detector
!pip install -r requirements.txt
```

## 4. Add the dataset

Obtain the dataset used by the original experiment and place the file at:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/datasets/confidence_statements1.json
```

For a local run, place it under the configured artifact root, for example:

```text
artifacts/datasets/confidence_statements1.json
```

The file must contain the confidence statements expected by the dataset builder. See `data/README.md` for the data-location convention. The dataset is not generated by the experiment scripts, and private or restricted data should not be committed to GitHub.

## 5. Configure the experiment

All experiment scripts read `configs/default.yaml`. The default settings are:

- model: `mistralai/Mistral-7B-Instruct-v0.1`;
- 4-bit quantization with `float16` computation;
- hidden layers 10 through 25;
- last-token representation (`rep_token=-1`);
- one paired-difference operation;
- batch size 32 and maximum input length 512;
- 512 training pairs;
- confidence threshold 0.5;
- maximum 60 generated tokens and repetition penalty 1.1;
- deterministic experiment seed 0.

The default artifact root is `/content/drive/MyDrive/INKER_Confidence_Detector`. For a local run, edit `paths.project_dir` in `configs/default.yaml` to a writable directory containing `datasets`, `models`, and `results` subdirectories.

## 6. Train the confidence detector

Run from the repository root:

```bash
python experiments/train_detector.py
```

Training constructs the dataset, creates the training split, extracts hidden states, computes paired differences, fits one PCA direction per layer, applies the label-based sign correction, and saves the reader here:

```text
<project_dir>/models/inker_rep_reader.pkl
```

Training is needed when the reader does not exist or when the training configuration changes. It loads Mistral and may take substantial time and GPU memory.

## 7. Evaluate the synthetic replication

Run after training:

```bash
python experiments/evaluate_detector.py
```

This rebuilds the deterministic dataset construction, loads the saved reader, and evaluates the eval and test splits. It reports ROC-AUC, pairwise accuracy (`score(confident) > score(unconfident)`), and per-topic behavior.

Outputs:

```text
<project_dir>/results/replication/replication_metrics.csv
<project_dir>/results/replication/eval_per_topic.csv
<project_dir>/results/replication/test_per_topic.csv
```

## 8. Run confidence-only inference

Run after the reader has been trained:

```bash
python experiments/run_confidence_only.py
```

For each question, the model generates an answer, scores every generated token, averages layers 10 through 25, and applies causal normalization. A content token below the threshold causes the confidence-only trigger.

Outputs:

```text
<project_dir>/results/confidence_only/confidence_only_questions.csv
<project_dir>/results/confidence_only/confidence_only_tokens.csv
```

## 9. Run the full-K test suite

Run after the reader has been trained:

```bash
python experiments/run_test_suite.py
```

This stores the generated answer, expected answer where available, complexity proxy `E`, confidence statistics, full-K trigger, and confidence-only trigger.

Outputs:

```text
<project_dir>/results/full_k/test_suite_questions.csv
<project_dir>/results/full_k/test_suite_tokens.csv
```

The `auto_correct` column is only a substring convenience check when an expected answer exists. It is not a robust semantic correctness metric.

## 10. Run the complete workflow

After setup and authentication, run these commands in order:

```bash
python experiments/train_detector.py
python experiments/evaluate_detector.py
python experiments/run_confidence_only.py
python experiments/run_test_suite.py
```

If `<project_dir>/models/inker_rep_reader.pkl` already exists, omit the first command. Each script also accepts an alternative configuration:

```bash
python experiments/evaluate_detector.py --config configs/my_config.yaml
```

Use the same `--config` option for the other experiment scripts when needed.

## 11. Understand the final results

Inspect the result files in this order:

1. `replication_metrics.csv`: confirms whether the learned direction separates synthetic confident and unconfident examples.
2. `eval_per_topic.csv` and `test_per_topic.csv`: show topics with weak or unstable pairwise separation.
3. `confidence_only_questions.csv` and its token file: show generated answers and causal normalized confidence values.
4. `test_suite_questions.csv` and its token file: compare full-K and confidence-only decisions and inspect individual tokens.
5. `docs/results.md`: record metrics, failure cases, configuration, and any conclusions after a run.

The full-K results use the rule-based `E` proxy, not INKER's trained Eva evaluator. The `auto_correct` field is not a semantic factuality evaluator. Claims about complete end-to-end INKER reproduction must therefore be qualified.

## Troubleshooting

**`FileNotFoundError: Representation reader not found`**

Run `train_detector.py` first, or update `paths.project_dir` so the configured `models/inker_rep_reader.pkl` path points to the trained reader.

**CUDA out of memory**

Reduce `detector.batch_size` or `detector.max_length`, or use a GPU with more memory. Record configuration changes with the results because they may affect runtime and outputs.

**Hugging Face authentication or access errors**

Run `hf auth whoami`, confirm model access, and authenticate again with `hf auth login`.

**Missing dataset**

Check the exact path in `configs/default.yaml` and confirm that `confidence_statements1.json` exists there.

**Results differ between runs**

The configured seed is `0`, but different PyTorch, CUDA, GPU, or dependency versions can still produce small numerical differences. Record the Python, PyTorch, Transformers, GPU, and configuration versions with each run.

## Further documentation

- `docs/methodology.md`: mathematical derivation of the detector and scoring.
- `docs/replication.md`: what is preserved, what is intentionally unusual, and which parts are not exact INKER components.
- `docs/results.md`: template for recording empirical results and analysis.
