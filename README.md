# INKER Confidence Detector Replication

This repository contains a cleaned, modular implementation of the INKER confidence-detector experiment. It loads Mistral-7B-Instruct, learns a confidence direction with paired hidden-state differences and PCA, and evaluates confidence on synthetic statements and generated answers.

This guide is specifically for running the project in **Google Colab** with Google Drive. The default configuration expects a CUDA-capable Colab GPU and stores the trained reader and result files in Drive.

> Important limitation: the confidence-direction/PCA pipeline is reproduced from the original experiment. The query-complexity value `E` is a rule-based proxy, not INKER's trained T5-large Eva evaluator. Read `docs/replication.md` before interpreting the full-K results.

## Pipeline overview

The project has two stages: learn a confidence direction, then use it to score generated answers.

### A. Learn the confidence direction

```mermaid
flowchart LR
    A[confidence_statements1.json] --> B[Confident and unconfident statements]
    B --> C[Progressively truncated prefixes]
    C --> D[Mistral hidden states<br/>layers 10-25]
    D --> E[Paired hidden-state differences]
    E --> F[Center differences]
    F --> G[One-component PCA per layer]
    G --> H[Correct PCA sign]
    H --> I[(inker_rep_reader.pkl)]

    classDef input fill:#e8f1ff,stroke:#377dbe,color:#12304a;
    classDef process fill:#f4f0ff,stroke:#7957b8,color:#2b1d4d;
    classDef output fill:#e7f6ed,stroke:#37845a,color:#153d25;
    class A input;
    class B,C,D,E,F,G,H process;
    class I output;
```

**In plain language:** the training script creates matched confident and unconfident examples, observes their hidden states, and saves one learned direction for each selected layer.

### B. Score a generated answer

```mermaid
flowchart LR
    Q[Question] --> M[Mistral generates an answer]
    M --> T[One hidden state per answer token]
    T --> P[Project onto learned directions]
    P --> L[Average layers 10-25]
    L --> R[Raw token score m_i]
    R --> N[Causal min-max normalization]
    N --> S[Normalized confidence m_tilde_i]
    S --> C[Confidence-only decision]
    S --> K[Full-K score<br/>K(t_i) = (E - m_tilde_i) * s_i]

    classDef input fill:#e8f1ff,stroke:#377dbe,color:#12304a;
    classDef process fill:#fff4df,stroke:#c77b20,color:#4a2b08;
    classDef decision fill:#e7f6ed,stroke:#37845a,color:#153d25;
    class Q input;
    class M,T,P,L,R,N process;
    class S,C,K decision;
```

**Why normalization is causal:** when token `i` is scored, only tokens `0...i` can influence its normalized value. Future tokens cannot change an earlier confidence decision.

The confidence-only experiment uses `confidence_i = m_tilde_i`. The full-K experiment uses `K(t_i) = (E - m_tilde_i) * s_i`. The full mathematical derivation is in `docs/methodology.md`.

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
│   ├── train_detector.py
│   ├── evaluate_detector.py
│   ├── run_confidence_only.py
│   └── run_test_suite.py
└── src/inker/
```

## Artifact layout in Google Drive

The dataset, trained reader, and experiment outputs are stored outside GitHub so they persist after Colab disconnects:

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

The checked-in `configs/default.yaml` already points to this Drive location.

## 1. Create a Colab notebook and GPU runtime

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Select **Runtime > Change runtime type**.
4. Set **Hardware accelerator** to **GPU**, then save.

Run this first cell:

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

Continue only when `CUDA available: True` is printed.

## 2. Mount Google Drive

Run this cell and authorize access when prompted:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Drive stores the trained reader and CSV results so they survive a temporary Colab runtime reset.

## 3. Clone the repository and install dependencies

Run this cell:

```bash
!git clone https://github.com/TalaChehade/confidence-detector.git
%cd confidence-detector
!pip install -q -r requirements.txt
```

If the runtime already contains the folder, skip `git clone` and run only:

```python
%cd confidence-detector
```

## 4. Authenticate with Hugging Face

The model is `mistralai/Mistral-7B-Instruct-v0.1`. You need a Hugging Face account, model access, and a user access token.

1. Open the **Secrets** panel in Colab using the key icon.
2. Add a secret named `HF_TOKEN`.
3. Enable notebook access for that secret.
4. Run this cell:

```python
from google.colab import userdata
from huggingface_hub import login

login(userdata.get("HF_TOKEN"))
```

Never print the token or put it in `default.yaml`, a source file, or a committed notebook.

## 5. Add the dataset to Drive

Obtain the dataset used by the original experiment and place it at:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/datasets/confidence_statements1.json
```

The dataset is not generated by the scripts. Upload it to Drive before continuing. Confirm that Colab can see it:

```python
from pathlib import Path

dataset = Path(
    "/content/drive/MyDrive/INKER_Confidence_Detector/"
    "datasets/confidence_statements1.json"
)
print("Dataset exists:", dataset.exists())
```

The output must be `Dataset exists: True`.

## 6. Review the configuration

The experiment reads `configs/default.yaml`. The default settings are:

- model: `mistralai/Mistral-7B-Instruct-v0.1`;
- 4-bit quantization with `float16` computation;
- hidden layers 10 through 25;
- last-token representation (`rep_token=-1`);
- batch size 32 and maximum input length 512;
- 512 training pairs;
- confidence threshold 0.5;
- maximum 60 generated tokens and repetition penalty 1.1;
- deterministic seed 0.

The default artifact root is `/content/drive/MyDrive/INKER_Confidence_Detector`, with `datasets`, `models`, and `results` beneath it. You normally do not need to edit the configuration.

## 7. Train the confidence detector

Run this Colab cell from the `confidence-detector` directory:

```bash
!python experiments/train_detector.py
```

Training constructs the dataset, extracts hidden states, computes paired differences, fits one PCA direction per layer, applies the label-based sign correction, and saves:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/models/inker_rep_reader.pkl
```

This is the longest step and may require substantial GPU memory. Do not rerun it unless the reader is missing or the training configuration changed.

## 8. Evaluate the synthetic replication

Run after training completes:

```bash
!python experiments/evaluate_detector.py
```

This evaluates the synthetic eval and test splits using ROC-AUC, pairwise accuracy, and per-topic behavior. It writes:

```text
results/replication/replication_metrics.csv
results/replication/eval_per_topic.csv
results/replication/test_per_topic.csv
```

## 9. Run confidence-only inference

Run after the reader has been trained:

```bash
!python experiments/run_confidence_only.py
```

The model generates answers, scores every answer token, averages layers 10 through 25, and applies causal normalization. It writes:

```text
results/confidence_only/confidence_only_questions.csv
results/confidence_only/confidence_only_tokens.csv
```

## 10. Run the full-K test suite

Run after the reader has been trained:

```bash
!python experiments/run_test_suite.py
```

This writes the generated answers, confidence statistics, complexity proxy `E`, full-K trigger, and confidence-only trigger:

```text
results/full_k/test_suite_questions.csv
results/full_k/test_suite_tokens.csv
```

The `auto_correct` column is only a substring check when an expected answer exists. It is not a robust semantic correctness metric.

## 11. Complete Colab cell order

After the setup cells, run these cells in order:

```bash
!python experiments/train_detector.py
```

```bash
!python experiments/evaluate_detector.py
```

```bash
!python experiments/run_confidence_only.py
```

```bash
!python experiments/run_test_suite.py
```

If the reader already exists in Drive, skip the training cell.

## 12. Understand the final results

Inspect the files in this order:

1. `replication_metrics.csv`: checks whether the learned direction separates synthetic confident and unconfident examples.
2. `eval_per_topic.csv` and `test_per_topic.csv`: show topics with weak or unstable separation.
3. `confidence_only_questions.csv` and its token file: show generated answers and normalized confidence values.
4. `test_suite_questions.csv` and its token file: compare full-K and confidence-only decisions.
5. `docs/results.md`: record metrics, failure cases, configuration, and conclusions after a run.

The full-K results use the rule-based `E` proxy, not INKER's trained Eva evaluator. Claims about complete end-to-end INKER reproduction must therefore be qualified.

## Troubleshooting

**CUDA is unavailable**

Open **Runtime > Change runtime type**, select **GPU**, reconnect, and rerun the GPU check cell.

**Colab disconnected or restarted**

Remount Drive, change back to the repository directory with `%cd confidence-detector`, reinstall requirements if needed, and continue from the last completed step. The trained reader and CSV files in Drive are preserved.

**`FileNotFoundError: Representation reader not found`**

Run the training cell first and confirm that `inker_rep_reader.pkl` exists in the Drive `models` directory.

**CUDA out of memory**

Reduce `detector.batch_size` or `detector.max_length` in `configs/default.yaml`, restart the runtime, and rerun training. Record configuration changes with the results.

**Hugging Face authentication or access errors**

Confirm that the `HF_TOKEN` Colab Secret is available to the notebook, that the account has access to the Mistral model, and rerun the `login(userdata.get("HF_TOKEN"))` cell.

**Missing dataset**

Confirm that `confidence_statements1.json` is in the exact Drive path shown in Step 5 and that the dataset check prints `True`.

## Further documentation

- `docs/methodology.md`: mathematical derivation of the detector and scoring.
- `docs/replication.md`: preserved behavior, intentional quirks, and replication limits.
- `docs/results.md`: template for recording empirical results and analysis.
