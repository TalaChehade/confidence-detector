# INKER Confidence Detector Replication

A cleaned and modular implementation of the confidence-detector experiment used
in the INKER replication work.

The repository separates model loading, dataset construction, hidden-state
extraction, paired-difference PCA, scoring, token-level inference, and
experiments so that the method can be reproduced without relying on one large
Colab notebook.

> Important: the confidence-direction/PCA pipeline is reproduced from the
> original experiment. The query-complexity value `E` is still a rule-based
> proxy, not INKER's trained T5-large Eva evaluator. See
> `docs/replication.md`.

## Pipeline overview

Training:

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
mean centering
          |
          v
1-component PCA per layer
          |
          v
PCA sign correction
          |
          v
inker_rep_reader.pkl
```

Generated-answer confidence:

```text
Question
   |
   v
Mistral answer
   |
   v
hidden state for each answer token
   |
   v
signed projection onto PCA direction
   |
   v
average layers 10-25
   |
   v
raw confidence m_i
   |
   v
causal min-max normalization
   |
   v
m_tilde_i
   |
   +----------------------------+
   |                            |
   v                            v
confidence-only             full activation
confidence=m_tilde_i        K(t_i)=(E-m_tilde_i)s_i
```

For the full mathematical derivation, see `docs/methodology.md`.

## Repository structure

```text
inker-confidence-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
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
└── src/
    └── inker/
        ├── __init__.py
        ├── config.py
        ├── model.py
        ├── dataset.py
        ├── hidden_states.py
        ├── directions.py
        ├── scoring.py
        ├── complexity.py
        └── generation.py
```

## External Google Drive structure

By default, artifacts are stored outside GitHub:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/
├── datasets/
│   └── confidence_statements1.json
├── models/
│   └── inker_rep_reader.pkl
├── results/
│   ├── replication/
│   ├── confidence_only/
│   └── full_k/
└── logs/
```

You can change these paths in `configs/default.yaml`.

## 1. Colab setup

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repository:

```bash
!git clone YOUR_REPOSITORY_URL
%cd inker-confidence-detector
```

Install dependencies:

```bash
!pip install -r requirements.txt
```

## 2. Hugging Face authentication

Store your Hugging Face token in Colab Secrets as `HF_TOKEN`.

Then authenticate:

```python
from google.colab import userdata
from huggingface_hub import login

HF_TOKEN = userdata.get("HF_TOKEN")
login(HF_TOKEN)
```

The experiment scripts can use the cached Hugging Face login. Alternatively:

```python
import os
os.environ["HF_TOKEN"] = HF_TOKEN
```

Never commit your Hugging Face token to GitHub.

## 3. Dataset setup

Place:

```text
confidence_statements1.json
```

at:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/datasets/
```

or modify `configs/default.yaml`.

## 4. Configuration

The default experiment uses:

- model: `mistralai/Mistral-7B-Instruct-v0.1`
- 4-bit quantization
- layers 10 through 25
- last-token representation (`rep_token=-1`)
- one paired-difference operation
- batch size 32
- maximum input length 512
- 512 training pairs
- confidence threshold 0.5
- deterministic generation
- maximum 60 generated tokens
- repetition penalty 1.1

All experiment scripts read these values from:

```text
configs/default.yaml
```

## 5. Train the confidence detector

Run:

```bash
!python experiments/train_detector.py
```

This performs:

1. dataset construction;
2. training split creation;
3. hidden-state extraction;
4. paired differences;
5. PCA independently for layers 10-25;
6. sign correction;
7. serialization of the representation reader.

Output:

```text
/content/drive/MyDrive/INKER_Confidence_Detector/
models/inker_rep_reader.pkl
```

Training is required only when the representation reader does not already
exist or when you intentionally change the training configuration.

## 6. Evaluate the synthetic replication

Run:

```bash
!python experiments/evaluate_detector.py
```

This rebuilds the same deterministic dataset construction, loads the saved
representation reader, and evaluates the eval/test splits.

Metrics:

- ROC-AUC
- pairwise accuracy, where a pair is correct when
  `score(confident) > score(unconfident)`
- per-topic pairwise accuracy

Outputs:

```text
results/replication/
├── replication_metrics.csv
├── eval_per_topic.csv
└── test_per_topic.csv
```

## 7. Run confidence-only generated-answer inference

Run:

```bash
!python experiments/run_confidence_only.py
```

For each question, the model generates an answer, obtains a hidden state for
every generated token, projects that state onto the learned confidence
directions, averages layers, and applies causal normalization.

The standalone confidence-only experiment defines:

`confidence_i = m_tilde_i`.

With the default threshold:

- `confidence_i >= 0.5` -> confident
- `confidence_i < 0.5` -> unconfident

The question triggers when any content token is below the threshold.

Outputs:

```text
results/confidence_only/
├── confidence_only_questions.csv
└── confidence_only_tokens.csv
```

## 8. Run the structured full-K test suite

Run:

```bash
!python experiments/run_test_suite.py
```

This uses:

`K(t_i) = (E - m_tilde_i) * s_i`.

It stores the model answer, expected answer when available, `E`,
`mean_m_tilde`, `min_m_tilde`, `max_K`, the full-K trigger, and the original
confidence-only comparison trigger.

Outputs:

```text
results/full_k/
├── test_suite_questions.csv
└── test_suite_tokens.csv
```

The `auto_correct` field is only a simple substring convenience check when an
expected answer exists. It must not be treated as a robust semantic correctness
metric in research conclusions.

## 9. PCA formulation

At layer `l`, the training procedure first constructs paired differences:

`Delta h_i^(l) = h_(2i)^(l) - h_(2i+1)^(l)`.

Their mean is:

`mu_l = (1/N) sum_i Delta h_i^(l)`.

PCA is fitted to:

`z_i^(l) = Delta h_i^(l) - mu_l`.

The first principal component `v_l` maximizes projected variance:

`v_l = argmax_(||v||=1) sum_i (z_i^(l) dot v)^2`.

Because PCA direction sign is arbitrary, the training labels are used to choose
`sign_l`. A new representation is scored as:

`m_l(x) = sign_l * ((h_l(x) - mu_l) dot v_l)`.

The final raw score is the average across layers:

`m(x) = (1/|L|) sum_(l in L) m_l(x)`.

See `docs/methodology.md` for the complete derivation and token-level equations.

## 10. Causal normalization

For token `i`, only scores from tokens `0...i` may be used.

If the observed causal range is nonzero:

`m_tilde_i = (m_i - min(m_0...m_i)) /
             (max(m_0...m_i) - min(m_0...m_i))`.

The first token, or an equal-valued window, receives `0.5`.

This prevents future answer tokens from influencing the current token's
normalized confidence.

## 11. Replication caveat

The repository currently contains two conceptually different parts:

1. the paired-difference PCA confidence detector, reproduced from the original
   experiment; and
2. the query complexity term `E`, implemented with the original notebook's
   rule-based proxy.

The proxy is:

`E = 0.5 * min(number_of_words / 25, 1)
   + 0.5 * min(number_of_multihop_cues / 3, 1)`.

This is not the trained T5-large Eva evaluator described by INKER. Therefore
claims about complete INKER reproduction should be qualified accordingly.

## 12. Results analysis

After rerunning the cleaned implementation, add research results to:

```text
docs/results.md
```

The most important follow-up analysis is whether the confidence direction
actually tracks factual correctness on generated answers rather than merely
separating the synthetic confident/unconfident training styles.

That analysis should be performed only after the clean implementation is
verified to reproduce the original experiment outputs.
