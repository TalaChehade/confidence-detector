# Eva training guide

Eva is not a binary classifier and is not used as a pre-trained model. It is
the fine-tuned T5-Large query-complexity classifier from Adaptive-RAG (Jeong et
al., 2024), with generated labels:

- `A`: no retrieval;
- `B`: single-step retrieval;
- `C`: multi-step retrieval.

Prepare the official data archive and normalize its silver and inductive-bias
labels:

```bash
python experiments/prepare_adaptive_rag_data.py \
  --download-to data/adaptive_rag_data.tar.gz \
  --extract-to data/adaptive_rag_official \
  --output data/adaptive_rag_eva.json
```

By default this selects the official `flan_t5_xl` silver labels. Use
`--label-source gpt` or `--label-source flan_t5_xxl` when Eva should match one
of those generation models.

Fine-tune T5-Large:

```bash
python experiments/train_complexity_evaluator.py data/adaptive_rag_eva.json \
  --output-dir models/eva
```

The script uses learning rate `3e-5`, maximum source length `384`, document
stride reference `128`, batch sizes `32`/`100`, AdamW weight decay `0.01`, and
15 epochs. It selects the checkpoint with the best validation macro-F1.

At inference, it evaluates the first generated-token logits for `A`, `B`, and
`C`, then reports the static complexity score used by INKER:

$$E = 0\cdot P(A) + 0.5\cdot P(B) + 1\cdot P(C).$$

Run evaluator-only testing with:

```bash
python experiments/test_complexity_evaluator.py --eva-model models/eva
```

Run it in the detector pipeline with:

```bash
python experiments/run_combined_detection.py --eva-model models/eva
```

The combined score remains $K(t_i) = (E - m_{\tilde{i}})s_i$. Training data
must not overlap with the final QA test queries.
