# Replication Status

This repository is a cleaned, modular version of the original INKER
confidence-detector Colab experiment.

## Preserved exactly from the original experiment

- Mistral-7B-Instruct-v0.1
- 4-bit loading configuration
- left-padding tokenizer and `pad_token_id = 0`
- layers 10 through 25
- last-token hidden-state representation for statement training
- confident/unconfident truncation procedure
- randomized training pair order and within-pair order
- shifted eval/test construction
- paired-difference hidden states
- one-component PCA per layer
- PCA sign disambiguation
- signed projection and layer averaging
- causal min-max normalization
- NLTK English stop-word filtering
- threshold 0.5
- deterministic generation with repetition penalty 1.1
- full `K(t_i)` trigger and the original comparison trigger
- later standalone confidence-only token experiment

## Important non-exact INKER component

The query-complexity term `E` in this repository is the same rule-based proxy
used in the original Colab experiment. It is not INKER's trained T5-large
Eva evaluator.

Therefore, the PCA confidence-direction part can be described as a replication
of the Appendix-B confidence-detector recipe, while the complete end-to-end
INKER retrieval activation should be described as a partial replication unless
the actual Eva evaluator is substituted.

## Intentionally unusual behavior retained

The eval/test split pairs:

`honest_statements[:-1]`

with:

`untruthful_statements[1:]`

This can cross topic boundaries. It is retained because changing it would alter
the experiment rather than merely refactor it.

The test set also uses `reshaped_data[-300:-1]`, producing an odd number of
texts. Evaluation truncates to complete pairs before computing metrics, exactly
as in the original notebook.
