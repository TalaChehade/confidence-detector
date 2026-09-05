# Methodology

## 1. Goal

The detector learns directions in a language model's hidden-state space that
separate text phrased confidently from text phrased unconfidently. Those
directions are then used to assign confidence scores to generated answer tokens.

The current implementation uses Mistral-7B-Instruct-v0.1 and transformer layers
10 through 25.

## 2. Training data construction

For each topic, the source JSON contains confident and unconfident statements.
Each confident statement is paired with an unconfident statement.

Both statements are tokenized and converted into progressively truncated
prefixes. If the two statements yield different numbers of prefixes, `zip`
keeps only the shared number of truncations.

Each confident prefix is wrapped with:

`[INST] Pretend you're a confident person making statements about the world. [/INST]`

Each unconfident prefix is wrapped with:

`[INST] Pretend you're an unconfident person making statements about the world. [/INST]`

## 3. Last-token hidden-state representation

For text `x`, layer `l`, and hidden dimension `d`, let

`h_l(x) in R^d`

denote the hidden state at the representative token.

The tokenizer uses left padding and the detector uses `rep_token = -1`, so the
representation is the final real token for every sequence in a batch.

## 4. Paired differences

After the two members of each training pair are randomly ordered, hidden states
occur consecutively in the flattened training array.

For pair `i` at layer `l`, the code forms

`Delta h_i^(l) = h_(2i)^(l) - h_(2i+1)^(l)`.

Because the pair order is randomized, this difference is not guaranteed to be
confident-minus-unconfident. PCA is sign-invariant, and orientation is corrected
later.

## 5. Mean centering

For `N` paired differences,

`mu_l = (1/N) * sum_i Delta h_i^(l)`.

The centered difference is

`z_i^(l) = Delta h_i^(l) - mu_l`.

The stored `H_train_means[l]` is this mean of the paired-difference
representations, with shape `(1, d)`.

## 6. PCA confidence direction

For every selected layer independently, one-component PCA is fitted to the
centered paired differences.

Equivalently, the first principal component is the unit vector `v_l` that
maximizes projected variance:

`v_l = argmax_(||v||=1) sum_i ( z_i^(l) dot v )^2`.

This is the dominant axis along which the paired confident/unconfident
representations vary.

PCA does not define an intrinsic sign: `v_l` and `-v_l` describe the same axis.

## 7. Sign correction

The original, un-differenced training hidden states are centered using the
stored mean and projected onto `v_l`.

For every training pair, the code checks whether the known confident member has
the minimum or maximum projection.

Let:

`p_max = fraction of pairs where confident member has maximum projection`

`p_min = fraction of pairs where confident member has minimum projection`

Then:

`sign_l = sign(p_max - p_min)`.

If the result is zero, the implementation uses `+1`.

The PCA direction itself remains unchanged. The sign is stored separately and
must be applied exactly once during scoring.

## 8. Statement confidence score

For a new representation `h_l(x)`, the layer score is:

`m_l(x) = sign_l * ((h_l(x) - mu_l) dot v_l)`.

Scores are averaged across the selected layers:

`m(x) = (1/|L|) * sum_(l in L) m_l(x)`.

For the current configuration:

`L = {10, 11, ..., 25}`.

## 9. Synthetic evaluation

Eval/test texts are arranged conceptually as:

`[confident_1, unconfident_1, confident_2, unconfident_2, ...]`.

If a split contains an odd number of texts, the trailing text is discarded.

Binary labels are generated as:

`[1, 0, 1, 0, ...]`.

Two metrics are computed:

- ROC-AUC over all text scores.
- Pairwise accuracy: the fraction of pairs satisfying
  `score(confident) > score(unconfident)`.

## 10. Generated answer token scoring

A question is wrapped as:

`[INST] Answer the question directly and concisely. Do not provide extra context.

QUESTION [/INST]`

Generation is deterministic (`do_sample=False`).

The complete prompt+answer sequence is passed through Mistral again so the
hidden state of every generated token can be extracted.

For generated token `t_i`, the same signed confidence projection is computed at
every selected layer and averaged, producing a raw token score `m_i`.

## 11. Causal normalization

Raw token scores are converted to a normalized sequence without looking at
future tokens.

For token `i`, define the causal window:

`W_i = {m_0, ..., m_i}`.

Let:

`lo_i = min(W_i)`

`hi_i = max(W_i)`.

If `i = 0`, or if `hi_i - lo_i < 1e-8`, the normalized score is `0.5`.

Otherwise:

`m_tilde_i = (m_i - lo_i) / (hi_i - lo_i)`.

Therefore:

`m_tilde_i in [0, 1]`.

A token's normalized score can depend only on itself and earlier tokens.

## 12. Content-token mask

Special and formatting-only tokens are skipped.

For remaining tokens, the tokenizer marker `▁` or `Ġ` is removed, the token is
lowercased, and NLTK English stop words are filtered.

Define:

`s_i = 1` for a content token,

`s_i = 0` for a stop word or ignored token.

Only content tokens participate in retrieval-trigger decisions.

## 13. Query-complexity proxy E

The current implementation does not contain INKER's trained T5-large Eva
evaluator. It preserves the original notebook's rule-based stand-in.

The cue set is:

`{and, who, which, before, after, same, both, compare, than}`.

If `n` is the number of words and `c` is the number of cue hits:

`length_component = min(n / 25, 1)`

`cue_component = min(c / 3, 1)`

and:

`E = 0.5 * length_component + 0.5 * cue_component`.

`E` is clipped to `[0,1]` and rounded to four decimals.

## 14. Full K(t_i) activation

The full experiment computes:

`K(t_i) = (E - m_tilde_i) * s_i`.

With threshold `tau = 0.5`, the full detector triggers if any content token
satisfies:

`K(t_i) > tau`.

The notebook also records an older comparison rule:

`1 - m_tilde_i > tau`.

At `tau = 0.5`, this corresponds to identifying tokens with
`m_tilde_i < 0.5`.

## 15. Standalone confidence-only experiment

The later confidence-only experiment ignores `E` entirely and directly treats:

`confidence_i = m_tilde_i`.

A token is:

- confident when `confidence_i >= 0.5`;
- unconfident when `confidence_i < 0.5`.

The question-level confidence-only trigger activates when any content token has
confidence below the threshold.

## 16. Pipeline

Training:

`JSON statements`
` -> confident/unconfident prefixes`
` -> Mistral hidden states`
` -> paired differences`
` -> mean centering`
` -> PCA per layer`
` -> sign correction`
` -> saved representation reader`

Inference:

`question`
` -> Mistral answer`
` -> answer-token hidden states`
` -> signed PCA projections`
` -> average layers`
` -> causal normalization`
` -> content-token filter`
` -> confidence-only trigger and/or K(t_i)`
