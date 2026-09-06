# How Eva is trained: a worked example

Eva is a language model used as a **three-class query-complexity classifier**.
It starts as T5-Large, an encoder-decoder language model. Fine-tuning teaches
it to generate one label token rather than a natural-language answer:

| Label | Retrieval strategy | Normalized complexity |
|---|---|---:|
| `A` | No retrieval | 0.0 |
| `B` | Single-step retrieval | 0.5 |
| `C` | Multi-step retrieval | 1.0 |

This is still ordinary supervised neural-network training. The difference from
a classifier with a separate `Linear(... -> 3)` head is that T5's existing
decoder vocabulary produces the labels `A`, `B`, and `C`.

## Where a training label comes from

The official Adaptive-RAG data combines two kinds of automatically generated
labels:

1. **Silver labels.** A query is run through no-retrieval, single-step, and
   multi-step QA systems. The cheapest strategy that answers it correctly is
   assigned: `A` before `B` before `C`.
2. **Inductive-bias labels.** Queries left unlabeled because all strategies
   fail are assigned `B` if they come from a single-hop dataset and `C` if they
   come from a multi-hop dataset.

The training script uses the official combined `binary_silver/train.json` and
the official silver `valid.json`. It does not infer labels from query length,
keywords, or a hand-written complexity heuristic.

## Small by-hand example

Imagine the prepared training JSON contains this one record:

```json
{
  "question": "Who wrote Hamlet and in which year was it first performed?",
  "answer": "B"
}
```

Suppose this record received its silver label because no retrieval failed, but
single-step retrieval and multi-step retrieval both answered correctly. The
cheapest successful strategy is single-step retrieval, so the target is `B`.

### 1. Tokenize the source and target

T5's encoder receives the question tokens. The exact IDs vary by tokenizer
version, so these are conceptual tokens:

```text
encoder input:  [Who] [wrote] [Hamlet] [and] [in] [which] ... [</s>]
decoder target: [B] [</s>]
```

The decoder is given its start token and must predict the target sequence. At
the first decoding position, it produces a logit for every token in the T5
vocabulary. We only care that the logit for `B` should become larger than the
ones for `A` and `C`.

### 2. Compute the loss

Assume that, early in training, the first-step logits for the three label
tokens are:

| Candidate label | Logit |
|---|---:|
| `A` | 0.2 |
| `B` (correct target) | 0.4 |
| `C` | 0.1 |

After softmax over these label logits, imagine the probabilities are:

| Candidate label | Probability |
|---|---:|
| `A` | 0.32 |
| `B` | 0.39 |
| `C` | 0.29 |

The first-token cross-entropy loss is:

$$-\log P(B) = -\log(0.39) \approx 0.94.$$

The end-of-sequence token also has a standard language-model loss. The trainer
averages these token losses for the example. A high loss means T5 did not give
enough probability to the correct generated label.

### 3. Back-propagate and update T5

Back-propagation calculates how every T5 parameter contributed to the loss.
AdamW then changes those parameters using the paper's learning rate `3e-5` and
weight decay `0.01`. After many examples, questions whose wording and reasoning
requirements resemble this one should give a larger decoder score to `B`.

No model is trained to write a long answer here. It only learns the mapping:

```text
question text -> generate A, B, or C
```

## Why the Colab run says micro-batch 1 but effective batch 32

The paper specifies training batch size 32. T5-Large at length 384 does not fit
as a physical batch of 32 on a 16 GB Colab GPU. Our script therefore does this:

```text
example 1  -> forward pass -> save gradients
example 2  -> forward pass -> add gradients
...
example 32 -> forward pass -> add gradients -> one AdamW update
```

That is gradient accumulation. It produces one optimizer update from 32
examples, matching the intended effective batch size while requiring memory for
only one example at a time. Gradient checkpointing recomputes selected forward
activations during back-propagation to further reduce memory use.

## What happens after fine-tuning: calculating E

For a new question, Eva performs one decoder step and reads the probabilities
of generating `A`, `B`, and `C`. For example:

| Label | Eva probability |
|---|---:|
| `A` | 0.10 |
| `B` | 0.65 |
| `C` | 0.25 |

The predicted class is `B`, because it has the largest probability. INKER also
needs a static scalar complexity score compatible with token confidence in
`[0, 1]`, so this repository uses the expected normalized class level:

$$E = 0\cdot P(A) + 0.5\cdot P(B) + 1\cdot P(C)$$

For the example:

$$E = 0\cdot0.10 + 0.5\cdot0.65 + 1\cdot0.25 = 0.575.$$

That one value is calculated once from the input query and held fixed throughout
answer generation. For each generated non-stop-word token, the detector then
uses:

$$K(t_i) = (E - m_{\tilde{i}})s_i.$$

Here `m_tilde_i` comes from the separate confidence detector and `s_i` is 1
for a content token and 0 for a stop word. Eva does not retrain during
generation and it does not inspect the generated answer when computing `E`.
