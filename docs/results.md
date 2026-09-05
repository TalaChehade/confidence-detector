# Analysis of Confidence and Replication Results

## 1. Purpose and scope

This report analyses the result files produced by the recent experiments. The files contain three related evaluations:

1. A small **confidence-only demonstration** using three factual questions.
2. A 13-example **test suite** comparing a full-K decision rule with a confidence-only decision rule.
3. A broader **replication experiment** reporting ROC AUC and pairwise accuracy on evaluation and test splits, including topic-level breakdowns.

The analysis describes what was tested, what the reported values show, where the evidence is incomplete, and how the next experimental iteration could make the conclusions stronger.

A key distinction is necessary throughout: a trigger indicates that the method considers an answer uncertain or in need of additional handling. It does not, by itself, prove that the generated answer is factually wrong. Likewise, a high confidence score does not prove correctness.

## 2. Files and experimental layers

| Experimental layer | Files | Main purpose |
|---|---|---|
| Confidence-only demonstration | `confidence_only_questions.csv`, `confidence_only_tokens.csv` | Examine token-level confidence and aggregate confidence for three factual answers. |
| Full test suite | `test_suite_questions.csv`, `test_suite_tokens.csv` | Compare full-K and confidence-only trigger decisions across deliberately varied question types. |
| Replication benchmark | `replication_metrics.csv`, `eval_per_topic.csv`, `test_per_topic.csv` | Measure how well the replicated scoring method separates paired examples overall and by topic. |

The token files provide intermediate evidence for the question-level decisions. Punctuation, stop words, and end-of-sequence tokens are now filtered before scoring: they are marked `SKIP`, have `is_content=False`, and have blank score fields. This is intentional and should not be treated as missing model output. The filtering also normalizes tokenizer markers such as `▁` before identifying stop words.

## 3. What the tests actually were

### 3.1 Confidence-only demonstration

Three simple factual questions were generated and scored:

- the capital of France;
- the author of *Romeo and Juliet*;
- the largest planet in the Solar System.

For each answer, the experiment recorded the generated response, expected answer, a confidence threshold of `0.5`, the mean confidence over content tokens, the minimum and maximum confidence, the number of content tokens, and the number of low-confidence content tokens.

The token-level file shows that confidence is calculated only for content tokens. Punctuation, stop words, and `</s>` are marked `SKIP` and excluded from the aggregate calculation. The use of subword tokens is visible in examples such as `▁Rome` + `o` and `▁Jul` + `iet`; therefore, token-level uncertainty does not always correspond directly to a complete human-readable word.

### 3.2 Full test suite

The 13 test cases were designed to cover several behaviors rather than to estimate general accuracy. The categories were:

- 2 paper case studies;
- 2 high-confidence-correct cases;
- 4 high-confidence-wrong-target cases;
- 2 numeric-specification cases;
- 1 ambiguous or time-sensitive question;
- 1 question with a deliberately low-confidence expected answer;
- 1 multihop question.

For each test, the export includes the question, generated answer, expected answer where available, an `auto_correct` label where available, the parameter `E`, mean and minimum transformed confidence (`m_tilde`), maximum observed `K`, and the trigger decisions for both methods.

The token-level file additionally records the raw score, transformed confidence, binary token indicator `s_i`, token-level `K`, and whether a token is considered content. In the updated export, stop words and other excluded tokens have blank score fields and are not included in the content-token aggregates. The exact mathematical definitions of `E`, `m_tilde`, `s_i`, and `K` are not included in the result files. Consequently, this report interprets them operationally from their recorded values rather than claiming a formula that cannot be verified from the exports alone.

### 3.3 Replication benchmark

The replication results evaluate pairwise discrimination rather than direct answer accuracy. The reported metrics are:

- **ROC AUC:** how well the score ranks positive examples above negative examples across thresholds;
- **pairwise accuracy:** the proportion of evaluated pairs for which the preferred ordering is correct.

The topic files break pairwise accuracy down across 27 topics. They contain 256 pairs in evaluation and 251 pairs in testing.

## 4. Results

### 4.1 Confidence-only demonstration

| Question | Mean confidence | Minimum confidence | Content tokens | Low-confidence content tokens | Trigger |
|---|---:|---:|---:|---:|---|
| Capital of France | 0.3083 | 0.0000 | 3 | 2 | Yes |
| Author of *Romeo and Juliet* | 0.6141 | 0.0000 | 6 | 2 | Yes |
| Largest planet | 0.5073 | 0.0000 | 6 | 2 | Yes |

The method now triggers on all three answers. France contains three scored content tokens, but two of them (`France` and `Paris`) are below the threshold after stop words are removed. Shakespeare and Jupiter each contain two low-confidence content tokens. This shows that the method is sensitive to token-level uncertainty even when the answer is semantically correct.

The result is therefore useful as a demonstration of uncertainty detection, but it is not an accuracy study. All three generated answers are correct, yet all three are flagged. In this sample, the trigger behaves as a conservative uncertainty signal, not as a direct error detector.

The token results also illustrate why aggregate confidence should be interpreted carefully. Stop words no longer contribute noise to the aggregate, but the answer about Shakespeare still contains confident tokens such as `William`, `Shakespeare`, and `Rome`, alongside low-confidence tokens such as `wrote` and `iet`. The last token is only a subword fragment, so a low score on that fragment may not mean that the model is uncertain about the underlying name. Similar effects occur for the split representation of `Jupiter`.

### 4.2 Full test suite: trigger behavior

| Decision rule | Triggered | Not triggered | Trigger rate |
|---|---:|---:|---:|
| Full-K | 0/13 | 13/13 | 0.0% |
| Confidence-only | 12/13 | 1/13 | 92.3% |

The difference between the two rules is the clearest result in the test suite. The full-K rule never triggered, whereas the confidence-only rule triggered on 12 of 13 cases. The confidence-only rule did not trigger only for:

- the multihop Guns N' Roses question.

This is important: the generated answer was `1987`, while the expected answer was `1999`, and `auto_correct` was `False`, but confidence-only still did not trigger. This is a concrete false-negative example for the current confidence-only decision behavior. It suggests that a response can be confidently generated while still being factually incorrect, especially in a multihop setting.

The test suite also includes several cases in which the generated answer appears plausible or is marked correct, but the confidence-only method triggers. For example, the water formula is marked correct but triggers, and the Kellerman answer is marked correct but triggers. This reinforces the distinction between uncertainty detection and factual correctness: the method may identify answers for review without reliably classifying them as wrong.

The full-K result should be treated cautiously. A zero trigger rate can mean that the method is appropriately conservative, but it can also mean that the threshold, normalization, or implementation makes the rule too difficult to activate. Without the definition of $K$, its threshold, and the full decision equation, the exported values cannot distinguish these explanations.

### 4.3 Test-suite coverage and labels

The test suite is useful as a qualitative stress test, but its labels are incomplete:

- only 4 of 13 rows contain `auto_correct=True`;
- 2 rows contain `auto_correct=False`;
- 7 rows have no expected answer or correctness label;
- one row has a missing generated answer.

This prevents a complete confusion matrix. It is possible to identify at least one false negative (the multihop item), but precision, recall, false-positive rate, and calibration cannot be computed for the whole suite from these files alone.

The categories themselves are valuable because they probe known failure modes: stale or ambiguous facts, numerical claims, multihop reasoning, and intentionally wrong targets. However, the sample is too small and hand-selected to support broad claims about model reliability.

### 4.4 Replication metrics

| Split | ROC AUC | Pairwise accuracy |
|---|---:|---:|
| Evaluation | 0.9450 | 0.96875 |
| Test | 0.9257 | 0.92829 |

The replicated method performs strongly on both splits. Test performance is lower than evaluation performance by approximately:

- **0.0194 ROC AUC**;
- **0.0405 pairwise accuracy**.

This decrease is not automatically evidence of serious overfitting, but it shows a generalization gap. The test set remains strong overall, while being less uniformly separable than the evaluation set.

The aggregate pairwise counts are consistent with the topic files:

- evaluation: 248 correct comparisons out of 256;
- test: 233 correct comparisons out of 251.

### 4.5 Topic-level behavior

Evaluation performance is concentrated near the ceiling. Twenty-two of the 27 topics have pairwise accuracy of 1.0. The weakest evaluation topics are:

| Topic | Pairwise accuracy | Pairs |
|---|---:|---:|
| Health | 0.5000 | 6 |
| Nature | 0.7143 | 7 |
| Food | 0.8571 | 7 |
| Animals | 0.9167 | 12 |
| History | 0.9286 | 14 |

The test split exposes a broader set of weaknesses:

| Topic | Pairwise accuracy | Pairs |
|---|---:|---:|
| Health | 0.5714 | 7 |
| Economics | 0.6667 | 9 |
| Science | 0.7143 | 7 |
| Miscellaneous | 0.8000 | 10 |
| Nature | 0.8333 | 6 |
| Animals | 0.8333 | 12 |
| History | 0.8462 | 13 |
| Law | 0.8889 | 9 |
| Philosophy | 0.9231 | 13 |
| Space Exploration | 0.9444 | 18 |

Eleven test topics have perfect accuracy, but the perfect scores should be interpreted with their sample sizes in mind. Several topics contain only 2 to 7 pairs, so one or two additional errors could substantially change their percentages. The lowest test scores are also based on relatively small samples. Topic-level rankings are useful for diagnosing where the method struggles, but they should not yet be treated as stable estimates of topic reliability.

## 5. Main interpretation

Taken together, the results support four guarded conclusions:

1. **The confidence-only method detects token-level uncertainty, but uncertainty is not equivalent to factual incorrectness.** It flags two of three correct demonstration answers and flags several answers that are labelled correct in the test suite.
2. **The current full-K rule is operationally inactive on the 13-case suite.** Since it never triggers, it cannot currently be compared meaningfully with confidence-only in terms of detection quality.
3. **The replicated pairwise scorer has strong overall discrimination.** ROC AUC is above 0.92 on both splits, and pairwise accuracy exceeds 0.92 on both.
4. **Generalization is weaker for particular topics.** Health is weak on both evaluation and test data, while Economics and Science become especially problematic on the test split. This points toward domain sensitivity, data variation, or insufficient topic coverage.

These conclusions concern the observed exports only. They do not establish that the method is calibrated, that it improves answer factuality, or that it will behave similarly on a larger deployment distribution.

## 6. Gaps and threats to validity

### 6.1 Tokenization effects

The method operates partly at token level, but the tokenizer splits words and names into subwords. A single uncertain subword can cause an otherwise correct answer to be flagged. Future analyses should report both token-level and word-level or answer-span-level measures.

### 6.2 Temporal and ambiguous questions

Questions such as the last president of the United States are time-sensitive. The result export has a current answer but no timestamp or reference date. Such items need explicit temporal framing; otherwise, a disagreement may reflect changing truth conditions rather than model failure.

### 6.3 The detector fires on syntactic scaffolding, not just factual uncertainty.

Verbs that lack real information trigger low confidence scores and lead to retrieval even when unnecessary. Additionally, restating the question in the response flags the sentence as low confidence.

## 7. Recommended future approach

### 7.1 Reduce tokenization artefacts

Evaluate confidence at multiple levels:

- token level;
- reconstructed word level;
- named-entity or answer-span level;
- complete answer level.

For example, combine subword scores for `Jupiter` or `Shakespeare` before deciding that the answer contains an uncertain token. Compare mean, minimum, lower quantile, and proportion-below-threshold aggregations rather than relying on one statistic.
