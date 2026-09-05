# Methodology

## 1. Goal

The detector learns directions in a language model's hidden-state space that separate text phrased confidently from text phrased unconfidently. Those directions are then used to assign confidence scores to generated answer tokens.

The current implementation uses Mistral-7B-Instruct-v0.1 and transformer layers 10 through 25.

## 2. Training data construction

For each topic, the source JSON contains confident and unconfident statements. Each confident statement is paired with an unconfident statement.

Both statements are tokenized and converted into progressively truncated prefixes. If the two statements yield different numbers of prefixes, `zip` keeps only the shared number of truncations.

Each confident prefix is wrapped with:

`[INST] Pretend you're a confident person making statements about the world. [/INST]`

Each unconfident prefix is wrapped with:

`[INST] Pretend you're an unconfident person making statements about the world. [/INST]`

## 3. Extracting the Representation

To measure what the model "knows," we extract the hidden representation of a text input $x$ at a specific layer $l$.

* **Representative Token:** We look at the final real token in the sequence.

## 4. Measuring Differences Between Paired Examples

We train the detector using pairs of statements (one confident, one unconfident).

1. Training examples are grouped into consecutive pairs in a flattened array.

```text
[ Pair i: Confident ]   --->  h_(2i)
[ Pair i: Unconfident ] --->  h_(2i+1)   ==>   Δh_i^(l) = h_(2i)^(l) - h_(2i+1)^(l)
```

2. For pair $i$ at layer $l$, we compute the raw vector difference:

$$\Delta h_i^{(l)} = h_{2i}^{(l)} - h_{2i+1}^{(l)}$$

> **Note:** Because the order within each pair is randomized, this difference might be $(\text{Confident} - \text{Unconfident})$ or $(\text{Unconfident} - \text{Confident})$. PCA finds the axis of variation regardless of orientation, which is corrected in later steps.


## 5. Centering the Difference Vectors

Before running PCA, we center the difference vectors around their mean.

1. Calculate the average difference vector across all $N$ training pairs:

$$\mu_l = \frac{1}{N} \sum_{i=1}^{N} \Delta h_i^{(l)}$$

2. Subtract this mean from each pair's difference to get the centered difference $z_i^{(l)}$:

$$z_i^{(l)} = \Delta h_i^{(l)} - \mu_l$$


## 6. Finding the "Confidence Axis" via PCA

We identify the direction in the model's hidden space that captures the most variance between confident and unconfident statements.

For each selected layer independently, we fit a 1-component PCA to find a unit vector $v_l$ that maximizes projected variance:

$$v_l = \arg\max_{\|v\|=1} \sum_i \left( z_i^{(l)} \cdot v \right)^2$$

This unit vector $v_l$ represents the **dominant axis of variation** between confident and unconfident representations.


## 7. Aligning the Direction (Sign Correction)

PCA determines the line of variation, but not its direction ($v_l$ and $-v_l$ are equivalent). We orient $v_l$ so that positive projections reliably indicate higher confidence.

1. Center the original (un-differenced) hidden states using $\mu_l$ and project them onto $v_l$.
2. For each training pair, check whether the known confident statement gets the higher or lower projection.
3. Compute the proportions:
   * $p_{\text{max}}$: Fraction of pairs where the confident statement has the **maximum** projection.
   * $p_{\text{min}}$: Fraction of pairs where the confident statement has the **minimum** projection.
4. Determine the direction sign:

$$\text{sign}_l = \text{sign}(p_{\text{max}} - p_{\text{min}})$$


## 8. Scoring a Statement

To score a new text $x$ at layer $l$, center its representation, project it onto $v_l$, and apply the sign correction:

$$m_l(x) = \text{sign}_l \cdot \left( (h_l(x) - \mu_l) \cdot v_l \right)$$

To get the overall **Statement Confidence Score**, average $m_l(x)$ across all evaluated layers $L = \{10, 11, \dots, 25\}$:

$$m(x) = \frac{1}{|L|} \sum_{l \in L} m_l(x)$$


## 9. Evaluating Performance (Synthetic Testing)

To evaluate the detector, test texts are arranged as alternating pairs:

$$\text{Texts} = [\text{confident}_1, \text{unconfident}_1, \text{confident}_2, \text{unconfident}_2, \dots]$$

*(If the total text count is odd, the trailing statement is discarded.)*

Binary labels are generated as $[1, 0, 1, 0, \dots]$. Two key metrics are computed:

* **ROC-AUC:** Overall score separation between confident and unconfident texts.
* **Pairwise Accuracy:** Fraction of pairs satisfying $\text{score}(\text{confident}) > \text{score}(\text{unconfident})$.


## 10. Scoring Generated Answer Tokens

During inference, we evaluate confidence token-by-token as the model generates an answer.

```text
+-----------------------------------------------------------------------+
|  Input Prompt                                                         |
|  [INST] Answer the question directly and concisely... QUESTION [/INST] |
+-----------------------------------------------------------------------+
                                    │
                                    ▼
+-----------------------------------------------------------------------+
|  Generated Answer: "The capital of France is Paris."                  |
+-----------------------------------------------------------------------+
                                    │
                                    ▼
             Extract Hidden States for Every Generated Token
                                    │
                                    ▼
        Apply Signed PCA Projection across Layers L = {10..25}
                                    │
                                    ▼
        Raw Token Confidence Scores: [m_0, m_1, m_2, ..., m_n]
```

1. Format the question inside the instruction template.
2. Generate the answer deterministically (`do_sample=False`).
3. Pass the full `Prompt + Answer` sequence back through the model to extract hidden states for each generated token $t_i$.
4. Compute raw token score $m_i$ by applying the signed PCA projection across layers $L$ and averaging them.


## 11. Causal Score Normalization

Raw scores fluctuate depending on context. To normalize scores to $[0, 1]$ without looking into the future, we use a **causal sliding window**.

For token $i$, define the causal window using current and past raw scores:

$$W_i = \{m_0, m_1, \dots, m_i\}$$

Find the local minimum and maximum:

$$\text{lo}_i = \min(W_i), \quad \text{hi}_i = \max(W_i)$$

Compute the normalized score $\tilde{m}_i$:

$$\tilde{m}_i = \begin{cases} 0.5 & \text{if } i = 0 \text{ or } (\text{hi}_i - \text{lo}_i) < 10^{-8} \\[6pt] \dfrac{m_i - \text{lo}_i}{\text{hi}_i - \text{lo}_i} & \text{otherwise} \end{cases}$$

This ensures $\tilde{m}_i \in [0, 1]$ while strictly respecting causality.


## 12. Filtering Non-Content Tokens

Punctuation, stop words, and formatting markers do not reflect factual confidence. We construct a **Content Mask** $s_i$:

1. Remove special tokenizer markers (`▁`, `Ġ`) and convert text to lowercase.
2. Filter out standard NLTK English stop words and special formatting tokens.
3. Define the mask:

$$s_i = \begin{cases} 1 & \text{if } t_i \text{ is a content token} \\ 0 & \text{if } t_i \text{ is a stop word or ignored token} \end{cases}$$

Only content tokens ($s_i = 1$) participate in retrieval triggering decisions.


## 13. Standalone Confidence-Only Experiment

In the simplified setup, we bypass $E$ and directly measure token confidence:

$$\text{confidence}_i = \tilde{m}_i$$

* **Confident:** $\text{confidence}_i \ge 0.5$
* **Unconfident:** $\text{confidence}_i < 0.5$

The question-level confidence trigger activates if **at least one** content token ($s_i = 1$) has confidence below $0.5$.


## 16. Pipeline Architecture

```text
================================================================================
                                TRAINING PHASE
================================================================================
[ JSON Statements ] 
       │
       ▼
[ Extract Hidden States (Mistral) ] ──► [ Compute Paired Differences ]
                                                  │
                                                  ▼
[ Sign Correction ] ◄── [ PCA per Layer ] ◄── [ Mean Centering ]
       │
       ▼
[ Save Projection Weights & Means ]

================================================================================
                               INFERENCE PHASE
================================================================================
[ Input Question ]
       │
       ▼
[ Generate Answer (Mistral) ] ──► [ Extract Token Hidden States ]
                                                  │
                                                  ▼
[ Calculate Trigger K(t_i) ] ◄── [ Causal Normalization & Masking ] ◄── [ Signed PCA Projection ]
       │
       ▼
[ Trigger Retrieval if K(t_i) > τ ]
```
