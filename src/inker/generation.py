"""
Live token-level generation and retrieval-trigger evaluation for INKER.

This module connects the trained internal confidence representation detector
to autoregressive language-model generation.

Unlike a post-hoc implementation that first generates an entire answer and
then analyzes it, this module evaluates confidence LIVE during generation.

For each generated token t_i, the procedure is:

    1. Generate token t_i.
    2. Feed t_i through the language model.
    3. Extract its hidden state from every confidence-detector layer.
    4. Project those hidden states onto the learned confidence directions.
    5. Average the signed layer projections to obtain raw confidence m_i.
    6. Causally normalize the confidence score to obtain m_tilde_i.
    7. Compute the content mask s_i.
    8. Combine internal confidence with external query complexity:

           K(t_i) = (E - m_tilde_i) * s_i

    9. If:

           K(t_i) > threshold

       stop generation immediately and report that retrieval should occur.

Because this decision happens before the next token is generated, the module
implements an ONLINE / LIVE IE-KRT retrieval trigger.

Important
---------
This module implements the LIVE RETRIEVAL TRIGGER.

Therefore, when a trigger occurs, this module stops generation and returns a
structured result describing:

    - which token triggered,
    - its confidence,
    - its K(t_i),
    - the partial generated answer,
    - and the complete token history available up to the trigger.

That output can later be passed directly to the retrieval module.

Internal confidence
-------------------
For generated token t_i and detector layer l:

    m_i^(l)
        = sign^(l)
          * ((h_i^(l) - mu^(l)) @ v^(l))

where:

    h_i^(l)
        Hidden representation of token t_i at layer l.

    mu^(l)
        Mean paired-difference representation stored during confidence
        direction training.

    v^(l)
        PCA confidence direction.

    sign^(l)
        Orientation indicating which side of the PCA axis corresponds to
        greater confidence.

Layer scores are averaged:

    m_i = mean_l(m_i^(l))

Causal normalization
--------------------
At generation step i:

    m_tilde_i = scale([m_0, ..., m_i])[-1]

using causal min-max scaling.

Only scores that already exist at generation step i are used.

No future token confidence is available or used.

Content mask
------------
Confidence is calculated BEFORE the content mask is applied.

Every ordinary generated token therefore contributes to the causal confidence
history.

After normalization:

    s_i = 1
        for content-bearing tokens.

    s_i = 0
        for stop words, punctuation, and formatting tokens.

Thus:

    K(t_i) = (E - m_tilde_i) * s_i

automatically gives:

    K(t_i) = 0

for masked tokens.

This separation is important because the content mask should determine
whether a token is allowed to trigger retrieval, not alter the confidence
normalization process itself.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch


# ==========================================================================
# Stop words
# ==========================================================================

FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "on", "or", "our",
    "ours", "ourselves", "she", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
}


def _load_stop_words() -> set[str]:
    """
    Load the English NLTK stop-word list when available.

    No automatic download is performed here. This avoids unexpected network
    requests when the repository is imported on Colab, a server, or an
    offline machine.

    If NLTK or its stop-word corpus is unavailable, the repository's built-in
    fallback list is used.
    """

    try:
        from nltk.corpus import stopwords

        return set(
            stopwords.words("english")
        )

    except Exception:
        return set(
            FALLBACK_STOP_WORDS
        )


STOP_WORDS = _load_stop_words()


# ==========================================================================
# Special / formatting tokens
# ==========================================================================

SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "<mask>",
}


def clean_token_text(
    token: str,
) -> str:
    """
    Convert a tokenizer-level token into plain lowercase text.

    SentencePiece tokenizers commonly represent preceding whitespace with:

        ▁

    while GPT-style tokenizers commonly use:

        Ġ

    Examples
    --------
    ``▁France`` -> ``france``
    ``Ġthe``    -> ``the``
    """

    return (
        token
        .replace("▁", "")
        .replace("Ġ", "")
        .lower()
        .strip()
    )


def is_formatting_token(
    token: str,
) -> bool:
    """
    Return True when a token contains no letters or digits.

    Examples
    --------
    "."        -> True
    ","        -> True
    ":"        -> True
    "▁"        -> True
    "France"   -> False
    """

    stripped = re.sub(
        r"[^a-zA-Z0-9]",
        "",
        token,
    )

    return stripped == ""


def get_content_mask(
    token: str,
) -> int:
    """
    Compute the INKER token mask s_i.

    Returns
    -------
    1
        Token is content-bearing and may trigger retrieval.

    0
        Token is a stop word, punctuation token, formatting token,
        or special token.

    Important
    ---------
    This mask is calculated AFTER confidence scoring.

    Therefore s_i determines whether the token can trigger retrieval, but
    does not determine whether it participates in confidence normalization.
    """

    if token in SPECIAL_TOKENS:
        return 0

    if is_formatting_token(
        token
    ):
        return 0

    cleaned = clean_token_text(
        token
    )

    if cleaned == "":
        return 0

    if cleaned in STOP_WORDS:
        return 0

    return 1


# ==========================================================================
# Causal normalization
# ==========================================================================

def compute_current_causal_normalized_score(
    raw_history: Sequence[float],
) -> float:
    """
    Compute m_tilde_i for the most recently generated token.

    Given the raw confidence history:

        [m_0, m_1, ..., m_i]

    calculate:

        lo_i = min(m_0, ..., m_i)

        hi_i = max(m_0, ..., m_i)

    and:

                        m_i - lo_i
        m_tilde_i = ----------------
                       hi_i - lo_i

    If only one score exists, or if all values are equal, a neutral value of
    0.5 is returned.

    This computation is causal because it never uses future token scores.
    """

    if not raw_history:
        raise ValueError(
            "raw_history cannot be empty."
        )

    if len(raw_history) == 1:
        return 0.5

    lo = min(
        raw_history
    )

    hi = max(
        raw_history
    )

    if hi - lo < 1e-8:
        return 0.5

    return float(
        (
            raw_history[-1]
            - lo
        )
        / (
            hi
            - lo
        )
    )


def compute_causal_normalized_scores(
    raw_vals: Sequence[float],
) -> List[float]:
    """
    Compute causal normalized confidence for an entire sequence.

    This helper is retained because experiment and visualization scripts may
    still need to normalize a previously stored raw-confidence sequence.

    Each position is normalized using only its own prefix.
    """

    normalized = []

    for i in range(
        len(raw_vals)
    ):
        normalized.append(
            compute_current_causal_normalized_score(
                raw_vals[
                    :i + 1
                ]
            )
        )

    return normalized


# ==========================================================================
# Complexity evaluator compatibility
# ==========================================================================

def _extract_complexity_score(
    result: Any,
) -> float:
    """
    Extract the continuous query-complexity value E.

    Supported complexity evaluator outputs:

    1. Numeric value:

           0.73

    2. Dictionary:

           {
               "E": 0.73,
               ...
           }

    3. Object exposing:

           result.E

    The latter supports the ``ComplexityResult`` object returned by
    ``src/inker/complexity.py``.
    """

    if isinstance(
        result,
        (float, int, np.floating),
    ):
        E = float(
            result
        )

    elif isinstance(
        result,
        dict,
    ):
        if "E" not in result:
            raise ValueError(
                "Complexity result dictionary does not contain 'E'."
            )

        E = float(
            result["E"]
        )

    elif hasattr(
        result,
        "E",
    ):
        E = float(
            result.E
        )

    else:
        raise TypeError(
            "complexity_fn must return a numeric E value, "
            "a dictionary containing 'E', or an object exposing '.E'."
        )

    if not 0.0 <= E <= 1.0:
        raise ValueError(
            f"Complexity score E must lie in [0, 1]. "
            f"Received: {E}"
        )

    return E


# ==========================================================================
# Repetition penalty
# ==========================================================================

def _apply_repetition_penalty(
    logits: torch.Tensor,
    sequence_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply the standard Hugging Face-style repetition penalty.

    This reproduces the behavior used by ``model.generate`` for greedy
    decoding:

    - positive logits for previously seen tokens are divided by the penalty;
    - negative logits are multiplied by the penalty.

    Parameters
    ----------
    logits:
        Next-token logits of shape:

            [1, vocabulary_size]

    sequence_ids:
        All token IDs currently present in the sequence.

    penalty:
        Repetition penalty. ``1.0`` disables the penalty.
    """

    if penalty == 1.0:
        return logits

    if penalty <= 0.0:
        raise ValueError(
            "repetition_penalty must be greater than 0."
        )

    adjusted = logits.clone()

    previously_seen = torch.unique(
        sequence_ids
    )

    token_scores = adjusted[
        0,
        previously_seen,
    ]

    token_scores = torch.where(
        token_scores < 0,
        token_scores * penalty,
        token_scores / penalty,
    )

    adjusted[
        0,
        previously_seen,
    ] = token_scores

    return adjusted


# ==========================================================================
# Confidence projection
# ==========================================================================

def _score_hidden_state(
    hidden_states: Sequence[torch.Tensor],
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
) -> tuple[float, Dict[int, float]]:
    """
    Project one generated token's hidden representation onto the learned
    confidence direction.

    Parameters
    ----------
    hidden_states:
        ``outputs.hidden_states`` returned by the language model after feeding
        the newly generated token.

    rep_reader:
        Learned confidence detector containing:

            directions
            H_train_means
            signs

    layers:
        Transformer layers used by the detector.

    Returns
    -------
    raw_score:
        Mean signed confidence projection across detector layers.

    layer_scores:
        Individual signed confidence score for every detector layer.
    """

    layer_scores: Dict[int, float] = {}

    for layer in layers:

        # The current forward pass contains only the newly generated token
        # when KV caching is enabled. Therefore index -1 refers to that token.
        hidden_state = (
            hidden_states[layer][
                0,
                -1,
                :
            ]
            .float()
            .cpu()
            .numpy()
        )

        train_mean = (
            np.asarray(
                rep_reader[
                    "H_train_means"
                ][layer]
            )
            .reshape(-1)
        )

        direction = (
            np.asarray(
                rep_reader[
                    "directions"
                ][layer]
            )
            .reshape(-1)
        )

        sign = float(
            rep_reader[
                "signs"
            ][layer]
        )

        if (
            hidden_state.shape[0]
            != train_mean.shape[0]
            or hidden_state.shape[0]
            != direction.shape[0]
        ):
            raise ValueError(
                f"Dimension mismatch at layer {layer}: "
                f"hidden={hidden_state.shape}, "
                f"mean={train_mean.shape}, "
                f"direction={direction.shape}"
            )

        centered = (
            hidden_state
            - train_mean
        )

        projection = float(
            np.dot(
                centered,
                direction,
            )
        )

        signed_score = (
            sign
            * projection
        )

        layer_scores[
            int(layer)
        ] = float(
            signed_score
        )

    if not layer_scores:
        raise ValueError(
            "No detector layers were provided."
        )

    raw_score = float(
        np.mean(
            list(
                layer_scores.values()
            )
        )
    )

    return (
        raw_score,
        layer_scores,
    )


# ==========================================================================
# Prompt construction
# ==========================================================================

def _build_prompt(
    question: str,
    system_message: str,
) -> str:
    """
    Construct the Mistral instruction prompt used for generation.
    """

    return (
        f"[INST] "
        f"{system_message}\n\n"
        f"{question.strip()} "
        f"[/INST]"
    )


# ==========================================================================
# Live generation engine
# ==========================================================================

def _live_generate(
    question: str,
    tokenizer: Any,
    model: Any,
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
    *,
    E: Optional[float],
    trigger_threshold: float,
    confidence_threshold: float,
    max_new_tokens: int,
    repetition_penalty: float,
    system_message: str,
    stop_on_trigger: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Internal live autoregressive generation engine.

    The important generation order is:

        current prefix
            ↓
        choose next token
            ↓
        feed newly generated token through model
            ↓
        obtain h_i
            ↓
        calculate m_i
            ↓
        calculate m_tilde_i
            ↓
        calculate s_i
            ↓
        calculate K(t_i)
            ↓
        check trigger
            ↓
        only then allow token i+1

    This means retrieval decisions are made online rather than after the
    response has already been generated.
    """

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than zero."
        )

    if not 0.0 <= trigger_threshold <= 1.0:
        raise ValueError(
            "trigger_threshold must lie in [0, 1]."
        )

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must lie in [0, 1]."
        )

    if repetition_penalty <= 0:
        raise ValueError(
            "repetition_penalty must be greater than zero."
        )

    model.eval()

    prompt = _build_prompt(
        question,
        system_message,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(
        model.device
    )

    prompt_ids = inputs[
        "input_ids"
    ]

    if "attention_mask" in inputs:
        attention_mask = inputs[
            "attention_mask"
        ]

    else:
        attention_mask = torch.ones_like(
            prompt_ids
        )

    # ------------------------------------------------------------------
    # Initial forward pass.
    #
    # The final prompt position predicts the first generated token.
    # ------------------------------------------------------------------

    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    past_key_values = (
        outputs.past_key_values
    )

    next_token_logits = (
        outputs.logits[
            :,
            -1,
            :
        ]
    )

    full_sequence_ids = (
        prompt_ids.clone()
    )

    generated_ids: List[int] = []
    token_entries: List[Dict[str, Any]] = []

    # Raw confidence history used for causal normalization.
    #
    # Every non-special generated token contributes, INCLUDING:
    #
    #     - stop words
    #     - punctuation
    #
    # s_i is applied only after m_tilde_i is calculated.
    raw_history: List[float] = []

    first_trigger: Optional[Dict[str, Any]] = None

    eos_token_ids = set()

    if tokenizer.eos_token_id is not None:

        if isinstance(
            tokenizer.eos_token_id,
            (list, tuple, set),
        ):
            eos_token_ids.update(
                int(x)
                for x in tokenizer.eos_token_id
            )

        else:
            eos_token_ids.add(
                int(
                    tokenizer.eos_token_id
                )
            )

    # ==================================================================
    # LIVE AUTOREGRESSIVE LOOP
    # ==================================================================

    for token_index in range(
        max_new_tokens
    ):

        # --------------------------------------------------------------
        # 1. Apply repetition penalty and greedily choose token t_i.
        # --------------------------------------------------------------

        adjusted_logits = (
            _apply_repetition_penalty(
                logits=next_token_logits,
                sequence_ids=full_sequence_ids,
                penalty=repetition_penalty,
            )
        )

        next_token_id = int(
            torch.argmax(
                adjusted_logits,
                dim=-1,
            ).item()
        )

        generated_ids.append(
            next_token_id
        )

        # --------------------------------------------------------------
        # 2. Decode tokenizer-level representation.
        # --------------------------------------------------------------

        token = tokenizer.convert_ids_to_tokens(
            next_token_id
        )

        decoded_token = tokenizer.decode(
            [next_token_id],
            skip_special_tokens=True,
        )

        # --------------------------------------------------------------
        # EOS ends generation.
        #
        # We do not use EOS as a confidence / retrieval candidate.
        # --------------------------------------------------------------

        if next_token_id in eos_token_ids:

            token_entries.append({
                "token_index":
                    token_index,

                "token":
                    token,

                "decoded_token":
                    decoded_token,

                "token_id":
                    next_token_id,

                "raw_score":
                    None,

                "m_tilde":
                    None,

                "s_i":
                    0,

                "K":
                    None,

                "is_content":
                    False,

                "is_special":
                    True,

                "triggered":
                    False,

                "confidence_status":
                    "SPECIAL",
            })

            break

        # --------------------------------------------------------------
        # 3. Append the new token to the complete sequence.
        # --------------------------------------------------------------

        token_tensor = torch.tensor(
            [[next_token_id]],
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )

        full_sequence_ids = torch.cat(
            [
                full_sequence_ids,
                token_tensor,
            ],
            dim=1,
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (
                        attention_mask.shape[0],
                        1,
                    ),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        # --------------------------------------------------------------
        # 4. Feed t_i through the model.
        #
        # Because the prompt prefix is already stored in past_key_values,
        # only the newly generated token must be processed.
        #
        # The resulting hidden state at position -1 is therefore the
        # representation of the token that was just generated.
        #
        # The returned logits predict token t_(i+1).
        # --------------------------------------------------------------

        with torch.no_grad():

            token_outputs = model(
                input_ids=token_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

        past_key_values = (
            token_outputs.past_key_values
        )

        next_token_logits = (
            token_outputs.logits[
                :,
                -1,
                :
            ]
        )

        # --------------------------------------------------------------
        # 5. Calculate raw internal confidence m_i.
        #
        # IMPORTANT:
        #
        # This happens for stop words and punctuation as well.
        # --------------------------------------------------------------

        raw_score, layer_scores = (
            _score_hidden_state(
                hidden_states=token_outputs.hidden_states,
                rep_reader=rep_reader,
                layers=layers,
            )
        )

        raw_history.append(
            raw_score
        )

        # --------------------------------------------------------------
        # 6. Causal normalization.
        #
        # m_tilde_i depends ONLY on:
        #
        #     m_0 ... m_i
        #
        # and therefore contains no future information.
        # --------------------------------------------------------------

        m_tilde = (
            compute_current_causal_normalized_score(
                raw_history
            )
        )

        # --------------------------------------------------------------
        # 7. Determine s_i AFTER confidence calculation.
        # --------------------------------------------------------------

        s_i = get_content_mask(
            token
        )

        is_content = bool(
            s_i == 1
        )

        # --------------------------------------------------------------
        # 8. Confidence-only status.
        # --------------------------------------------------------------

        if s_i == 0:

            confidence_status = (
                "MASKED"
            )

        elif m_tilde >= confidence_threshold:

            confidence_status = (
                "CONFIDENT"
            )

        else:

            confidence_status = (
                "UNCONFIDENT"
            )

        # --------------------------------------------------------------
        # 9. Full INKER activation.
        #
        # If E is absent, this generation is running in confidence-only
        # mode and K remains None.
        # --------------------------------------------------------------

        if E is None:

            K = None
            triggered = False

        else:

            K = float(
                (
                    E
                    - m_tilde
                )
                * s_i
            )

            triggered = bool(
                s_i == 1
                and K > trigger_threshold
            )

        entry = {
            "token_index":
                token_index,

            "token":
                token,

            "decoded_token":
                decoded_token,

            "token_id":
                next_token_id,

            "raw_score":
                float(
                    raw_score
                ),

            "layer_scores":
                layer_scores,

            "m_tilde":
                float(
                    m_tilde
                ),

            "s_i":
                int(
                    s_i
                ),

            "K":
                K,

            "is_content":
                is_content,

            "is_special":
                False,

            "triggered":
                triggered,

            "confidence_status":
                confidence_status,
        }

        token_entries.append(
            entry
        )

        # --------------------------------------------------------------
        # 10. Live console output.
        # --------------------------------------------------------------

        if verbose:

            if E is None:

                print(
                    f"[{token_index:02d}] "
                    f"{token!r:<18} "
                    f"m={raw_score:>9.4f} "
                    f"m_tilde={m_tilde:>6.4f} "
                    f"s_i={s_i} "
                    f"{confidence_status}"
                )

            else:

                status = (
                    " <-- RETRIEVAL TRIGGER"
                    if triggered
                    else ""
                )

                print(
                    f"[{token_index:02d}] "
                    f"{token!r:<18} "
                    f"m={raw_score:>9.4f} "
                    f"m_tilde={m_tilde:>6.4f} "
                    f"s_i={s_i} "
                    f"K={K:>7.4f}"
                    f"{status}"
                )

        # --------------------------------------------------------------
        # 11. Stop IMMEDIATELY when retrieval is triggered.
        #
        # No later token is generated.
        # --------------------------------------------------------------

        if triggered:

            if first_trigger is None:
                first_trigger = dict(
                    entry
                )

            if stop_on_trigger:
                break

    # ==================================================================
    # Final decoding
    # ==================================================================

    answer_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # Only content tokens are used for aggregate detector metrics.
    content_entries = [
        entry
        for entry in token_entries
        if (
            entry["m_tilde"] is not None
            and entry["s_i"] == 1
        )
    ]

    content_confidences = [
        entry["m_tilde"]
        for entry in content_entries
    ]

    K_values = [
        entry["K"]
        for entry in content_entries
        if entry["K"] is not None
    ]

    low_confidence_tokens = [
        entry
        for entry in content_entries
        if (
            entry["m_tilde"]
            < confidence_threshold
        )
    ]

    return {
        "question":
            question,

        "answer_text":
            answer_text,

        "E":
            E,

        "trigger_threshold":
            trigger_threshold,

        "confidence_threshold":
            confidence_threshold,

        "retrieval_triggered":
            first_trigger is not None,

        "stopped_for_retrieval":
            bool(
                first_trigger is not None
                and stop_on_trigger
            ),

        "trigger_token":
            (
                first_trigger
                if first_trigger is not None
                else None
            ),

        "num_generated_tokens":
            len(
                generated_ids
            ),

        "num_scored_tokens":
            len(
                raw_history
            ),

        "num_content_tokens":
            len(
                content_entries
            ),

        "num_low_confidence_tokens":
            len(
                low_confidence_tokens
            ),

        "mean_m_tilde":
            (
                float(
                    np.mean(
                        content_confidences
                    )
                )
                if content_confidences
                else None
            ),

        "min_m_tilde":
            (
                float(
                    np.min(
                        content_confidences
                    )
                )
                if content_confidences
                else None
            ),

        "max_m_tilde":
            (
                float(
                    np.max(
                        content_confidences
                    )
                )
                if content_confidences
                else None
            ),

        "max_K":
            (
                float(
                    np.max(
                        K_values
                    )
                )
                if K_values
                else None
            ),

        "would_trigger_confidence_only":
            len(
                low_confidence_tokens
            )
            > 0,

        "token_entries":
            token_entries,
    }


# ==========================================================================
# Public: full live INKER trigger
# ==========================================================================

def answer_with_confidence(
    question: str,
    tokenizer: Any,
    model: Any,
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
    complexity_fn: Callable[[str], Any],
    threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    max_new_tokens: int = 60,
    repetition_penalty: float = 1.1,
    verbose: bool = True,
    stop_on_trigger: bool = True,
    system_message: str = (
        "Answer the question directly and concisely. "
        "Do not provide extra context."
    ),
) -> Dict[str, Any]:
    """
    Generate an answer with LIVE INKER retrieval-trigger evaluation.

    External query complexity is calculated once before generation:

        E = complexity_fn(question)

    Then every generated token is evaluated online:

        m_i
            ↓
        m_tilde_i
            ↓
        s_i
            ↓
        K(t_i) = (E - m_tilde_i) * s_i

    If:

        K(t_i) > threshold

    the generation stops immediately before token t_(i+1) is generated.

    Parameters
    ----------
    threshold:
        Retrieval activation threshold tau.

    confidence_threshold:
        Threshold used only for the confidence-only comparison baseline.

    stop_on_trigger:
        If True, generation stops at the first retrieval trigger.

        This should normally remain True when replicating live IE-KRT.

        Setting it to False is useful only for diagnostic experiments where
        the researcher wants to observe all potential trigger positions.

    Returns
    -------
    dict
        Complete live-generation result including complexity, generated text,
        token confidence values, K values, and the first trigger event.
    """

    complexity_result = complexity_fn(
        question
    )

    E = _extract_complexity_score(
        complexity_result
    )

    if verbose:

        print(
            "=" * 80
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Query complexity E: {E:.4f}"
        )

        print(
            f"Retrieval threshold tau: {threshold:.4f}"
        )

        print(
            "-" * 80
        )

    result = _live_generate(
        question=question,
        tokenizer=tokenizer,
        model=model,
        rep_reader=rep_reader,
        layers=layers,
        E=E,
        trigger_threshold=threshold,
        confidence_threshold=confidence_threshold,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        system_message=system_message,
        stop_on_trigger=stop_on_trigger,
        verbose=verbose,
    )

    # Preserve the complete external evaluator result when possible.
    if hasattr(
        complexity_result,
        "to_dict",
    ):

        result[
            "complexity_result"
        ] = (
            complexity_result.to_dict()
        )

    else:

        result[
            "complexity_result"
        ] = (
            complexity_result
        )

    if verbose:

        print(
            "-" * 80
        )

        print(
            f"Generated answer: "
            f"{result['answer_text']}"
        )

        if result[
            "retrieval_triggered"
        ]:

            trigger = result[
                "trigger_token"
            ]

            print(
                "\nRETRIEVAL TRIGGERED"
            )

            print(
                f"Token: "
                f"{trigger['token']}"
            )

            print(
                f"m_tilde: "
                f"{trigger['m_tilde']:.4f}"
            )

            print(
                f"s_i: "
                f"{trigger['s_i']}"
            )

            print(
                f"K(t_i): "
                f"{trigger['K']:.4f}"
            )

            print(
                "\nGeneration stopped before "
                "the next token so retrieval can be performed."
            )

        else:

            print(
                "\nNo retrieval trigger occurred."
            )

        print(
            "=" * 80
        )

    return result


# ==========================================================================
# Public: live confidence-only baseline
# ==========================================================================

def answer_with_confidence_only(
    question: str,
    tokenizer: Any,
    model: Any,
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
    expected_answer: Optional[str] = None,
    threshold: float = 0.5,
    max_new_tokens: int = 60,
    repetition_penalty: float = 1.1,
    verbose: bool = False,
    system_message: str = (
        "Answer the question directly and concisely. "
        "Do not provide extra context."
    ),
) -> Dict[str, Any]:
    """
    Run LIVE generation using only the internal confidence detector.

    This baseline does not use query complexity E.

    Content tokens are classified as:

        CONFIDENT
            m_tilde_i >= threshold

        UNCONFIDENT
            m_tilde_i < threshold

    Stop words and punctuation still contribute to the causal confidence
    history but receive s_i=0 and status ``MASKED``.

    Unlike the full INKER mode, this function does not stop generation when
    confidence becomes low. It records all low-confidence positions so the
    complete generated answer can be inspected.
    """

    result = _live_generate(
        question=question,
        tokenizer=tokenizer,
        model=model,
        rep_reader=rep_reader,
        layers=layers,
        E=None,
        trigger_threshold=0.5,
        confidence_threshold=threshold,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        system_message=system_message,
        stop_on_trigger=False,
        verbose=verbose,
    )

    result[
        "expected_answer"
    ] = expected_answer

    result[
        "threshold"
    ] = threshold

    # Compatibility with the previous confidence-only interface.
    result[
        "mean_confidence"
    ] = result[
        "mean_m_tilde"
    ]

    result[
        "min_confidence"
    ] = result[
        "min_m_tilde"
    ]

    result[
        "max_confidence"
    ] = result[
        "max_m_tilde"
    ]

    result[
        "would_trigger"
    ] = result[
        "would_trigger_confidence_only"
    ]

    return result
