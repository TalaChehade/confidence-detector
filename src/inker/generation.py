import re
import numpy as np
import torch

try:
    import nltk
    from nltk.corpus import stopwords

    try:
        STOP_WORDS = set(
            stopwords.words("english")
        )
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(
            stopwords.words("english")
        )
except Exception:
    # This fallback should rarely be used. Installing NLTK and its stopword
    # corpus is recommended for faithful reproduction.
    STOP_WORDS = set()


SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "<mask>",
    "<0x0A>",
    "▁",
}


def is_formatting_token(token):
    stripped = re.sub(
        r"[^a-zA-Z0-9]",
        "",
        token,
    )
    return stripped == ""


def compute_causal_normalized_scores(raw_vals):
    """
    Eq. (2): m_tilde_i = scale([m_0, ..., m_i])[-1].

    The normalization is causal and never uses future token scores.
    """
    causal_scores = []

    for i in range(len(raw_vals)):
        window = raw_vals[:i + 1]

        if len(window) == 1:
            causal_scores.append(0.5)
            continue

        lo, hi = min(window), max(window)

        if hi - lo < 1e-8:
            causal_scores.append(0.5)
        else:
            causal_scores.append(
                (window[-1] - lo)
                / (hi - lo)
            )

    return causal_scores


def _generate_and_score_raw_tokens(
    question,
    tokenizer,
    model,
    rep_reader,
    layers,
    max_new_tokens=60,
    repetition_penalty=1.1,
    system_message=(
        "Answer the question directly and concisely. "
        "Do not provide extra context."
    ),
):
    """
    Shared implementation of the generation + raw confidence projection path.
    This refactors structure only; the numerical operations match the notebook.
    """
    prompt = (
        f"[INST] {system_message}\n\n"
        f"{question.strip()} [/INST]"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_ids = (
        output_ids[0][input_length:]
    )

    answer_text = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    ).strip()

    with torch.no_grad():
        full_outputs = model(
            output_ids,
            output_hidden_states=True,
        )

    answer_tokens = (
        tokenizer.convert_ids_to_tokens(
            answer_ids
        )
    )

    token_entries = []

    for i, token in enumerate(answer_tokens):
        pos = input_length + i

        if (
            token in SPECIAL_TOKENS
            or is_formatting_token(token)
        ):
            token_entries.append({
                "token_index": i,
                "token": token,
                "raw_score": None,
                "s_i": 0,
                "skip": True,
                "is_content": False,
            })
            continue

        clean = (
            token
            .replace("▁", "")
            .replace("Ġ", "")
            .lower()
            .strip()
        )

        s_i = (
            0
            if clean in STOP_WORDS
            else 1
        )

        layer_scores = []

        for layer in layers:
            hs = (
                full_outputs
                .hidden_states[layer][0, pos, :]
                .float()
                .cpu()
                .numpy()
            )

            centered = (
                hs
                - rep_reader[
                    "H_train_means"
                ][layer].flatten()
            )

            proj = np.dot(
                centered,
                rep_reader[
                    "directions"
                ][layer],
            )

            signed_score = (
                rep_reader[
                    "signs"
                ][layer]
                * proj
            )

            layer_scores.append(
                signed_score
            )

        raw_score = float(
            np.mean(layer_scores)
        )

        token_entries.append({
            "token_index": i,
            "token": token,
            "raw_score": raw_score,
            "s_i": s_i,
            "skip": False,
            "is_content": bool(s_i == 1),
        })

    scoreable = [
        e
        for e in token_entries
        if not e["skip"]
    ]

    raw_vals = [
        e["raw_score"]
        for e in scoreable
    ]

    causal_norm = (
        compute_causal_normalized_scores(
            raw_vals
        )
        if len(raw_vals) > 1
        else [0.5] * len(raw_vals)
    )

    for entry, m_tilde in zip(
        scoreable,
        causal_norm,
    ):
        entry["m_tilde"] = float(m_tilde)

    return answer_text, token_entries


def answer_with_confidence(
    question,
    tokenizer,
    model,
    rep_reader,
    layers,
    complexity_fn,
    threshold=0.5,
    max_new_tokens=60,
    repetition_penalty=1.1,
    verbose=True,
):
    """
    Full K(t_i) experiment from the original notebook.

        K(t_i) = (E - m_tilde_i) * s_i

    The notebook also records an older confidence-only comparison trigger as
        (1 - m_tilde_i) > threshold.
    """
    E = complexity_fn(question)

    answer_text, token_entries = (
        _generate_and_score_raw_tokens(
            question=question,
            tokenizer=tokenizer,
            model=model,
            rep_reader=rep_reader,
            layers=layers,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    scoreable = [
        e
        for e in token_entries
        if not e["skip"]
    ]

    for entry in scoreable:
        entry["K"] = (
            (E - entry["m_tilde"])
            * entry["s_i"]
        )

    content_entries = [
        e
        for e in scoreable
        if e["s_i"] == 1
    ]

    K_values = [
        e["K"]
        for e in content_entries
    ]

    would_trigger = any(
        k > threshold
        for k in K_values
    )

    would_trigger_confidence_only = any(
        (1 - e["m_tilde"]) > threshold
        for e in content_entries
    )

    if verbose:
        print(f"Q: {question}")
        print(f"E (complexity proxy): {E}")
        print(f"A: {answer_text}\n")

        print(
            f"{'Token':<18} "
            f"{'m_tilde':>8} "
            f"{'K(t_i)':>8}  "
            f"{'Status'}"
        )
        print("-" * 50)

        for e in scoreable:
            status = (
                "TRIGGER"
                if (
                    e["s_i"] == 1
                    and e["K"] > threshold
                )
                else ""
            )

            print(
                f"{e['token']:<18} "
                f"{e['m_tilde']:>8.4f} "
                f"{e['K']:>8.4f}  "
                f"{status}"
            )

        print(
            f"\nWould trigger "
            f"(full K(t_i) w/ E={E}): "
            f"{would_trigger}"
        )

        print(
            "Would trigger "
            "(confidence-only, no E): "
            f"{would_trigger_confidence_only}"
        )

    return {
        "question": question,
        "answer_text": answer_text,
        "E": E,
        "mean_m_tilde": (
            float(np.mean([
                e["m_tilde"]
                for e in content_entries
            ]))
            if content_entries
            else None
        ),
        "min_m_tilde": (
            float(np.min([
                e["m_tilde"]
                for e in content_entries
            ]))
            if content_entries
            else None
        ),
        "max_K": (
            float(np.max(K_values))
            if K_values
            else None
        ),
        "would_trigger_full": would_trigger,
        "would_trigger_confidence_only":
            would_trigger_confidence_only,
        "token_entries": token_entries,
    }


def answer_with_confidence_only(
    question,
    tokenizer,
    model,
    rep_reader,
    layers,
    expected_answer=None,
    threshold=0.5,
    max_new_tokens=60,
    repetition_penalty=1.1,
):
    """
    Standalone confidence-only token experiment from the later notebook cells.

    Here confidence is m_tilde itself:
        GREEN: confidence >= threshold
        RED:   confidence < threshold

    Trigger if ANY content token has confidence < threshold.
    """
    answer_text, token_entries = (
        _generate_and_score_raw_tokens(
            question=question,
            tokenizer=tokenizer,
            model=model,
            rep_reader=rep_reader,
            layers=layers,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    scoreable = [
        e
        for e in token_entries
        if e["raw_score"] is not None
    ]

    for entry in token_entries:
        if entry["raw_score"] is None:
            entry["m_tilde"] = None
            entry["confidence"] = None
            entry["status"] = "SKIP"
            continue

        # m_tilde was set by the shared helper.
        entry["confidence"] = float(
            entry["m_tilde"]
        )

        entry["status"] = (
            "CONFIDENT"
            if entry["confidence"] >= threshold
            else "UNCONFIDENT"
        )

    content_entries = [
        e
        for e in scoreable
        if e["is_content"]
    ]

    low_confidence_tokens = [
        e
        for e in content_entries
        if e["confidence"] < threshold
    ]

    would_trigger = (
        len(low_confidence_tokens) > 0
    )

    confidences = [
        e["confidence"]
        for e in content_entries
    ]

    if confidences:
        mean_confidence = float(
            np.mean(confidences)
        )
        min_confidence = float(
            np.min(confidences)
        )
        max_confidence = float(
            np.max(confidences)
        )
    else:
        mean_confidence = None
        min_confidence = None
        max_confidence = None

    return {
        "question": question,
        "answer_text": answer_text,
        "expected_answer": expected_answer,
        "threshold": threshold,
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "num_content_tokens":
            len(content_entries),
        "num_low_confidence_tokens":
            len(low_confidence_tokens),
        "would_trigger": would_trigger,
        "token_entries": token_entries,
    }
