"""
INKER retrieval-trigger computation.

This module contains the mathematical retrieval-trigger rule used by IE-KRT.

Given:

    E
        External query-complexity score.

    m_tilde_i
        Causally normalized internal confidence of generated token t_i.

    s_i
        Binary token-content mask.

the token activation is:

    K(t_i) = (E - m_tilde_i) * s_i

where:

    E in [0, 1]

    m_tilde_i in [0, 1]

    s_i in {0, 1}

A retrieval trigger occurs when:

    K(t_i) > tau

where tau is the retrieval threshold.

Important
---------
This module performs ONLY the trigger mathematics.

It does not:

    - generate tokens,
    - extract hidden states,
    - calculate raw confidence,
    - normalize confidence,
    - determine whether a token is a stop word,
    - calculate query complexity,
    - retrieve documents,
    - or restart generation.

Those responsibilities belong to other modules.

The purpose of keeping the trigger logic isolated here is to ensure that
offline experiments and live generation use exactly the same definition of:

    K(t_i)

and exactly the same threshold rule.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List


@dataclass(frozen=True)
class TokenActivation:
    """
    Retrieval-trigger information for one generated token.

    Attributes
    ----------
    token:
        Original tokenizer-level token.

    confidence:
        Normalized internal confidence:

            m_tilde_i

    content_mask:
        Binary token-content mask:

            s_i

    activation:
        INKER activation score:

            K(t_i)

    triggered:
        Whether:

            K(t_i) > threshold
    """

    token: str
    confidence: float
    content_mask: int
    activation: float
    triggered: bool

    def to_dict(self) -> dict:
        """
        Convert the activation record to a JSON-friendly dictionary.

        The output names intentionally use the mathematical notation used
        throughout the INKER replication.
        """

        return {
            "token":
                self.token,

            "m_tilde":
                float(
                    self.confidence
                ),

            "s_i":
                int(
                    self.content_mask
                ),

            "K":
                float(
                    self.activation
                ),

            "triggered":
                bool(
                    self.triggered
                ),
        }


def _validate_probability_like(
    name: str,
    value: float,
) -> float:
    """
    Validate a scalar that must lie in [0, 1].

    Used for both:

        E
        m_tilde_i
        threshold
    """

    try:
        value = float(
            value
        )

    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a numeric value."
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], "
            f"received {value}."
        )

    return value


def _validate_content_mask(
    content_mask: int,
) -> int:
    """
    Validate the binary token-content mask s_i.
    """

    # bool is acceptable because:
    #
    #     False -> 0
    #     True  -> 1
    #
    # but convert explicitly so returned data is consistent.

    if content_mask not in (
        0,
        1,
        False,
        True,
    ):
        raise ValueError(
            "content_mask (s_i) must be either 0 or 1."
        )

    return int(
        content_mask
    )


def compute_activation(
    E: float,
    m_tilde: float,
    content_mask: int,
) -> float:
    """
    Compute INKER activation for one generated token.

    Equation
    --------

        K(t_i) = (E - m_tilde_i) * s_i

    Parameters
    ----------
    E:
        External query-complexity score in [0, 1].

    m_tilde:
        Causally normalized internal confidence score in [0, 1].

    content_mask:
        Binary token mask:

            1 -> content token
            0 -> masked token

    Returns
    -------
    float
        Token activation K(t_i).

    Interpretation
    --------------
    For a content token:

        s_i = 1

    therefore:

        K(t_i) = E - m_tilde_i

    A complex query combined with low confidence therefore produces a larger
    activation.

    For a masked token:

        s_i = 0

    therefore:

        K(t_i) = 0

    regardless of query complexity or confidence.
    """

    E = _validate_probability_like(
        "E",
        E,
    )

    m_tilde = _validate_probability_like(
        "m_tilde",
        m_tilde,
    )

    content_mask = _validate_content_mask(
        content_mask
    )

    return float(
        (
            E
            - m_tilde
        )
        * content_mask
    )


def retrieval_is_triggered(
    activation: float,
    threshold: float = 0.5,
) -> bool:
    """
    Decide whether one activation triggers retrieval.

    Retrieval occurs when:

        K(t_i) > threshold

    The strict greater-than operator is intentional.

    Therefore:

        K(t_i) == threshold

    does NOT trigger retrieval.
    """

    threshold = _validate_probability_like(
        "threshold",
        threshold,
    )

    try:
        activation = float(
            activation
        )

    except (TypeError, ValueError) as exc:
        raise TypeError(
            "activation must be numeric."
        ) from exc

    # K can theoretically lie in:
    #
    #     [-1, 1]
    #
    # because:
    #
    #     E        in [0, 1]
    #     m_tilde  in [0, 1]
    #
    # so:
    #
    #     E - m_tilde in [-1, 1]
    #
    # We therefore do NOT incorrectly restrict activation itself to [0, 1].

    if not -1.0 <= activation <= 1.0:
        raise ValueError(
            "activation K(t_i) should lie in [-1, 1], "
            f"received {activation}."
        )

    return bool(
        activation
        > threshold
    )


def evaluate_token_activation(
    token: str,
    E: float,
    m_tilde: float,
    content_mask: int,
    threshold: float = 0.5,
) -> TokenActivation:
    """
    Compute both activation and retrieval decision for one token.

    This is the main helper that live generation should call.

    It performs:

        E
        +
        m_tilde_i
        +
        s_i
             ↓
          K(t_i)
             ↓
        K(t_i) > threshold
             ↓
        triggered
    """

    activation = compute_activation(
        E=E,
        m_tilde=m_tilde,
        content_mask=content_mask,
    )

    triggered = retrieval_is_triggered(
        activation=activation,
        threshold=threshold,
    )

    return TokenActivation(
        token=str(
            token
        ),
        confidence=float(
            m_tilde
        ),
        content_mask=int(
            content_mask
        ),
        activation=float(
            activation
        ),
        triggered=bool(
            triggered
        ),
    )


def evaluate_sequence_activation(
    tokens: Iterable[str],
    confidences: Iterable[float],
    content_masks: Iterable[int],
    E: float,
    threshold: float = 0.5,
) -> List[TokenActivation]:
    """
    Evaluate INKER activation over an already available token sequence.

    This function is useful for:

        - offline analysis,
        - saved token-confidence experiments,
        - debugging,
        - ablation studies,
        - and validating live-generation outputs.

    Parameters
    ----------
    tokens:
        Generated tokenizer-level tokens.

    confidences:
        Corresponding normalized confidence values:

            m_tilde_0,
            m_tilde_1,
            ...

    content_masks:
        Corresponding binary masks:

            s_0,
            s_1,
            ...

    E:
        External query-level complexity score.

    threshold:
        Retrieval threshold tau.

    Returns
    -------
    list[TokenActivation]
        One activation record for every token.

    Raises
    ------
    ValueError
        If the three token-level sequences do not have identical lengths.
    """

    tokens = list(
        tokens
    )

    confidences = list(
        confidences
    )

    content_masks = list(
        content_masks
    )

    if not (
        len(tokens)
        == len(confidences)
        == len(content_masks)
    ):
        raise ValueError(
            "tokens, confidences, and content_masks must have "
            "the same length. "
            f"Received: tokens={len(tokens)}, "
            f"confidences={len(confidences)}, "
            f"content_masks={len(content_masks)}."
        )

    # Validate query-level parameters once before looping.
    E = _validate_probability_like(
        "E",
        E,
    )

    threshold = _validate_probability_like(
        "threshold",
        threshold,
    )

    results: List[
        TokenActivation
    ] = []

    for (
        token,
        confidence,
        mask,
    ) in zip(
        tokens,
        confidences,
        content_masks,
    ):

        result = evaluate_token_activation(
            token=token,
            E=E,
            m_tilde=confidence,
            content_mask=mask,
            threshold=threshold,
        )

        results.append(
            result
        )

    return results


def sequence_triggers_retrieval(
    activations: Iterable[TokenActivation],
) -> bool:
    """
    Return True if at least one token triggers retrieval.

    This performs:

        any(
            K(t_i) > threshold
        )

    where the individual threshold decisions have already been stored in each
    ``TokenActivation`` object.
    """

    return any(
        activation.triggered
        for activation
        in activations
    )


def first_retrieval_trigger(
    activations: Iterable[TokenActivation],
) -> TokenActivation | None:
    """
    Return the first token that triggered retrieval.

    Returns
    -------
    TokenActivation
        First triggered token.

    None
        If no token triggered retrieval.

    This helper is especially useful when comparing an offline sequence
    evaluation against the live generation loop, because live INKER should
    stop at the first trigger.
    """

    for activation in activations:

        if activation.triggered:
            return activation

    return None
