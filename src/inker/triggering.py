# src/inker/trigger.py

"""
INKER retrieval-trigger computation.

Given:

    E          = external query-complexity score
    m_tilde_i  = normalized internal confidence for token i
    s_i        = token-content mask

the INKER activation score is:

    K(t_i) = (E - m_tilde_i) * s_i

Retrieval is triggered when:

    K(t_i) > threshold

for a content token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TokenActivation:
    """
    Activation information for one generated token.
    """

    token: str
    confidence: float
    content_mask: int
    activation: float
    triggered: bool

    def to_dict(self):
        return {
            "token": self.token,
            "m_tilde": self.confidence,
            "s_i": self.content_mask,
            "K": self.activation,
            "triggered": self.triggered,
        }


def compute_activation(
    E: float,
    m_tilde: float,
    content_mask: int,
) -> float:
    """
    Compute the INKER activation score for a single token.

    K(t_i) = (E - m_tilde_i) * s_i
    """

    if not 0.0 <= E <= 1.0:
        raise ValueError(
            f"E must lie in [0, 1], received {E}."
        )

    if not 0.0 <= m_tilde <= 1.0:
        raise ValueError(
            f"m_tilde must lie in [0, 1], received {m_tilde}."
        )

    if content_mask not in (0, 1):
        raise ValueError(
            "content_mask must be either 0 or 1."
        )

    return (E - m_tilde) * content_mask


def evaluate_token_activation(
    token: str,
    E: float,
    m_tilde: float,
    content_mask: int,
    threshold: float = 0.5,
) -> TokenActivation:
    """
    Calculate activation and retrieval decision for one token.
    """

    activation = compute_activation(
        E=E,
        m_tilde=m_tilde,
        content_mask=content_mask,
    )

    triggered = activation > threshold

    return TokenActivation(
        token=token,
        confidence=m_tilde,
        content_mask=content_mask,
        activation=activation,
        triggered=triggered,
    )


def evaluate_sequence_activation(
    tokens: Iterable[str],
    confidences: Iterable[float],
    content_masks: Iterable[int],
    E: float,
    threshold: float = 0.5,
) -> List[TokenActivation]:
    """
    Compute INKER activation scores for an entire generated response.

    Parameters
    ----------
    tokens:
        Generated tokens.

    confidences:
        Normalized token confidence scores m_tilde_i.

    content_masks:
        Binary content masks s_i.

    E:
        Query-level external complexity score.

    threshold:
        Retrieval threshold tau.

    Returns
    -------
    List[TokenActivation]
    """

    tokens = list(tokens)
    confidences = list(confidences)
    content_masks = list(content_masks)

    if not (
        len(tokens)
        == len(confidences)
        == len(content_masks)
    ):
        raise ValueError(
            "tokens, confidences, and content_masks "
            "must have the same length."
        )

    results = []

    for token, confidence, mask in zip(
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

        results.append(result)

    return results


def sequence_triggers_retrieval(
    activations: Iterable[TokenActivation],
) -> bool:
    """
    Return True if at least one generated token triggers retrieval.
    """

    return any(
        item.triggered
        for item in activations
    )
