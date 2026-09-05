MULTIHOP_CUES = {
    "and",
    "who",
    "which",
    "before",
    "after",
    "same",
    "both",
    "compare",
    "than",
}


def estimate_complexity_proxy(question):
    """
    Rule-based E proxy from the original notebook.

    This is a stand-in for INKER's trained T5-large Eva evaluator and is NOT
    an exact reproduction of that component.
    """
    words = question.lower().split()
    n_words = len(words)

    cue_hits = sum(
        1
        for w in words
        if w.strip("?,.") in MULTIHOP_CUES
    )

    length_component = min(
        n_words / 25.0,
        1.0,
    )

    cue_component = min(
        cue_hits / 3.0,
        1.0,
    )

    E = (
        0.5 * length_component
        + 0.5 * cue_component
    )

    return round(
        min(max(E, 0.0), 1.0),
        4,
    )
