from __future__ import annotations

"""Пресеты наборов вариантов связности и их разворачивание."""

from typing import List, Tuple


PRESETS = {
    # максимально безопасный по времени и интерпретации
    "basic": [
        "corr_full",
        "coh_full",
        "mutinf_full",
    ],
    # частотные/сложностные
    "spectral": [
        "coh_full",
        "fftcoh_full",
    ],
    "entropy": [
        "mutinf_full",
        "mutinf_partial",
    ],
    # направленные/каузальные
    "causal": [
        "granger_directed",
        "te_directed",
        "ah_directed",
    ],
    # «всё адекватное»: без слишком экспериментальных комбинаций
    "full": [
        "corr_full",
        "corr_partial",
        "coh_full",
        "mutinf_full",
        "mutinf_partial",
        "granger_directed",
        "te_directed",
        "ah_directed",
    ],
    "all": [
        "corr_full",
        "corr_partial",
        "coh_full",
        "mutinf_full",
        "mutinf_partial",
        "granger_directed",
        "te_full",
        "te_partial",
        "te_directed",
        "ah_full",
        "ah_partial",
        "ah_directed",
    ],
}


def expand_variants(tokens: List[str]) -> Tuple[List[str], str]:
    """Разворачивает смесь пресетов и явных variant-ов.

    Args:
        tokens: список токенов, где каждый элемент — либо имя пресета,
            либо конкретный variant.

    Returns:
        Кортеж `(variants, explain_text)`:
          - `variants`: уникальный список в исходном порядке.
          - `explain_text`: пояснение, какие пресеты развернулись во что.
    """
    out: List[str] = []
    explain: List[str] = []

    for token in tokens:
        key = str(token).strip().lower()
        if not key:
            continue
        if key in PRESETS:
            vs = PRESETS[key]
            explain.append(f"preset '{key}' -> {', '.join(vs)}")
            out.extend(vs)
        else:
            out.append(key)

    # unique with stable order
    seen = set()
    uniq: List[str] = []
    for variant in out:
        if variant not in seen:
            uniq.append(variant)
            seen.add(variant)

    if not explain:
        explain.append("variants задан списком напрямую")

    return uniq, "\n".join(explain)
