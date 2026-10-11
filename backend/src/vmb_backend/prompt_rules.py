"""マニフェストの prompt.rules による LLM 出力の検証と、システムプロンプトへの埋め込み。"""

from __future__ import annotations

import re

from .manifest import DurationSpec, PromptRules

# 英語のプロンプトに許す文字（ASCII と、LLM が出しがちな記号）
_ALLOWED_NON_ASCII = set("’‘“”—–…")


def render_rules(rules: PromptRules, duration: DurationSpec | None) -> list[str]:
    """機械的に検証するルールを、LLM に渡す箇条書きにする（検証と同じ内容）。"""
    lines: list[str] = []
    if rules.language == "en":
        lines.append("The prompt must be written in English only (ASCII characters).")
    if rules.starts_with:
        lines.append(f'The prompt must begin with "{rules.starts_with}".')
    if rules.max_words:
        lines.append(f"The prompt must be at most {rules.max_words} words.")
    if rules.forbidden_terms:
        lines.append("Do not use these terms: " + ", ".join(rules.forbidden_terms) + ".")
    if duration:
        lines.append(
            f"duration is in seconds, between {duration.min:g} and {duration.max:g} "
            f"(in steps of {duration.step:g})."
        )
    return lines


def check(prompt: str, duration_value: float | None, rules: PromptRules, duration: DurationSpec | None) -> list[str]:
    """違反の一覧（英語。LLM への再生成の指示にもそのまま使う）。空なら合格。"""
    violations: list[str] = []
    text = prompt.strip()
    if not text:
        return ["The prompt is empty."]
    if rules.language == "en":
        bad = sorted({c for c in text if ord(c) > 127 and c not in _ALLOWED_NON_ASCII})
        if bad:
            violations.append(f"The prompt contains non-English characters: {''.join(bad[:10])}")
    if rules.starts_with and not text.lower().startswith(rules.starts_with.lower()):
        violations.append(f'The prompt does not begin with "{rules.starts_with}".')
    if rules.max_words:
        words = len(text.split())
        if words > rules.max_words:
            violations.append(f"The prompt has {words} words; the maximum is {rules.max_words}.")
    for term in rules.forbidden_terms:
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            violations.append(f'The prompt uses the forbidden term "{term}".')
    if duration is not None:
        if duration_value is None:
            violations.append("duration is missing.")
        elif not duration.min <= duration_value <= duration.max:
            violations.append(f"duration {duration_value:g} is outside {duration.min:g}-{duration.max:g} seconds.")
    return violations
