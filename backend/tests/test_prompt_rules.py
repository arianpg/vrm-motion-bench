from pathlib import Path

from vmb_backend.llm import output_schema, system_prompt
from vmb_backend.manifest import DurationSpec, PromptRules, load_manifests
from vmb_backend.prompt_rules import check

RULES = PromptRules(language="en", starts_with="A person", max_words=10, forbidden_terms=["ball"])
DURATION = DurationSpec(min=1, max=10, step=0.1)


def test_valid_prompt():
    assert check("A person walks forward and waves.", 5.0, RULES, DURATION) == []


def test_violations():
    v = check("人が歩く", 12.0, RULES, DURATION)
    assert any("non-English" in x for x in v)
    assert any("begin with" in x for x in v)
    assert any("outside" in x for x in v)
    assert any("forbidden" in x for x in check("A person kicks a ball.", 3, RULES, DURATION))
    assert any("maximum is 10" in x for x in check("A person " + "walks " * 12, 3, RULES, DURATION))
    assert check("A person walks.", None, RULES, DURATION) == ["duration is missing."]


def test_typographic_quotes_allowed():
    assert check("A person’s arms swing.", 3, RULES, DURATION) == []


def test_kimodo_system_prompt_contains_guide_and_rules():
    manifests = load_manifests(Path(__file__).resolve().parents[2] / "adapters")
    m = manifests["kimodo"]
    text = system_prompt(m)
    assert 'begin with "A person"' in text
    assert "locomotion, gestures" in text
    schema = output_schema(m)
    assert schema["required"] == ["prompt", "duration", "notes"]
