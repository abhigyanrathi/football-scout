"""Static check of the actions each workflow pins."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PINS = {
    "pages.yml": {
        "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/configure-pages": "45bfe0192ca1faeb007ade9deae92b16b8254a0d",
        "actions/upload-pages-artifact": "fc324d3547104276b827a68afc52ff2a11cc49c9",
        "actions/deploy-pages": "368f82528645a54fb793d4d04e342629a3f51346",
    },
    "ci.yml": {
        "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/setup-node": "820762786026740c76f36085b0efc47a31fe5020",
    },
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_each_workflow_pins_its_actions_to_their_commits(name):
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    uses = re.findall(r"^\s*(?:-\s+)?uses:\s*([^@\s]+)@(\S+)", text, flags=re.MULTILINE)
    assert len(uses) == len(PINS[name]) and dict(uses) == PINS[name]


def test_every_workflow_has_its_pins_listed():
    names = {p.name for p in WORKFLOWS.iterdir() if p.suffix in {".yml", ".yaml"}}
    assert names == set(PINS)
