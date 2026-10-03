"""Static check of the workflow that publishes the scouting site."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/configure-pages": "45bfe0192ca1faeb007ade9deae92b16b8254a0d",
    "actions/upload-pages-artifact": "fc324d3547104276b827a68afc52ff2a11cc49c9",
    "actions/deploy-pages": "368f82528645a54fb793d4d04e342629a3f51346",
}


def test_the_workflow_pins_each_action_to_its_commit():
    text = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    uses = re.findall(r"^\s*(?:-\s+)?uses:\s*([^@\s]+)@(\S+)", text, flags=re.MULTILINE)
    assert len(uses) == len(PINS) and dict(uses) == PINS
