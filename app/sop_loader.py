import json
from pathlib import Path
from app.state import SOP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOP_FILE = PROJECT_ROOT / "sops.json"


def load_sops() -> list[SOP]:
    """
    Load all SOPs directly from the JSON policy file.
    """
    with SOP_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [SOP.model_validate(sop) for sop in data["sops"]]