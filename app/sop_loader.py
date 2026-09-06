import hashlib
import json
from pathlib import Path

from app.state import SOP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOP_FILE = PROJECT_ROOT / "sops.json"


def load_sops() -> list[SOP]:
    """
    Load all SOPs from the external JSON policy file.

    Returns:
        list[SOP]: Independently maintained policy rules.
    """

    with SOP_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [SOP.model_validate(sop) for sop in data["sops"]]


def get_sop_file_hash() -> str:
    """
    Return a SHA-256 hash of the canonical SOP JSON.

    The hash is used to determine whether the derived FAISS
    index needs to be rebuilt.
    """

    with SOP_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    canonical_json = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def main() -> None:
    """Run a small manual SOP loading check."""

    sops = load_sops()

    print(f"Loaded {len(sops)} SOPs.")

    for sop in sops:
        print(
            f"{sop.id} | "
            f"{sop.category} | "
            f"{sop.severity} | "
            f"{sop.condition_type}"
        )

    print(f"SOP hash: {get_sop_file_hash()}")


if __name__ == "__main__":
    main()