import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.sop_loader import get_sop_file_hash, load_sops
from app.state import SOP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

INDEX_FILE = ARTIFACTS_DIR / "sop_index.faiss"
METADATA_FILE = ARTIFACTS_DIR / "sop_metadata.json"
HASH_FILE = ARTIFACTS_DIR / "sop_hash.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# This is a semantic similarity threshold, not a probability
# or model confidence score.
SIMILARITY_THRESHOLD = 0.35


class SOPRetriever:
    """
    Semantic SOP retriever backed by FAISS.

    FAISS is responsible only for finding policies that are
    semantically relevant to the user's request.

    It does NOT decide whether an SOP actually applies.
    """

    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.sops = load_sops()

        self.index: faiss.Index | None = None
        self._build_or_load_index()

    def _current_hash_matches(self) -> bool:
        """Check whether the stored index was built from the current SOP file."""

        if not HASH_FILE.exists():
            return False

        stored_hash = HASH_FILE.read_text(
            encoding="utf-8"
        ).strip()

        return stored_hash == get_sop_file_hash()

    def _build_index(self) -> None:
        """Build a new FAISS index from the current SOP descriptions."""

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        texts = [sop.description for sop in self.sops]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(
            index,
            str(INDEX_FILE),
        )

        metadata = [
            {
                "id": sop.id,
                "description": sop.description,
            }
            for sop in self.sops
        ]

        METADATA_FILE.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        HASH_FILE.write_text(
            get_sop_file_hash(),
            encoding="utf-8",
        )

        self.index = index

    def _load_index(self) -> None:
        """Load the existing FAISS index."""

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

    def _build_or_load_index(self) -> None:
        """
        Load the existing index when the SOP file is unchanged.

        Rebuild automatically when policies have been added or changed.
        """

        artifacts_exist = (
            INDEX_FILE.exists()
            and METADATA_FILE.exists()
            and HASH_FILE.exists()
        )

        if artifacts_exist and self._current_hash_matches():
            self._load_index()
        else:
            self._build_index()

    def retrieve(
        self,
        query: str,
    ) -> list[SOP]:
        """
        Retrieve SOPs semantically relevant to the user's query.

        No top-k cutoff is used for policy selection.
        All SOPs are searched and only those above the
        similarity threshold are returned.

        Returns:
            list[SOP]: Candidate policies ordered by similarity.
        """

        if self.index is None:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            len(self.sops),
        )

        candidates = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            if float(score) < SIMILARITY_THRESHOLD:
                continue

            candidates.append(
                (
                    float(score),
                    self.sops[index_position],
                )
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            sop
            for _, sop in candidates
        ]


def main() -> None:
    """Run a few manual semantic retrieval checks."""

    retriever = SOPRetriever()

    queries = [
        "Is it safe to go cycling today?",
        "Should I take my child to the park in this heat?",
        "Would today be suitable for having a picnic?",
        "Can I go fishing today?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        results = retriever.retrieve(query)

        if not results:
            print("No candidate SOPs found.")
            continue

        for sop in results:
            print(
                f"{sop.id} | "
                f"{sop.category} | "
                f"{sop.severity}"
            )


if __name__ == "__main__":
    main()