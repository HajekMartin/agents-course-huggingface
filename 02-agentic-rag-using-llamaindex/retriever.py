"""Local guest records, exact lookup, and a persistent Ollama vector index."""

import hashlib
import json
import re
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "guests.json"


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_name(value: str) -> str:
    return re.sub(r"^(dr\.?|lady|sir)\s+", "", normalize(value))


def relation_category(relation: str) -> str:
    """Derive a coarse category using explicit rules; retain the original relation."""
    text = normalize(relation)
    if text in {"no relation", "none", "unknown", ""}:
        return "none"
    words = set(re.findall(r"\w+", text))
    if "friend" in words or "friends" in words:
        return "friend"
    if words & {"family", "sister", "brother", "mother", "father", "cousin", "spouse"}:
        return "family"
    if words & {"colleague", "coworker", "coauthor"}:
        return "colleague"
    return "other"


class GuestCatalog:
    """Validate a local snapshot and retain stable IDs for source attribution."""

    def __init__(self, path: Path = DATA_PATH):
        path = Path(path).resolve()
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict) or "guests" not in snapshot:
            raise ValueError("The guest snapshot must be an object containing a guests list.")
        rows = snapshot["guests"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("The guest snapshot must contain a nonempty guests list.")
        self.guests = []
        seen = set()
        try:
            source_file = path.relative_to(ROOT).as_posix()
        except ValueError:
            source_file = str(path)
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Guest {position} must be an object.")
            for field in ("name", "relation", "description", "email"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"Guest {position} needs a string field '{field}'.")
            name = normalize_name(row["name"])
            if not name or name in seen:
                raise ValueError(f"Empty or duplicate guest name: {row['name']}")
            seen.add(name)
            category = row.get("relation_category", relation_category(row["relation"]))
            if not isinstance(category, str) or category not in {"friend", "family", "colleague", "none", "other"}:
                raise ValueError(f"Invalid relation_category for {row['name']}: {category}")
            guest_id = f"guest-{position:04d}"
            self.guests.append({
                **{field: row[field] for field in ("name", "relation", "description", "email")},
                "id": guest_id,
                "relation_category": category,
                "source": f"{source_file}#{guest_id}",
            })
        self.by_id = {guest["id"]: guest for guest in self.guests}
        self.source_dataset = snapshot.get("dataset", "Local guest snapshot")


class GuestRetriever:
    """Rank guest profiles by embedding similarity; do not treat ranking as filtering."""

    def __init__(self, catalog: GuestCatalog, embed_model, cache_dir: Path = ROOT / "storage", embedding_revision: str = ""):
        self.catalog = catalog
        self.embed_model = embed_model
        configuration = {
            "guests": catalog.guests,
            "embedding": embed_model.model_name,
            "embedding_revision": embedding_revision,
            "query_prefix": embed_model.query_instruction,
            "text_prefix": embed_model.text_instruction,
            "schema_version": 1,
        }
        digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
        self.index_dir = Path(cache_dir) / digest
        self._index = None

    def get_index(self):
        if self._index is not None:
            return self._index
        if (self.index_dir / "docstore.json").exists():
            storage = StorageContext.from_defaults(persist_dir=str(self.index_dir))
            self._index = load_index_from_storage(storage, embed_model=self.embed_model)
        else:
            nodes = [
                TextNode(
                    id_=guest["id"],
                    text="\n".join(f"{field}: {guest[field]}" for field in ("name", "relation", "description")),
                    metadata={"guest_id": guest["id"]},
                    excluded_embed_metadata_keys=["guest_id"],
                )
                for guest in self.catalog.guests
            ]
            self._index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(persist_dir=str(self.index_dir))
        return self._index

    def search(self, query: str, top_k: int = 5, min_score: float = 0.6) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20.")
        if type(min_score) not in {int, float} or not -1 <= min_score <= 1:
            raise ValueError("min_score must be a number between -1 and 1.")
        retriever = self.get_index().as_retriever(similarity_top_k=min(top_k, len(self.catalog.guests)))
        hits = retriever.retrieve(query)
        return {
            "query": query,
            "retrieval": "semantic_similarity",
            "exhaustive": False,
            "min_score": min_score,
            "note": "Ranked candidates above the similarity cutoff, not verified matches. Scores are not probabilities. Use profile evidence and cite sources; use filter_guests for exact lists.",
            "candidates": [
                {**self.catalog.by_id[hit.node.metadata["guest_id"]], "similarity_score": hit.score}
                for hit in hits
                if hit.score is not None and hit.score >= min_score
            ],
        }
