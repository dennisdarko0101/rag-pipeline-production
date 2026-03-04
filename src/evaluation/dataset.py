"""Evaluation dataset management: load, save, and validate Q&A pairs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default golden dataset path (relative to project root)
_DEFAULT_DATASET_PATH = Path("tests/eval/eval_dataset.json")


@dataclass
class QAPair:
    """A question-answer pair for evaluation.

    Attributes:
        question: The evaluation question.
        ground_truth: The expected (reference) answer.
        contexts: Optional pre-defined context passages.
        category: Classification of the question (straightforward, multi_chunk,
            unanswerable, adversarial).
        metadata: Arbitrary extra data.
    """

    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    category: str = "straightforward"
    metadata: dict = field(default_factory=dict)


class EvalDataset:
    """Manages a collection of Q&A pairs for RAG evaluation.

    Supports loading from / saving to JSON and filtering by category.
    """

    def __init__(self, pairs: list[QAPair] | None = None) -> None:
        self._pairs: list[QAPair] = pairs or []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pairs(self) -> list[QAPair]:
        return list(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> QAPair:
        return self._pairs[idx]

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, pair: QAPair) -> None:
        self._pairs.append(pair)

    def add_many(self, pairs: list[QAPair]) -> None:
        self._pairs.extend(pairs)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_category(self, category: str) -> EvalDataset:
        """Return a new EvalDataset containing only pairs of the given category."""
        filtered = [p for p in self._pairs if p.category == category]
        return EvalDataset(pairs=filtered)

    @property
    def categories(self) -> list[str]:
        return sorted({p.category for p in self._pairs})

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": "1.0",
            "total_pairs": len(self._pairs),
            "categories": {
                cat: sum(1 for p in self._pairs if p.category == cat) for cat in self.categories
            },
            "pairs": [asdict(p) for p in self._pairs],
        }

    def save(self, path: str | Path) -> None:
        """Save dataset to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("dataset_saved", path=str(path), count=len(self._pairs))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> EvalDataset:
        """Create an EvalDataset from a dictionary (e.g. parsed JSON)."""
        pairs: list[QAPair] = []
        for item in data.get("pairs", []):
            pairs.append(
                QAPair(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    contexts=item.get("contexts", []),
                    category=item.get("category", "straightforward"),
                    metadata=item.get("metadata", {}),
                )
            )
        return cls(pairs=pairs)

    @classmethod
    def load(cls, path: str | Path | None = None) -> EvalDataset:
        """Load a dataset from a JSON file.

        Args:
            path: Path to JSON file.  Defaults to the built-in golden dataset.

        Returns:
            Populated EvalDataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(path) if path else _DEFAULT_DATASET_PATH
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = cls.from_dict(data)
        logger.info("dataset_loaded", path=str(path), count=len(dataset))
        return dataset
