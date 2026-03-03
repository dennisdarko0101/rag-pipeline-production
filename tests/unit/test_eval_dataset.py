"""Unit tests for the evaluation dataset module."""

import tempfile
from pathlib import Path

from src.evaluation.dataset import EvalDataset, QAPair


class TestQAPair:
    def test_defaults(self) -> None:
        pair = QAPair(question="Q?", ground_truth="A.")
        assert pair.category == "straightforward"
        assert pair.contexts == []
        assert pair.metadata == {}

    def test_full_construction(self) -> None:
        pair = QAPair(
            question="Q?",
            ground_truth="A.",
            contexts=["c1"],
            category="adversarial",
            metadata={"source": "test"},
        )
        assert pair.category == "adversarial"
        assert pair.contexts == ["c1"]


class TestEvalDataset:
    def test_empty_dataset(self) -> None:
        ds = EvalDataset()
        assert len(ds) == 0
        assert ds.pairs == []

    def test_add_and_len(self) -> None:
        ds = EvalDataset()
        ds.add(QAPair("Q?", "A."))
        assert len(ds) == 1

    def test_add_many(self) -> None:
        ds = EvalDataset()
        ds.add_many([QAPair("Q1?", "A1."), QAPair("Q2?", "A2.")])
        assert len(ds) == 2

    def test_getitem(self) -> None:
        ds = EvalDataset(pairs=[QAPair("Q?", "A.")])
        assert ds[0].question == "Q?"

    def test_filter_by_category(self) -> None:
        ds = EvalDataset(
            pairs=[
                QAPair("Q1?", "A1.", category="easy"),
                QAPair("Q2?", "A2.", category="hard"),
                QAPair("Q3?", "A3.", category="easy"),
            ]
        )
        easy = ds.filter_by_category("easy")
        assert len(easy) == 2
        hard = ds.filter_by_category("hard")
        assert len(hard) == 1

    def test_categories(self) -> None:
        ds = EvalDataset(
            pairs=[
                QAPair("Q1?", "A1.", category="b"),
                QAPair("Q2?", "A2.", category="a"),
                QAPair("Q3?", "A3.", category="b"),
            ]
        )
        assert ds.categories == ["a", "b"]

    def test_to_dict(self) -> None:
        ds = EvalDataset(
            pairs=[
                QAPair("Q1?", "A1.", category="straightforward"),
                QAPair("Q2?", "A2.", category="adversarial"),
            ]
        )
        d = ds.to_dict()
        assert d["version"] == "1.0"
        assert d["total_pairs"] == 2
        assert d["categories"]["straightforward"] == 1
        assert d["categories"]["adversarial"] == 1
        assert len(d["pairs"]) == 2

    def test_save_and_load_roundtrip(self) -> None:
        ds = EvalDataset(
            pairs=[
                QAPair("Q?", "A.", contexts=["c"], category="multi_chunk", metadata={"k": "v"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            ds.save(path)

            loaded = EvalDataset.load(path)
            assert len(loaded) == 1
            assert loaded[0].question == "Q?"
            assert loaded[0].ground_truth == "A."
            assert loaded[0].contexts == ["c"]
            assert loaded[0].category == "multi_chunk"
            assert loaded[0].metadata == {"k": "v"}

    def test_from_dict(self) -> None:
        data = {
            "pairs": [
                {"question": "Q?", "ground_truth": "A."},
                {"question": "Q2?", "ground_truth": "A2.", "category": "hard"},
            ]
        }
        ds = EvalDataset.from_dict(data)
        assert len(ds) == 2
        assert ds[0].category == "straightforward"  # default
        assert ds[1].category == "hard"

    def test_load_missing_file_raises(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            EvalDataset.load("/nonexistent/path/dataset.json")

    def test_save_creates_parent_dirs(self) -> None:
        ds = EvalDataset(pairs=[QAPair("Q?", "A.")])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "dataset.json"
            ds.save(path)
            assert path.exists()

    def test_load_golden_dataset(self) -> None:
        """Verify the built-in golden dataset loads correctly."""
        golden_path = Path("tests/eval/eval_dataset.json")
        if golden_path.exists():
            ds = EvalDataset.load(golden_path)
            assert len(ds) >= 15
            assert "straightforward" in ds.categories
            assert "adversarial" in ds.categories
            assert "unanswerable" in ds.categories
            assert "multi_chunk" in ds.categories

    def test_pairs_returns_copy(self) -> None:
        """Modifying .pairs should not affect the internal list."""
        ds = EvalDataset(pairs=[QAPair("Q?", "A.")])
        pairs = ds.pairs
        pairs.append(QAPair("Extra?", "Extra."))
        assert len(ds) == 1  # unchanged
