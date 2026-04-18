"""Dataset shape checks (no MCP / API required)."""

from tests.helpers import load_golden_dataset


def test_golden_dataset_row_count_and_categories():
    rows = load_golden_dataset()
    assert 15 <= len(rows) <= 20
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for cat in ("happy_path", "edge_cases", "failure_cases"):
        assert counts.get(cat, 0) >= 3, f"need at least 3 {cat} examples"
