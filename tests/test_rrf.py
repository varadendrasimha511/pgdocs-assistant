from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.types import Hit


def h(i: int) -> Hit:
    return Hit(id=i, content=f"c{i}", section="s", chapter="c", url="u", version="16")


def test_doc_in_both_lists_wins():
    fused = reciprocal_rank_fusion([[h(1), h(2), h(3)], [h(3), h(4), h(5)]])
    assert fused[0].id == 3  # appears in both lists


def test_scores_use_rank_not_raw_score():
    fused = reciprocal_rank_fusion([[h(1), h(2)], [h(2), h(1)]])
    assert {x.id for x in fused} == {1, 2}
    assert abs(fused[0].score - fused[1].score) < 1e-12  # symmetric ranks -> tie


def test_dedupes_and_handles_empty_list():
    fused = reciprocal_rank_fusion([[h(1), h(1)], []])
    assert [x.id for x in fused].count(1) == 1
    assert reciprocal_rank_fusion([[], []]) == []