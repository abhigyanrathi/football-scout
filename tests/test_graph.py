import numpy as np
import pandas as pd
import pytest

from fbrecruit import graph, style

NODES = 2073
KINDS = {"both": 4243, "pass": 3680, "press": 1768}
LINKS = 9691
SUMS = {"lolo": -110.78309810208157, "pooled": -336.4133332394995}
HIDDEN = 969
AREAS = {"with links": 0.6762783544790465, "without links": 0.7581252043482103}
REGULARS = 1401
SHARES = {
    1: 0.19664525339043545,
    2: 0.1809421841541756,
    3: 0.2190578158458244,
    4: 0.20835117773019274,
}


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def inputs():
    need(style.player_path())
    need(graph.links_path())
    return graph.graph_inputs()


@pytest.mark.slow
def test_links_recomputed_from_the_caches():
    for lg in graph.LEAGUES:
        need(style.OUT / lg / "actions.parquet")
        need(style.OUT / lg / "pressures.parquet")
    nodes = pd.read_parquet(need(style.player_path()))
    assert len(nodes) == NODES
    _, _, dropped, links = graph.build_links(nodes)
    assert links.kind.value_counts().to_dict() == KINDS
    assert len(links) == LINKS
    assert all(d.empty for d in dropped.values())
    pd.testing.assert_frame_equal(links, pd.read_parquet(need(graph.links_path())))


@pytest.mark.slow
def test_pooled_embeddings_recomputed():
    nodes, _, x, linked = inputs()
    need(graph.embeddings_path())
    z, _ = graph.run(nodes, x, linked, graph.SEED, 0)
    assert np.array_equal(z, graph.saved_pooled(nodes))


@pytest.mark.slow
def test_saved_embedding_sums():
    e = pd.read_parquet(need(graph.embeddings_path()))
    assert e.groupby("branch").size().to_dict() == {"lolo": NODES, "pooled": NODES}
    sums = {b: float(g[graph.E].to_numpy(np.float64).sum()) for b, g in e.groupby("branch")}
    assert sums == SUMS


@pytest.mark.slow
def test_hidden_link_areas():
    nodes, _, x, linked = inputs()
    assert graph.hidden_links(nodes, x, linked, 0) == (HIDDEN, AREAS)


@pytest.mark.slow
def test_seed_neighbour_shares():
    nodes, _, x, linked = inputs()
    need(graph.embeddings_path())
    assert graph.neighbour_shares(nodes, x, linked, 0) == (REGULARS, SHARES)


def counts(rows, n):
    return pd.DataFrame(rows, columns=["player_a", "player_b", n]).assign(league="x", team_id=1)[
        [*graph.PAIR, n]
    ]


def pairs(frame):
    return list(zip(frame.player_a, frame.player_b, strict=True))


def test_the_receiver_is_the_next_actions_player_on_the_same_team():
    a = pd.DataFrame(
        {
            "league": ["x"] * 7,
            "game_id": [1, 1, 1, 1, 1, 2, 2],
            "period_id": [1, 1, 1, 1, 2, 2, 2],
            "team_id": [1, 1, 1, 2, 2, 2, 2],
            "player_id": [10, 11, 11, 20, 21, 22, 23],
            "type_name": ["pass", "cross", "pass", "pass", "pass", "pass", "pass"],
            "result_id": [graph.SUCCESS] * 7,
        }
    )
    # teammate; same player; other team; other period; other game; teammate; last action
    got = graph.receivers(a)
    assert got.tolist() == [11, pd.NA, pd.NA, pd.NA, pd.NA, 23, pd.NA]

    # 11 passes back to 10, 10 to 12, and 12's failed pass is left out but receives
    a = pd.concat([a, a.iloc[[0, 1]].assign(game_id=3, player_id=[11, 10])], ignore_index=True)
    a.loc[len(a)] = ["x", 3, 1, 1, 12, "pass", 0]
    passes = graph.completed_passes(a)
    assert len(passes) == 9
    assert graph.pass_counts(passes).to_dict("records") == [
        {"league": "x", "team_id": 1, "player_a": 10, "player_b": 11, "n_pass": 2},
        {"league": "x", "team_id": 1, "player_a": 10, "player_b": 12, "n_pass": 1},
        {"league": "x", "team_id": 2, "player_a": 22, "player_b": 23, "n_pass": 1},
    ]


def test_pressing_together_is_at_most_five_seconds_apart():
    rows = [
        # game, period, team, player, seconds
        (1, 1, 1, 10, 10.0), (1, 1, 1, 11, 15.0),  # exactly 5 s
        (2, 1, 1, 10, 10.0), (2, 1, 1, 11, 15.001),  # just over
        (3, 1, 1, 10, 10.0), (3, 1, 2, 20, 11.0),  # other team
        (4, 1, 1, 10, 10.0), (4, 1, 1, 10, 11.0),  # same player
        (5, 1, 1, 10, 10.0), (5, 2, 1, 11, 11.0),  # other period
        (6, 1, 1, 11, 102.0), (6, 1, 1, 10, 100.0), (6, 1, 1, 11, 104.0), (6, 1, 1, 12, 110.0),
    ]  # fmt: skip
    press = pd.DataFrame(rows, columns=["game_id", "period_id", "team_id", "player_id", "seconds"])
    got = graph.press_counts(press.assign(league="x"))
    assert got.to_dict("records") == [
        {"league": "x", "team_id": 1, "player_a": 10, "player_b": 11, "n_press": 3}
    ]


def test_each_node_keeps_its_top_five_partners_with_at_least_three():
    c = counts(
        [
            (1, 2, 10), (1, 3, 9), (1, 4, 8), (1, 5, 7), (1, 6, 6), (1, 8, 6), (1, 16, 3),
            (8, 9, 20), (8, 10, 20), (8, 11, 20), (8, 12, 20), (8, 13, 20),
            (2, 14, 2), (2, 15, 3),
        ],
        "n",
    )  # fmt: skip
    # 1 keeps 6 over 8 on the tie and 8 keeps five others; 16 keeps 1; 14 is under the floor
    assert pairs(graph.select(c, "n")) == [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 16), (2, 15),
        (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
    ]  # fmt: skip


def test_the_two_kinds_merge_into_one_link_per_pair():
    passes = counts([(1, 2, 5), (1, 3, 4)], "n_pass")
    presses = counts([(1, 2, 3), (1, 3, 1), (2, 3, 7)], "n_press")
    got = graph.merge_kinds(passes, presses)
    assert got[["player_a", "player_b", "kind", "n_pass", "n_press"]].to_dict("records") == [
        {"player_a": 1, "player_b": 2, "kind": "both", "n_pass": 5, "n_press": 3},
        {"player_a": 1, "player_b": 3, "kind": "pass", "n_pass": 4, "n_press": 1},
        {"player_a": 2, "player_b": 3, "kind": "press", "n_pass": 0, "n_press": 7},
    ]


def small_graph():
    nodes = pd.DataFrame(
        {
            "league": ["x"] * 8 + ["y"] * 2,
            "team_id": [1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
            "player_id": range(10),
        }
    )
    linked = np.array([[0, 1], [1, 2], [4, 5], [5, 6], [6, 7]])
    x = np.random.default_rng(0).normal(size=(len(nodes), 5)).astype(np.float32)
    return nodes, x, linked


def test_training_is_reproducible_by_seed():
    nodes, x, linked = small_graph()
    z, _ = graph.run(nodes, x, linked, 0, 0)
    again, _ = graph.run(nodes, x, linked, 0, 0)
    other, _ = graph.run(nodes, x, linked, 1, 0)
    assert np.array_equal(z, again)
    assert not np.array_equal(z, other)


def test_negatives_are_unlinked_pairs_of_different_nodes_on_one_team():
    nodes, _, linked = small_graph()
    pool = graph.unlinked_pairs(nodes, linked)
    assert pool.tolist() == [[0, 2], [0, 3], [1, 3], [2, 3], [4, 6], [4, 7], [5, 7], [8, 9]]
    drawn = graph.negatives(pool, 1000, np.random.default_rng(0))
    team = list(zip(nodes.league, nodes.team_id, strict=True))
    assert all(team[a] == team[b] and a != b for a, b in drawn)
    assert not {tuple(p) for p in drawn} & {tuple(p) for p in linked}
    assert len({tuple(p) for p in drawn}) == len(pool)
