import pandas as pd

from fbrecruit.links.statsbomb_tm import agreement_summary, player_table, shared_id_cases


def links():
    rows = [
        # player 1 at two clubs, one Transfermarkt id
        (1, "la_liga", 10, "matched", 100, 900),
        (1, "la_liga", 11, "matched", 100, 300),
        # player 2: one matched and one unmatched row
        (2, "la_liga", 10, "matched", 200, 500),
        (2, "serie_a", 20, "unmatched", pd.NA, 40),
        # players 3 and 4 share one id at one club
        (3, "serie_a", 21, "matched", 300, 800),
        (4, "serie_a", 21, "matched", 300, 100),
        # player 5 never matched
        (5, "serie_a", 22, "unmatched", pd.NA, 60),
    ]
    df = pd.DataFrame(
        rows, columns=["player_id", "league", "team_id", "status", "tm_player_id", "minutes"]
    )
    return df.astype({"tm_player_id": "Int64"})


def test_player_table_has_one_row_per_player():
    out = player_table(links()).set_index("player_id")
    assert out.index.tolist() == [1, 2, 3, 4, 5]
    assert out.loc[1, "tm_player_id"] == 100
    assert (out.loc[1, "minutes"], out.loc[1, "n_matched"], out.loc[1, "n_unmatched"]) == (
        1200,
        2,
        0,
    )
    assert (out.loc[2, "n_matched"], out.loc[2, "n_unmatched"], out.loc[2, "minutes"]) == (
        1,
        1,
        540,
    )
    assert pd.isna(out.loc[5, "tm_player_id"]) and out.loc[5, "n_tm_ids"] == 0


def test_player_with_two_ids_gets_no_id_and_is_counted():
    df = links()
    df.loc[1, "tm_player_id"] = 101
    out = player_table(df).set_index("player_id")
    assert pd.isna(out.loc[1, "tm_player_id"]) and out.loc[1, "n_tm_ids"] == 2


def test_shared_ids_are_split_by_player_and_club():
    cases = shared_id_cases(links())
    assert [t for t, _ in cases["same_player_two_clubs"]] == [100]
    assert [t for t, _ in cases["different_players_same_club"]] == [300]
    assert cases["different_players_different_clubs"] == []


def test_agreement_summary_keeps_only_450_plus_pairs():
    agree = pd.DataFrame(
        {
            "link_pass": [1, 1, 1, 2],
            "minutes": [1000, 2000, 100, 500],
            "ratio": [1.0, 1.5, 3.0, 0.9],
        }
    )
    out = agreement_summary(agree, "link_pass").set_index("link_pass")
    assert out.loc["ALL", "pairs"] == 3
    assert out.loc[1, "pairs"] == 2 and out.loc[1, "share_within"] == 0.5
    assert out.loc[2, "median_ratio"] == 0.9
