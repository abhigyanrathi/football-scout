import numpy as np
import pandas as pd

from fbrecruit.links.statsbomb_tm import match_pass2
from fbrecruit.links.teams import assign
from fbrecruit.links.text import norm, norm_club, ratio, tokens_contained


def test_norm_strips_accents_and_punctuation():
    assert norm("Sébastien Wüthrich") == "sebastien wuthrich"
    assert norm("Jérémie Porsan-Clemente") == "jeremie porsan clemente"


def test_norm_non_breaking_space_and_missing_values():
    assert norm("Rui\xa0Silva") == "rui silva"
    assert norm(None) == ""
    assert norm(float("nan")) == ""


def test_norm_special_letters():
    assert norm("Ole Selnæs") == "ole selnaes"
    assert norm("Martin Ødegaard") == "martin odegaard"
    assert norm("Łukasz Fabiański") == "lukasz fabianski"


def test_norm_club_drops_stop_words():
    assert norm_club("FC Barcelona") == "barcelona"
    assert norm_club("Udinese Calcio") == "udinese"
    assert norm_club("Real Betis Balompié SAD") == "real betis balompie"
    assert norm_club("AS Monaco") == "monaco"


def test_ratio_is_zero_for_empty_strings():
    assert ratio("", "abc") == 0.0
    assert ratio("abc", "abc") == 1.0


def test_assignment_is_one_to_one_where_best_matches_collide():
    sim = np.array([[0.9, 0.8, 0.1], [0.9, 0.1, 0.2], [0.3, 0.2, 0.7]])
    assert sim.argmax(axis=1).tolist() == [0, 0, 2]
    cols = assign(sim)
    assert cols.tolist() == [1, 0, 2]
    assert len(set(cols)) == 3


def test_tokens_contained_positive_and_negative():
    assert tokens_contained("Juanfran Torres", "Juan Francisco Torres Belén", "Juanfran")
    assert not tokens_contained("Robbie Brady", "Robert Brady", None)


def test_tokens_contained_minimum_tokens():
    assert tokens_contained("Neymar", "Neymar da Silva")
    assert not tokens_contained("Neymar", "Neymar da Silva", min_tokens=2)


def unmatched(name, nickname, tm_club_id=1):
    return pd.DataFrame(
        {
            "league": ["la_liga"],
            "tm_club_id": [tm_club_id],
            "player_name": [name],
            "nickname": [nickname],
            "status": ["unmatched"],
            "method": [""],
            "tm_player_id": pd.array([pd.NA], dtype="Int64"),
            "tm_name": [None],
            "link_pass": pd.array([pd.NA], dtype="Int64"),
        }
    )


def test_pass2_links_when_tokens_are_contained():
    cands = {("la_liga", 1): [(10, "Juanfran Torres", "juanfran torres"), (11, "Koke", "koke")]}
    out = match_pass2(unmatched("Juan Francisco Torres Belén", "Juanfran"), cands)
    row = out.iloc[0]
    assert (row.status, row.tm_player_id, row.link_pass) == ("matched", 10, 2)


def test_pass2_rejects_a_different_first_name():
    cands = {("la_liga", 1): [(10, "Robbie Brady", "robbie brady")]}
    out = match_pass2(unmatched("Robert Brady", None), cands)
    assert out.iloc[0].status == "unmatched"


def test_pass2_needs_exactly_one_candidate_and_two_tokens():
    two = {("la_liga", 1): [(10, "Juan Torres", "juan torres"), (11, "Francisco Torres", "x")]}
    out = match_pass2(unmatched("Juan Francisco Torres", None), two)
    assert out.iloc[0].status == "unmatched"
    one_word = {("la_liga", 1): [(12, "Torres", "torres")]}
    assert match_pass2(unmatched("Juan Torres", None), one_word).iloc[0].status == "unmatched"


def test_pass2_leaves_pass1_matches_alone():
    links = unmatched("Juan Francisco Torres Belén", "Juanfran")
    links.loc[0, ["status", "tm_player_id", "link_pass"]] = ["matched", 99, 1]
    cands = {("la_liga", 1): [(10, "Juanfran Torres", "juanfran torres")]}
    row = match_pass2(links, cands).iloc[0]
    assert (row.tm_player_id, row.link_pass) == (99, 1)
