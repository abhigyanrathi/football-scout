import pandas as pd

from fbrecruit.links.text import norm
from fbrecruit.links.wyscout_tm import link_pass1, link_pass2


def pool(*names):
    return [(i + 1, n, norm(n)) for i, n in enumerate(names)]


def wy(first, last, short, dob="1990-01-01", wy_id=1):
    return pd.DataFrame(
        {
            "wyId": [wy_id],
            "firstName": [first],
            "lastName": [last],
            "shortName": [short],
            "dob": [dob],
        }
    )


def test_single_candidate_at_threshold_links():
    out = link_pass1(wy("Marc", "Bartra", "M. Bartra"), {"1990-01-01": pool("Marc Bartra")})
    row = out.iloc[0]
    assert (row.status, row.tm_player_id, row.link_pass) == ("linked", 1, 1)


def test_two_candidates_at_threshold_are_ambiguous():
    same_day = {"1990-01-01": pool("Juan Perez", "Juan Peres")}
    row = link_pass1(wy("Juan", "Perez", "J. Perez"), same_day).iloc[0]
    assert row.status == "ambiguous"
    assert row.n_at_threshold == 2
    assert pd.isna(row.tm_player_id)


def test_other_statuses():
    by_dob = {"1990-01-01": pool("Completely Different")}
    assert link_pass1(wy("Juan", "Perez", "J. Perez"), by_dob).iloc[0].status == "below_threshold"
    no_day = wy("Juan", "Perez", "J. Perez", dob="1991-02-02")
    assert link_pass1(no_day, by_dob).iloc[0].status == "no_candidate_same_dob"


def test_pass2_resolves_an_ambiguous_player_by_tokens():
    same_day = {"1990-01-01": pool("Juan Perez", "Juan Peres")}
    first = link_pass1(wy("Juan", "Perez", "J. Perez"), same_day)
    row = link_pass2(first, same_day).iloc[0]
    assert (row.status, row.tm_player_id, row.link_pass) == ("linked", 1, 2)


def test_pass2_stays_ambiguous_when_two_candidates_qualify():
    same_day = {"1990-01-01": pool("Juan Perez", "Juan Perez")}
    first = link_pass1(wy("Juan", "Perez", "J. Perez"), same_day)
    assert first.iloc[0].status == "ambiguous"
    assert link_pass2(first, same_day).iloc[0].status == "ambiguous"


def test_pass2_uses_short_name_tokens_and_leaves_pass1_alone():
    by_dob = {"1990-01-01": pool("Nacho Fernandez")}
    unlinked = link_pass1(wy("José Ignacio", "Fernández Iglesias", "Nacho"), by_dob)
    assert unlinked.iloc[0].status == "below_threshold"
    assert link_pass2(unlinked, by_dob).iloc[0].tm_player_id == 1
    linked = link_pass1(wy("Marc", "Bartra", "M. Bartra"), {"1990-01-01": pool("Marc Bartra")})
    assert link_pass2(linked, {"1990-01-01": pool("Marc Bartra")}).iloc[0].link_pass == 1
