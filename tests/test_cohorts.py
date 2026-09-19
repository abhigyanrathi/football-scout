import pandas as pd

from fbrecruit.cohorts import add_cohort_rules, levels, mirror_labels

NAN = float("nan")


def moves(rows):
    df = pd.DataFrame(rows, columns=["player_id", "date", "from_club_id", "to_club_id", "fee"])
    return df.assign(date=pd.to_datetime(df.date)).rename(columns={"fee": "transfer_fee"})


def labelled():
    # the mover is the first row of each player; later rows are the mirror candidates
    rows = [
        (1, "2016-07-01", 1, 2, 5e6),  # paid, no mirror
        (2, "2016-07-01", 1, 2, 5e6),  # paid_mirror
        (2, "2018-07-01", 2, 1, 3e6),
        (3, "2016-06-30", 1, 2, 0),  # loan_return
        (3, "2015-07-01", 2, 1, 0),
        (4, "2016-07-01", 1, 2, 0),  # loan_out, mirror with a null fee
        (4, "2017-06-30", 2, 1, NAN),
        (5, "2016-07-01", 1, 2, NAN),  # both
        (5, "2015-07-01", 2, 1, 0),
        (5, "2017-06-30", 2, 1, 0),
        (6, "2016-07-01", 1, 2, 0),  # free_other, no mirror
        (7, "2016-07-01", 1, 2, 0),  # free_other, same-day mirror
        (7, "2016-07-01", 2, 1, 0),
        (8, "2016-07-01", 1, 2, 0),  # free_other, the mirror was paid
        (8, "2017-07-01", 2, 1, 2e6),
        (9, "2016-07-01", 1, 2, 0),  # free_other, mirror more than three years away
        (9, "2020-07-01", 2, 1, 0),
        (10, "2016-07-01", 1, 2, NAN),  # free_other, null fee
    ]
    tr = moves(rows)
    mover = tr.groupby("player_id").head(1).reset_index(drop=True)
    return mirror_labels(mover, tr).set_index("player_id")


def test_mirror_label_covers_all_six_labels():
    out = labelled().mirror_label
    assert out.to_dict() == {
        1: "paid",
        2: "paid_mirror",
        3: "loan_return",
        4: "loan_out",
        5: "both",
        6: "free_other",
        7: "free_other",
        8: "free_other",
        9: "free_other",
        10: "free_other",
    }


def test_same_day_mirror_counts_as_neither_earlier_nor_later():
    row = labelled().loc[7]
    assert (row.has_mirror, row.mirror_same_day) == (True, True)
    assert not row.mirror_zero_earlier and not row.mirror_zero_later


def test_mirror_beyond_three_years_is_ignored():
    assert labelled().loc[9].n_mirrors == 0


def test_null_fee_counts_as_free():
    out = labelled()
    assert out.loc[4].mirror_zero_later
    assert out.loc[10].mirror_label == "free_other"


def test_permanent_rule_and_sensitivity_set():
    rows = [
        ("paid", "2016-07-01"),
        ("paid_mirror", "2016-06-30"),
        ("free_other", "2016-06-30"),
        ("free_other", "2016-07-01"),
        ("loan_return", "2016-07-01"),
        ("loan_out", "2016-07-01"),
        ("both", "2016-07-01"),
    ]
    mv = pd.DataFrame(rows, columns=["mirror_label", "date"]).assign(
        date=lambda d: pd.to_datetime(d.date)
    )
    out = add_cohort_rules(mv)
    assert out.permanent.tolist() == [True, True, False, True, False, False, False]
    assert out.sensitivity.tolist() == [False] * 5 + [True, False]


def test_levels_follow_the_funnel():
    mv = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "minutes_next_season": [900, 900, 100, 900],
            "dest_club_in_scope_leagues": [True, False, True, True],
        }
    )
    n3, n4, n5 = levels(mv, {1, 2, 3})
    assert n3.tolist() == [True, True, True, False]
    assert n4.tolist() == [True, True, False, False]
    assert n5.tolist() == [True, False, False, False]
