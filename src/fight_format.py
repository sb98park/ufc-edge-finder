"""
One definition of "is this fight five rounds", because five copies was
demonstrably too many.

WHY THIS MODULE EXISTS. The rule -- a fight is scheduled for five rounds if it
is the Main Event OR carries a belt -- lived inline in five places:
card_matcher (x2), edge_finder (x3). When title-fight support was added, the
first patch updated two of them; lint's round-monotonic check caught the third
only because a title co-main happened to be on that week's card, and the
remaining two were found by grep. A rule copied five times is a rule that will
be updated four times.

Worse, the copies had already DRIFTED. Two read the flag as
`str(...).strip().lower() == "true"`, three as `bool(...)`. Those are not the
same test: `is_title_fight` arrives from a CSV, so a fight explicitly marked
`False` comes through as the STRING "False" -- and `bool("False")` is True.
Three of the five call sites would therefore have scheduled a non-title fight
for five rounds the moment any row carried an explicit False rather than a
blank. The round distribution and every Over/Under derived from it would have
been wrong, with nothing to notice.

CSV truthiness is the trap here, not the round rule.
"""


def is_truthy_flag(value) -> bool:
    """
    A CSV flag read as a boolean, correctly.

    `bool("False")` is True, and `is_title_fight` comes out of pandas as
    whatever was in the cell -- an empty string, the string "False", the
    string "True", a real bool, or NaN. Only explicit affirmatives count.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if value != value:          # NaN
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def is_five_round(row) -> bool:
    """
    Main events are five rounds; so are title fights wherever they sit.

    UFC schedules ALL main events for five rounds, belt or not, and a card
    carrying two belts puts a title fight in the co-main slot -- which is
    exactly the case that broke: deriving this from card_position alone
    modelled a title co-main as three rounds, giving a wrong round
    distribution, wrong finish probability and wrong Over/Under lines, with
    no error surfaced anywhere.
    """
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    if str(get("card_position", "") or "").strip() == "Main Event":
        return True
    return is_truthy_flag(get("is_title_fight", None))


def scheduled_rounds(row) -> int:
    return 5 if is_five_round(row) else 3
