"""
Splitting an event name into series and matchup, and what happens when no
main event has been announced.
"""
import sys

sys.path.insert(0, ".")
from generate_site import MAIN_EVENT_TBD, split_event_name   # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# The ordinary case, both card types.
check("numbered PPV splits",
      split_event_name("UFC 331: Van vs. Pantoja 2") == ("UFC 331", "Van vs. Pantoja 2"))
check("fight night splits",
      split_event_name("UFC Fight Night: Hooker vs. Parnasse")
      == ("UFC Fight Night", "Hooker vs. Parnasse"))
check("branded card splits",
      split_event_name("Noche UFC: Silva vs. Delgado") == ("Noche UFC", "Silva vs. Delgado"))
check("a rematch numeral stays with the matchup",
      split_event_name("UFC 331: Van vs. Pantoja 2")[1].endswith(" 2"))

# No headliner announced -- a normal state, not missing data.
check("no colon means no matchup", split_event_name("UFC 332") == ("UFC 332", None))
check("the series survives", split_event_name("UFC 332")[0] == "UFC 332")

# A colon whose tail is a venue is not a matchup.
check("a venue tail is not a matchup",
      split_event_name("UFC Fight Night: Las Vegas") == ("UFC Fight Night", None))
check("'vs' is what makes a matchup",
      split_event_name("UFC Fight Night: Abu Dhabi")[1] is None)

# Degenerate inputs must not produce a dangling separator or a crash.
check("trailing separator is not left on the series",
      split_event_name("UFC 333: ") == ("UFC 333", None))
check("empty name", split_event_name("") == ("", None))
check("None name", split_event_name(None) == ("", None))
check("bare separator does not empty the series",
      split_event_name(":")[0] != "")

# The two presentation rules the templates rely on.
def matchup_slot(n):        # must never be blank
    return split_event_name(n)[1] or MAIN_EVENT_TBD


def heading_slot(n):        # names the card; series alone still does
    s, m = split_event_name(n)
    return m or s


check("the matchup slot is never blank",
      all(matchup_slot(n) for n in ("UFC 332", "UFC Fight Night: Las Vegas", "", None)))
check("the matchup slot says TBD when unannounced",
      matchup_slot("UFC 332") == MAIN_EVENT_TBD)
check("the matchup slot is the real matchup when there is one",
      matchup_slot("UFC 331: Van vs. Pantoja 2") == "Van vs. Pantoja 2")
check("the heading slot names the card, not the placeholder",
      heading_slot("UFC 332") == "UFC 332" and MAIN_EVENT_TBD not in heading_slot("UFC 332"))

# The real schedule must round-trip: every card either names a matchup or is
# reported as unannounced, and nothing comes back blank.
import pandas as pd                                          # noqa: E402
names = set()
for p in ("data/fight_cards.csv", "data/future_cards.csv"):
    try:
        names |= set(pd.read_csv(p)["event_name"].dropna().astype(str))
    except (OSError, KeyError):
        pass
check("every scheduled card yields a non-blank series and matchup slot",
      all(split_event_name(n)[0] and matchup_slot(n) for n in names))
check("no scheduled card puts a venue in the matchup slot",
      not any((split_event_name(n)[1] or "") .lower() in ("las vegas", "abu dhabi")
              for n in names))

print(f"test_event_naming: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
