"""
One merging writer for data/source_health.json.

WHY A FILE AND NOT A LOG LINE. TheRundown was wired in, configured with a live
secret, and never once called -- the date list it needed was built from a field
Polymarket does not have, so it sat behind an empty `if` for days while the
site kept quoting Polymarket and looking healthy. The build log is the only
place that would have shown, and CI logs are not readable after the fact
without the gh CLI. This file is committed, so a diagnostic survives the run
that produced it.

WHY ONE WRITER. Four different places report into it, and the first version of
live_props' writer built a fresh dict and wrote it -- deleting every key it did
not own. A health file that erases the other half of its own health is worse
than no file: it reads as "nothing to report". The merge lived inline in one
writer and nowhere else, so every new reporter had to rediscover the rule.
"""

import json
import os
from datetime import datetime, timezone

PATH = "data/source_health.json"


def record(key: str, payload) -> bool:
    """Merge one reporter's block into the health file. Never raises.

    Returns whether it was written. A diagnostic that takes the build down is
    worse than the thing it was diagnosing.
    """
    try:
        existing = {}
        try:
            with open(PATH, encoding="utf-8") as fh:
                existing = json.load(fh)
        except (OSError, ValueError):
            pass
        if not isinstance(existing, dict):
            existing = {}
        existing[key] = payload
        existing["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except Exception as exc:                      # noqa: BLE001 -- see docstring
        print(f"[source_health] {key!r} not written ({exc}) -- continuing")
        return False
