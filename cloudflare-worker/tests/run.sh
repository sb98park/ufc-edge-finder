#!/usr/bin/env bash
# Every security property the paywall depends on, checked without deploying.
#
# These are the two places a subscription gate actually gets broken: a cookie
# a user can edit, and a token whose algorithm the verifier trusts. Both are
# cheap to test and expensive to get wrong, so they run offline in seconds.
set -e
cd "$(dirname "$0")"
for t in *.test.mjs; do echo "== $t"; node "$t"; done
echo "all worker tests passed"
