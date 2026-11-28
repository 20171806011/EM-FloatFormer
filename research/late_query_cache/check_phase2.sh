#!/usr/bin/env bash
set -eu -o pipefail
# Run from any directory inside the checkout. Give the pinned Mathlib checkout as $1.
root=$(git rev-parse --show-toplevel)
math=$(cd "${1:?pass the Mathlib v4.24.0 directory}" && pwd)
expected=f897ebcf72cd16f89ab4577d0c826cd14afaafc7
test "$(git -C "$math" rev-parse HEAD)" = "$expected"
src="$root/research/late_query_cache"
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
cp "$src"/Phase2*.lean "$stage/"
cd "$math"
mapfile -t deps < <(awk '/^import Mathlib\./ {gsub(/\./,"/",$2); print $2 ".lean"}' "$stage"/*.lean | sort -u)
if [ "${#deps[@]}" -gt 0 ]; then lake exe cache get "${deps[@]}"; fi
lake env lean --version
failed=0
for f in "$stage"/Phase2*.lean; do
  name=$(basename "$f")
  echo "=== $name ==="
  if ! lake env lean --root="$stage" -o "$stage/${name%.lean}.olean" "$f" >"$stage/$name.log" 2>&1; then failed=1; fi
  cat "$stage/$name.log"
  if grep -Fq 'sorryAx' "$stage/$name.log"; then failed=1; fi
done
exit "$failed"
