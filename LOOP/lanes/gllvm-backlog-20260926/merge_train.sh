#!/bin/bash
# Head-pinned merge-when-green train for GLLVModels.jl. Usage: merge_train.sh N:SHA [N:SHA ...]
# Advisory exemption: only "Frozen R 0.7.0 family smoke (advisory...)" may fail, and only at main's own
# range (passed >= 277 and failed <= 9). Every other check must be SUCCESS/NEUTRAL/SKIPPED.
R=itchyshin/GLLVModels.jl
CLONE="/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
for spec in "$@"; do
  N=${spec%%:*}; H=${spec#*:}
  cur=$(gh pr view $N -R $R --json headRefOid -q .headRefOid)
  if [ "$cur" != "$H" ]; then echo "NOT MERGED #$N: head moved ($cur != $H); stopping train"; exit 1; fi
  git -C "$CLONE" fetch -q origin
  if ! git -C "$CLONE" merge-tree --write-tree origin/main $H >/dev/null 2>&1; then echo "NOT MERGED #$N: conflicts with current main; stopping train"; exit 1; fi
  echo "#$N: waiting for checks on ${H:0:9}"
  for i in $(seq 1 180); do
    runs=$(gh api "repos/$R/commits/$H/check-runs?per_page=100" -q '.check_runs | group_by(.name) | map(max_by(.started_at))')
    n=$(echo "$runs" | jq 'length'); pend=$(echo "$runs" | jq '[.[]|select(.status!="completed")]|length')
    [ "$n" -gt 0 ] && [ "$pend" -eq 0 ] && break
    sleep 60
  done
  echo "$runs" | jq -r '.[]|"  \(.conclusion)\t\(.name)"'
  [ "$pend" -ne 0 ] && { echo "NOT MERGED #$N: checks still pending after 3 h"; exit 1; }
  bad=$(echo "$runs" | jq -r '[.[]|select((.conclusion|IN("success","neutral","skipped"))|not)|select(.name|startswith("Frozen R")|not)]|length')
  [ "$bad" -ne 0 ] && { echo "NOT MERGED #$N: non-advisory check not green; stopping train"; exit 1; }
  adv=$(echo "$runs" | jq -r '[.[]|select(.name|startswith("Frozen R"))|select(.conclusion!="success")]|.[0].id // empty')
  if [ -n "$adv" ]; then
    c=$(gh api repos/$R/actions/jobs/$adv/logs 2>/dev/null | grep -oE '[0-9]+ passed, [0-9]+ failed' | tail -1)
    p=$(echo "$c" | awk '{print $1}'); f=$(echo "$c" | awk '{print $3}')
    echo "  advisory: $c"
    if [ -z "$p" ] || [ "$p" -lt 277 ] || [ "$f" -gt 9 ]; then echo "NOT MERGED #$N: advisory outside main's range; stopping train"; exit 1; fi
  fi
  gh pr ready $N -R $R >/dev/null 2>&1
  gh pr merge $N -R $R ${MERGE_METHOD:---merge} --match-head-commit $H --delete-branch >/dev/null 2>&1
  st=$(gh pr view $N -R $R --json state,mergeCommit -q '"\(.state) \(.mergeCommit.oid // "")"')
  echo "#$N: $st"
  case "$st" in MERGED*) ;; *) echo "NOT MERGED #$N: merge call failed; stopping train"; exit 1;; esac
done
echo "TRAIN DONE"
