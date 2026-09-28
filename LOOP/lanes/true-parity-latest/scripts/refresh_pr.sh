#!/bin/bash
# Merge origin/main into a PR branch. Auto-resolves ONLY CHANGELOG.md and docs/dev-log/check-log.md by union
# (keep both sides); any other conflict aborts. Pushes by branch name (fast-forward only). No merge of the PR.
set -u
R=itchyshin/GLLVModels.jl
CLONE="/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
N=$1
B=$(gh pr view $N -R $R --json headRefName -q .headRefName)
git -C "$CLONE" fetch -q origin
WT=$(mktemp -d "$(dirname $0)/refresh.XXXX"); rmdir "$WT"
git -C "$CLONE" worktree add -q --detach "$WT" origin/$B || exit 1
cd "$WT"
cleanup(){ cd /; git -C "$CLONE" worktree remove --force "$WT" >/dev/null 2>&1; }
OLD=$(git rev-parse HEAD)
if ! git merge -q --no-edit -m "Merge origin/main into $B (refresh; log files union-resolved)" origin/main >/dev/null 2>&1; then
  for f in $(git diff --name-only --diff-filter=U); do
    case "$f" in
      CHANGELOG.md|docs/dev-log/check-log.md)
        T=$(mktemp -d); git show :2:"$f" > $T/o; git show :1:"$f" > $T/b; git show :3:"$f" > $T/t
        git merge-file --union $T/o $T/b $T/t; cp $T/o "$f"; rm -rf $T; git add "$f";;
      *) echo "ABORT #$N: non-log conflict in $f"; git merge --abort; cleanup; exit 1;;
    esac
  done
  git commit -q --no-edit || { echo "ABORT #$N: commit failed"; cleanup; exit 1; }
fi
# verify: vs main, the change relative to the old head touches only the two log files
extra=$(comm -13 <(git diff --name-only $(git merge-base origin/main $OLD) $OLD | sort) <(git diff --name-only origin/main HEAD | sort))
[ -n "$extra" ] && { echo "ABORT #$N: refresh made the PR touch new files: $extra"; cleanup; exit 1; }
git push -q origin HEAD:refs/heads/$B || { echo "ABORT #$N: push failed"; cleanup; exit 1; }
echo "#$N refreshed ${OLD:0:9} -> $(git rev-parse HEAD)"; cleanup
