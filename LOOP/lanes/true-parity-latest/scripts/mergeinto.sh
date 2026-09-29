#!/bin/bash
# usage: mergeinto.sh SRC_BRANCH DST_BRANCH  -- merge origin/SRC into DST, union-resolve log files only, ff push by name
set -u
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"; git fetch -q origin
src=$1; dst=$2; W=${TMPDIR:-/tmp}/mi.$RANDOM
git worktree add -q --detach "$W" "origin/$dst" || exit 1
cd "$W"; OLD=$(git rev-parse HEAD)
if ! git merge -q --no-edit -m "Merge $src into $dst (cascade after main refresh)" "origin/$src" >/dev/null 2>&1; then
  for f in $(git diff --name-only --diff-filter=U); do
    case "$f" in CHANGELOG.md|docs/dev-log/check-log.md)
      T=$(mktemp -d); git show :2:"$f" > "$T/o"; git show :1:"$f" > "$T/b"; git show :3:"$f" > "$T/t"
      git merge-file --union "$T/o" "$T/b" "$T/t"; cp "$T/o" "$f"; git add "$f";;
    *) echo "ABORT $dst: non-log conflict $f"; git merge --abort; cd /; git -C "/Users/z3437171/Dropbox/Github Local/GLLVM.jl" worktree remove "$W"; exit 1;;
    esac
  done
  git commit -q --no-edit
fi
git merge-base --is-ancestor "$OLD" HEAD && git push -q origin "HEAD:refs/heads/$dst" && echo "$dst -> $(git rev-parse HEAD)"
cd /; git -C "/Users/z3437171/Dropbox/Github Local/GLLVM.jl" worktree remove "$W"
