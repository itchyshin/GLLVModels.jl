#!/bin/sh
# usage: prstate.sh N  -> prints PR_<N>_MERGED, PR_<N>_OPEN_WITH_REASON (open + comment containing "RETURNED:"), or PR_<N>_UNRESOLVED
R=itchyshin/GLLVModels.jl; N=$1
s=$(gh pr view "$N" -R $R --json state -q .state 2>/dev/null)
if [ "$s" = "MERGED" ]; then echo "PR_${N}_MERGED"; exit 0; fi
if gh pr view "$N" -R $R --json comments -q '.comments[].body' 2>/dev/null | grep -q "RETURNED:"; then echo "PR_${N}_OPEN_WITH_REASON state=$s"; exit 0; fi
echo "PR_${N}_UNRESOLVED state=$s"
