GOAL: see GOAL-2026-09-28-overnight.md (run ended); now on-call for the #606 merge train.
STATE (2026-09-28 15:20Z): waiting on merge-train #606 (carries #519, #520, #529, #540 + 20 others; grouped-getLV lane owns it).
DONE: #551 merged with #540 branch, pushed 29bf83a31 (MERGEABLE). #529 merged into #551 locally at 0c5e1f100
  (one dispatcher, four methods; 1,015 tests pass), NOT pushed. #518 merged main locally at 9ebcd4fa5
  (runtests conflict, both includes kept; lane tests 232/232), NOT pushed.
NEXT: when #606 merges -> in the beta-getlv worktree merge origin/main, rerun the 5 grouped tests, push,
  `gh pr edit 551 --base main`, tell the grouped-getLV lane. #518: push only on that lane's ping.
OPEN GATES: NB re-run (~5,500 core-h DRAC) awaits Shinichi. Merges are the train's, not ours.
RESUME: read this file, then `gh pr view 606 --json state`.
