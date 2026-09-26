export const meta = {
  name: 'fix-build-review',
  description: 'Build one silent-failure fix PR in GLLVModels.jl, then independently review it',
  phases: [
    { title: 'Build', detail: 'Sonnet builder: test red on main, fix, green on branch, draft PR' },
    { title: 'Review', detail: 'Fable reviewer: independent evidence, verdict' },
  ],
}

const CLONE = '/Users/z3437171/Dropbox/Github Local/GLLVM.jl'
const KIT = '/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/LOOP/lanes/gllvm-backlog-20260926'
const it = args
const MODE_REVIEW = `on origin/main the code really fails silently at about the claimed rate, and on the branch every return is either -Inf or a certified stationary point (your own probe, your own seeds, an independent stationarity check sharing no code with the PR's test); whether results change where main was already healthy, both per site and for whole fits (report the largest change; a better optimum is still a change);`
const MODE_METHOD = `- Mirror the merged precedent PRs named below (read them with gh pr diff N -R itchyshin/GLLVModels.jl): damped step (halve any step that lowers the per-site log-posterior), a convergence test that needs both a small gradient and a small full step, one retry with a 20x iteration budget, return -Inf so the fitter's 1e12 failure sentinel fires when no stationary point is certified, and results unchanged (to 1e-8) at sites where the old loop converged. getLV/predict call sites keep a no-sentinel contract.
- Test first: a test that is red on origin/main and green on your branch. Tests assert RELATIONS (every returned value is either -Inf or a stationary point certified by an independent ForwardDiff check; old-healthy sites unchanged), never one seed's fitted number: CI runs Julia 1.10 and 1.13 on Linux, the same seed draws different data on those versions, and optimiser paths differ between Linux and macOS. A specific regression case goes in a committed literal fixture under test/fixtures/ (TOML, as #507 did). Register the new test in test/runtests.jl.
- Also check whole fits, not only sites: on a handful of datasets where main's fit converges with every site stationary, does the fitted loglik change on your branch? Report what you find (a better optimum still counts as a change).`

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['PR_OPENED', 'NOT_LIVE', 'BLOCKED', 'STOPPED_NEEDS_SHINICHI'] },
    pr_number: { type: 'number' },
    head_sha: { type: 'string' },
    rate_on_main: { type: 'string' },
    summary: { type: 'string' },
    healthy_results_change_evidence: { type: 'string' },
    not_covered: { type: 'string' },
  },
  required: ['status', 'summary', 'not_covered'],
}
const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['MERGE', 'DO_NOT_MERGE', 'NEEDS_SHINICHI'] },
    blocking: { type: 'array', items: { type: 'string' } },
    should_fix: { type: 'array', items: { type: 'string' } },
    healthy_results_change: { type: 'boolean' },
    max_change_at_healthy_sites: { type: 'string' },
    evidence: { type: 'string' },
    ci_state: { type: 'string' },
  },
  required: ['verdict', 'blocking', 'healthy_results_change', 'evidence'],
}

const COMMON_BUILD = `
You are building ONE pull request in the Julia package GLLVModels.jl (GitHub itchyshin/GLLVModels.jl; main clone at "${CLONE}"). It fixes a silent failure: code that can return a finite, plausible value when the computation actually failed, so a fit looks fine but is wrong.

How to work:
- Make your own worktree: git -C "${CLONE}" fetch origin, then git -C "${CLONE}" worktree add ~/local-scratch/GLLVM.jl-${it.key} -b ${it.branch} origin/main. If that worktree or branch already exists from an earlier cut-off attempt, inspect it and reuse what is sound.
- Before editing, claim a lease on the exact files you will change (never a directory): LANE_ID=${it.lane} bash ~/shinichi-brain/tools/lane_lease.sh --claim GLLVM.jl --paths <files>. If refused, wait a few minutes and retry; never bypass a refusal. Claim CHANGELOG.md only briefly, when you write its entry. Release your leases (--release) once the PR is pushed.
${it.method || MODE_METHOD}
- Run with JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 on both julia +1.10 and julia +1.13: your new test file plus the existing test files for this family. Not the full suite. Keep each Julia run under 30 minutes.
- CHANGELOG.md entry in the precedent's style. No em dashes anywhere you write (code comments, CHANGELOG, PR body).
- Commit messages end with the line: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
- Push and open a DRAFT PR against main. Body: what was wrong, the measured rate on main, the fix, test evidence (red on main, green on branch, both Julia versions), what happens to fits that were healthy, and what the PR does NOT cover. Write "Part of #${it.issue}", never "Fixes #${it.issue}" (tracking issue). Body ends with: 🤖 Generated with [Claude Code](https://claude.com/claude-code). Before creating the PR, run python3 ~/shinichi-brain/tools/agent_mention_check.py --text <body-file> and python3 ~/shinichi-brain/tools/slop_check.py <body-file>; fix what they flag.
- Never merge, never mark ready, never push anywhere but your branch.
Hard limits: do not edit src/model_selection.jl, src/cv.jl, the K/d argument handling in src/families/fit_gllvm.jl, the shared generic _laplace_mode (src/families/laplace.jl), Project.toml, frozen contracts or required-cell files, or anything in the gllvmTMB repo. Never use SendMessage to contact other agents; if something blocks you, say so in your result. If the fix needs new public API, stop and return STOPPED_NEEDS_SHINICHI.

THIS PR:
${it.task}
`

const reviewPrompt = (pr, head) => `
You are the independent reviewer for draft PR #${pr} (head ${head}) in itchyshin/GLLVModels.jl, main clone at "${CLONE}". Overnight, the lane merges a silent-failure fix without a human if you find nothing blocking and CI is green, so you are the last check before it lands. Decide whether the fix is correct, stays in scope, and leaves healthy fits alone.

Context: ${it.review_context}

Settle these with your own evidence, not the builder's: ${it.review_checks || MODE_REVIEW} whether the diff touches only what it claims and updates every call site; whether the PR's tests assert relations rather than one seed's outcome and pass on julia +1.10 and +1.13 (JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1); whether the body and CHANGELOG are accurate, say "Part of #N" for a tracking issue, and have no em dashes (run python3 ~/shinichi-brain/tools/agent_mention_check.py --text on the body).

Work in your own scratch worktree under ~/local-scratch/review-${pr}. Do not push, merge, comment on GitHub, or edit the PR, and never use SendMessage to contact other agents. Pending CI is not blocking (the merge train gates on it); report its state. Write your review to ${KIT}/reviews/pr-${pr}.md. Verdict: MERGE only if nothing blocks and healthy fits are unchanged; NEEDS_SHINICHI if healthy-fit results change, public API is added, or a kernel shared by many families changes; otherwise DO_NOT_MERGE with the blocking items.
`

phase('Build')
const b = await agent(COMMON_BUILD, { label: `build:${it.key}`, phase: 'Build', schema: BUILD_SCHEMA, model: 'sonnet', effort: 'high' })
if (!b || b.status !== 'PR_OPENED' || !b.pr_number) { log(`build:${it.key} ended ${b ? b.status : 'null'}; no review`); return { item: it.key, build: b, review: null } }
phase('Review')
const r = await agent(reviewPrompt(b.pr_number, b.head_sha || 'the current head'), { label: `review:${it.key}`, phase: 'Review', schema: REVIEW_SCHEMA, model: 'fable', effort: 'high' })
return { item: it.key, build: b, review: r }
