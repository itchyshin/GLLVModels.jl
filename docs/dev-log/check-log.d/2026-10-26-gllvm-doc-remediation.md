# Check-log: first-wave twin documentation remediation

**Goal:** Correct the approved Julia documentation findings D011, D012, and D036 in coordination with the separate gllvmTMB changes. D010 remains pending its bridge/inference owner.

**Commands and results:**

- Local Documenter/VitePress build with deployment disabled: exited 0. Doctests, cross references, and page rendering passed; VitePress completed. Warnings were large bundle size and default `package.json` template substitution.
- Verified generated HTML and Documenter Markdown for the score uncertainty wording and the explicit Poisson import.
- Executed the exact documented Poisson CV examples for random-cell and site-block splits; both returned finite metrics.
- `git diff --check` passed. Prose style checks returned zero findings.

**Scope:** Documentation and example import only. No engine change, deployment, parity claim, or interval calibration claim.
