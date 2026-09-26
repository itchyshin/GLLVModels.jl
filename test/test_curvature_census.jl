# Structural guard for the Fisher-vs-observed Laplace curvature fault class.
#
# The class exists because `_glm_weight` serves two roles: the Fisher-scoring mode
# search (where expected information is correct) and the Laplace log-det (where TMB
# uses the OBSERVED joint Hessian). A family that ships only a Fisher weight and says
# nothing is silently in the second role with the wrong matrix.
#
# Every prose census of this class so far has been built with `grep`, and every one
# missed sites:
#   * `^_glm_weight(` skipped Beta (`beta.jl:21`) and GP1 (`gp1.jl:65`) — both use
#     `function … end` block form.
#   * A `src/families/*.jl` sweep skipped `_glm_weight(::Normal)` at `spde_latent.jl:54`,
#     which lives outside that directory.
#
# So this guard does NOT grep. It reflects over the method table, which cannot miss a
# definition by formatting or location. Its job is to make the fault class impossible to
# EXTEND silently: add a family with a `_glm_weight` and no curvature declaration and
# this test fails, naming the file and line and telling you what to declare.
#
# It deliberately does not assert which choice is correct — that is a per-family
# modelling decision. It asserts only that a choice was MADE and written down.
#
# REVISED 2026-08-26 after an adversarial review found two holes in the first version:
#
#   1. OVER-CERTIFICATION. The census keyed on the family alone, but the trait is keyed
#      on (family, link). `_glm_weight_matches_observed(::Binomial, ::LogitLink) = true`
#      caused the whole Binomial family to read as safe — while Binomial+probit and
#      Binomial+cloglog report `matches_observed = false` and `_default_hessian = :fisher`,
#      i.e. they are OPEN cells the guard was silently certifying.
#
#   2. UNDER-COVERAGE. The class has a SECOND substrate. Two-part families carry their
#      curvature in `_tp_pieces` (8 definitions across `twopart.jl` and `beta_hurdle.jl`),
#      documented as the identical defect at `twopart.jl:84-90`. The first version covered
#      none of them, so a 9th two-part family with a Fisher `Wc` would have extended the
#      class with this test green.

using GLLVModels, Test, InteractiveUtils, ForwardDiff, Distributions

const G = GLLVModels

_family_of(m) = (p = Base.unwrap_unionall(m.sig).parameters;
                 length(p) >= 2 ? p[2] : nothing)

# Families whose curvature question is OPEN: they ship a Fisher weight, TMB's log-det
# wants the observed one, and the flip is a pending maintainer decision (it is a parity
# change with a measured accuracy cost — see docs/dev-log/check-log.md 2026-08-26).
# This list is the ledger. Shrinking it is progress; growing it silently is the bug.
# NB2 left this set 2026-08-27 (both campaign metrics agreed). Beta, NB1 and
# StudentTFamily left 2026-08-27 under decision A. GeneralizedPoisson1 left
# 2026-08-28 the OTHER way — adjudicated and Fisher RETAINED (see
# DEFERRED_BY_DECISION below). TweedieED left 2026-08-28 under the maintainer's
# decision batch (docs/dev-log/decisions/2026-08-28-arc-decision-batch.md):
# flip to :observed for TMB/gllvmTMB structural parity. The set is now empty —
# Binomial/probit joins it here too (flipped 2026-08-28, same decision batch);
# Binomial/cloglog ALSO flips to :observed (2026-09-01, maintainer decisions
# round 1 item 2 — a confirmed Julia-side likelihood-value defect against R,
# not a pending-flip modelling question; see
# docs/dev-log/core070/cloglog-leaf-notes.md). All three are certified per
# (family, link) below, not by family name (Binomial is already "declared
# safe" as a family via the LogitLink trait, so probit and cloglog were never
# tracked in THIS set).
const KNOWN_OPEN = Set{Symbol}()

# Families where Fisher == observed for STRUCTURAL reasons that are not expressed as a
# `_glm_weight_matches_observed` trait, each with the reason recorded.
# MEASURED 2026-08-26. The first version of this file put Normal and Exponential in one
# "structurally exempt" dict. That conflated two different things, and the label was hiding
# a 705% discrepancy:
#
#   Normal / IdentityLink      worst Fisher-vs-observed gap    0.00%
#   Exponential / LogLink      worst Fisher-vs-observed gap  705.50%
#
# Exponential is NOT structurally safe. It is an open curvature cell deliberately left on
# `:fisher` for an unrelated reason (routing it through the grouped kernel made ‖Λ‖ run away
# to ~960 against a true 0.38 — exponential.jl:55-66). That is a DECISION, not an identity,
# and it must not be filed as though the two curvatures agree.
#
# Probe: docs/dev-log/pending/onepart-exempt-probe.jl

# Fisher ≡ observed as a matter of fact. Machine-checked below — an entry here that is not
# actually an identity FAILS.
const EXEMPT_BY_IDENTITY = Dict(
    :Normal => "Gaussian/identity: log-density quadratic in η, curvature is y-free",
)

# Curvatures genuinely DIFFER; Fisher is retained deliberately, for a reason on record.
# These are open cells with a decision attached, not safe cells. Each needs a citation.
# Exponential left this dict 2026-08-27: the grouped-kernel detour was retired
# (the :observed route now evaluates through the generic core, whose safe mode
# solver ended the ‖Λ‖ runaway), and `_default_hessian(::Exponential, ::LogLink)`
# is now DECLARED :observed — matching the fitter default shipped since
# 2026-08-24 and making the covariates/quadratic/mixed/SPDE/phylo-GLM/
# coevolution kernels agree with it.
const DEFERRED_BY_DECISION = Dict{Symbol,String}(
    :GeneralizedPoisson1 => "ADJUDICATED 2026-08-28, Fisher RETAINED on the 150-cell " *
        "campaign: medians lean observed (+0.1…+0.45) but a minority of cells derail " *
        "badly under the observed weight (means −5.6/−10.3 medium/strong; |err| 15.2 " *
        "vs 1.5) — the documented negative-curvature tail (1+2αy−αμ<0). A closed cell, " *
        "closed the other way; hessian=:observed stays reachable explicitly.",
)

@testset "Laplace curvature census (structural guard)" begin
    weight_methods = collect(methods(G._glm_weight))
    @test !isempty(weight_methods)

    # Families that declare Fisher ≡ observed via the trait.
    declared_safe = Set{Symbol}()
    for m in methods(G._glm_weight_matches_observed)
        f = _family_of(m)
        f === Any && continue
        push!(declared_safe, nameof(f))
    end

    # Families that supply an observed-curvature override.
    has_observed = Set{Symbol}()
    for m in methods(G._glm_obs_weight)
        f = _family_of(m)
        f === Any && continue          # the generic ForwardDiff fallback
        push!(has_observed, nameof(f))
    end

    undeclared = Tuple{Symbol,String}[]
    for m in weight_methods
        f = _family_of(m)
        f === nothing && continue
        name = nameof(f)
        site = string(basename(string(m.file)), ":", m.line)

        declared = name in declared_safe ||
                   name in has_observed ||
                   name in KNOWN_OPEN ||
                   haskey(EXEMPT_BY_IDENTITY, name) ||
                   haskey(DEFERRED_BY_DECISION, name)

        declared || push!(undeclared, (name, site))
    end

    if !isempty(undeclared)
        msg = join(["  $(n) @ $(s)" for (n, s) in undeclared], "\n")
        @error """
        A family defines `_glm_weight` but declares nothing about its Laplace curvature.

        $msg

        `_glm_weight` is used BOTH for the Fisher-scoring mode search and for the
        marginal's log-det. TMB's log-det uses the OBSERVED joint Hessian, so a family
        that only ships the Fisher weight silently computes a different marginal.

        Declare one of:
          * `_glm_weight_matches_observed(::F, ::L) = true`  — if the link is canonical
            and the two coincide (verify it, do not assume it);
          * `_glm_obs_weight(f::F, μ, n, me, y, link, η) = …` plus
            `_default_hessian(::F, ::L) = :observed`  — if they differ and you are fixing it;
          * add it to `KNOWN_OPEN` in this file — if the flip is a pending decision.
        """
    end
    @test isempty(undeclared)

    # The two buckets must stay disjoint: a family cannot be both fixed and open.
    @test isempty(intersect(has_observed, KNOWN_OPEN))

    # Every KNOWN_OPEN entry must actually still define a Fisher weight. If a family is
    # fixed and someone forgets to remove it here, this catches the stale ledger entry —
    # which is exactly how the old prose tally came to read "NB1 … fixed" for an
    # untouched line.
    weight_names = Set(nameof(_family_of(m)) for m in weight_methods
                       if _family_of(m) !== nothing)
    stale = setdiff(KNOWN_OPEN, weight_names)
    @test isempty(stale)

    # A fixed family must pair its override with a family-SPECIFIC `_default_hessian`,
    # or the override is dead code and the marginal still uses the Fisher weight.
    #
    # This must test for a *specialised* method, not merely an applicable one: the
    # generic fallback `_default_hessian(family, link::Link)` at laplace.jl:190 applies
    # to every family, so a `hasmethod` probe would pass vacuously for all of them.
    specialised_default = Set{Symbol}()
    for m in methods(G._default_hessian)
        f = _family_of(m)
        (f === nothing || f === Any) && continue
        push!(specialised_default, nameof(f))
    end
    for name in has_observed
        @test name in specialised_default
    end

    # ---- Hole 1: certification is per (family, link), not per family ----------------
    #
    # A trait declared for one link must not certify the family's other links. This is a
    # golden-set assertion: the computed set of certified cells must equal the recorded
    # one, so gaining OR losing a certification fails until the ledger is updated.
    _specialised(fn, F, L) = try
        _family_of(which(fn, Tuple{F, L})) !== Any
    catch
        false
    end

    CERTIFIED_CELLS = Set([
        (:TruncatedPoisson, :LogLink),    # trait: canonical log
        (:CensoredPoisson,  :LogLink),    # trait: hand-derived observed
        (:Poisson,          :LogLink),    # trait: canonical log
        (:Binomial,         :LogitLink),  # trait: canonical logit
        (:TruncatedNegBin2, :LogLink),    # _default_hessian = :observed
        (:Gamma,            :LogLink),    # _default_hessian = :observed
        (:NegativeBinomial, :LogLink),    # _default_hessian = :observed (2026-08-27 campaign)
        (:Beta,             :LogitLink),    # _default_hessian = :observed (decision A)
        (:NB1,              :LogLink),      # _default_hessian = :observed (decision A)
        (:StudentTFamily,   :IdentityLink), # _default_hessian = :observed (decision A)
        (:Exponential,      :LogLink),      # _default_hessian = :observed (declared; audit fix)
        (:TweedieED,        :LogLink),      # _default_hessian = :observed (2026-08-28 maintainer decision)
        (:Binomial,         :ProbitLink),   # _default_hessian = :observed (2026-08-28 maintainer decision)
        (:Binomial,         :CLogLogLink),  # _default_hessian = :observed (2026-09-01, confirmed defect fix — see docs/dev-log/core070/cloglog-leaf-notes.md)
    ])

    families = unique([_family_of(m) for m in weight_methods if _family_of(m) !== nothing])
    computed = Set{Tuple{Symbol,Symbol}}()
    for F in families, L in subtypes(G.Link)
        (_specialised(G._glm_weight_matches_observed, F, L) ||
         _specialised(G._default_hessian, F, L)) &&
            push!(computed, (nameof(F), nameof(L)))
    end
    @test computed == CERTIFIED_CELLS

    # ---- Hole 2: the two-part substrate --------------------------------------------
    #
    # `_tp_pieces` returns the Fisher `Wc` (see the comment at twopart.jl:84-90). A
    # two-part family is fixed only if it supplies a specialised `_tp_observed_Wc`.
    # Two-part families where Fisher == observed, MEASURED not assumed: a nested
    # ForwardDiff second derivative of the family's own `_tp_pieces` log-density wrt ηc
    # agrees with the returned Fisher `Wc` to 0.0% relative gap across the probe grid.
    # The instrument was validated against DeltaGamma's merged `_tp_observed_Wc` override
    # to <= 2.4e-16 before being trusted. Probe:
    # `docs/dev-log/pending/twopart-curvature-probe.jl`.
    TWOPART_STRUCTURALLY_EXEMPT = Dict(
        :DeltaLogNormal => "conditional part is Gaussian in ηc; curvature is y-free",
        :HurdlePoisson  => "canonical log link on the conditional Poisson part",
    )

    # Genuinely open: measured worst-case relative gap between Fisher and observed, and
    # how many probe cells have NEGATIVE observed curvature (a PD-guard risk if flipped).
    #   HurdleNB   361%, 0 negative      ZIPoisson  280%, 3 negative
    #   ZINB      1223%, 3 negative      ZIB        214%, 6 negative
    #   BetaHurdle 127%, 2 negative
    # HurdleNB re-measured for #500 (was 251%): the #484 chain-rule fix (a = r/(r+mu))
    # changed the Fisher Wc this gap is measured against, so the old figure was against
    # the pre-#484 wrong Fisher weight, not the current one.
    TWOPART_KNOWN_OPEN = Set([:HurdleNB, :ZIPoisson, :ZINB, :ZIB, :BetaHurdle])

    tp_observed = Set{Symbol}()
    for m in methods(G._tp_observed_Wc)
        f = _family_of(m)
        (f === nothing || f === Any) && continue
        push!(tp_observed, nameof(f))
    end

    tp_undeclared = Tuple{Symbol,String}[]
    for m in methods(G._tp_pieces)
        f = _family_of(m)
        f === nothing && continue
        name = nameof(f)
        (name in tp_observed || name in TWOPART_KNOWN_OPEN ||
         haskey(TWOPART_STRUCTURALLY_EXEMPT, name)) && continue
        push!(tp_undeclared, (name, string(basename(string(m.file)), ":", m.line)))
    end

    if !isempty(tp_undeclared)
        @error """
        A two-part family defines `_tp_pieces` but declares nothing about its curvature.

        $(join(["  $(n) @ $(s)" for (n, s) in tp_undeclared], "\n"))

        `_tp_pieces` returns the Fisher `Wc`; the marginal's log-det wants the observed
        one. Supply `_tp_observed_Wc(f::F, y, ηc, Wc)`, or add the family to
        `TWOPART_KNOWN_OPEN` in this file if the flip is a pending decision.
        """
    end
    @test isempty(tp_undeclared)

    # A two-part family cannot be both fixed and open.
    @test isempty(intersect(tp_observed, TWOPART_KNOWN_OPEN))

    # Stale-ledger check, same as for the single-part substrate.
    tp_names = Set(nameof(_family_of(m)) for m in methods(G._tp_pieces)
                   if _family_of(m) !== nothing)
    @test isempty(setdiff(TWOPART_KNOWN_OPEN, tp_names))

    # ---- Single-part identity claims are machine-checked -----------------------------
    #
    # An entry in EXEMPT_BY_IDENTITY asserts Fisher == observed as a fact. Verify it, so the
    # label cannot hide a discrepancy the way it hid Exponential's 705%.
    _sp_instance = Dict(:Normal => Normal(0.0, 1.0))
    _sp_link = Dict(:Normal => G.IdentityLink())
    for (name, _) in EXEMPT_BY_IDENTITY
        fam = get(_sp_instance, name, nothing); lnk = get(_sp_link, name, nothing)
        @test fam !== nothing && lnk !== nothing
        (fam === nothing || lnk === nothing) && continue
        for y in (-1.5, 0.4, 2.2), η in (-0.7, 0.2, 1.1)
            μ = G.linkinv(lnk, η); me = G.mu_eta(lnk, η)
            @test isapprox(G._glm_obs_weight(fam, μ, 1, me, y, lnk, η),
                           G._glm_weight(fam, μ, 1, me); rtol = 1e-8, atol = 1e-10)
        end
    end

    # A family cannot be both an identity and a deferred decision.
    @test isempty(intersect(keys(EXEMPT_BY_IDENTITY), keys(DEFERRED_BY_DECISION)))

    # ---- The exemption must be MEASURED, not asserted -------------------------------
    #
    # A negative control exposed this: moving a genuinely-open family into
    # TWOPART_STRUCTURALLY_EXEMPT with an invented justification passed silently. An
    # unchecked escape hatch can silence exactly the defect this guard exists to catch.
    #
    # So every exemption is now verified numerically: the observed curvature
    # (-d²logf/dηc², nested ForwardDiff on the family's own `_tp_pieces` log-density)
    # must equal the Fisher `Wc` the family returns. Claiming an exemption for a family
    # where they differ now FAILS.
    _tp_instance = Dict(:DeltaLogNormal => G.DeltaLogNormal(1.0),
                        :HurdlePoisson  => G.HurdlePoisson())

    for (name, reason) in TWOPART_STRUCTURALLY_EXEMPT
        fam = get(_tp_instance, name, nothing)
        @test fam !== nothing            # an exemption with no instance cannot be checked
        fam === nothing && continue
        for y in (0.0, 1.0, 4.0), ηz in (-0.8, 0.5), ηc in (-1.0, 0.3, 1.2)
            fisher = G._tp_pieces(fam, y, ηz, ηc)[4]
            observed = -ForwardDiff.derivative(
                a -> ForwardDiff.derivative(b -> G._tp_pieces(fam, y, ηz, b)[5], a), ηc)
            @test isapprox(observed, fisher; rtol = 1e-8, atol = 1e-10)
        end
    end
end
