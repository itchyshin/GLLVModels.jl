# The five admission-twin cases and two reproducers, as the Julia door spells
# them (the R spelling is in export_admission_twins_p1.R). Shared by
# test/parity/isdm_admission_twins.jl, export_julia_admission_estimates.jl and
# tools/true_parity_julia_receipts.jl. Requires `using GLLVModels, Distributions`
# and isdm_fixture_io.jl.
isdm_admission_r_values() = TOML.parsefile(joinpath(ISDM_FIXTURE_DIR, "r_values_admission_p1.toml"))

const ISDM_ADMISSION_CASES = ("adm_aliased", "adm_align", "adm_nooffset", "adm_zeroord", "adm_unbalanced")

function isdm_admission_case(name::AbstractString)
    cl = (Binomial(), CLogLogLink())
    if name == "adm_aliased"
        return (csv = "adm_aliased.csv",
                formula = :(value ~ 0 + trait + trait & env + offset(log_support)),
                family = isdm_sources(gbif = isdm_source(Poisson(); observation = :(~ access + access2)),
                                      survey = cl))
    elseif name == "adm_align"
        # survey is declared FIRST; the data (and sorted levels) put gbif first.
        return (csv = "adm_align.csv",
                formula = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support)),
                family = isdm_sources(survey = cl, gbif = Poisson()))
    elseif name == "adm_nooffset"
        return (csv = "adm_nooffset.csv",
                formula = :(value ~ 0 + trait + trait & env + trait & src_gbif),
                family = isdm_sources(gbif = Poisson(), survey = cl))
    elseif name == "adm_zeroord"
        return (csv = "adm_zeroord.csv",
                formula = :(value ~ 0 + trait + trait & env + offset(log_support)),
                family = isdm_sources(gbif = Poisson(), inat = Poisson()))
    elseif name == "adm_unbalanced"
        return (csv = "adm_unbalanced.csv",
                formula = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support) +
                            latent(0 + trait | cell_id, d = 1, unique = FALSE)),
                family = isdm_sources(gbif = Poisson(), survey = cl))
    end
    error("unknown iSDM admission case $name")
end
