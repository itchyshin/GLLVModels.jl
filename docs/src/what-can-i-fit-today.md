# What can I fit today?

Start with a documented route that matches both your question and your data.
GLLVModels.jl is experimental, so a successful fit is a starting point for
checking a result, not proof that it is ready to report.

## Three practical starting points

### Continuous measurements that may vary together

Use the [first-model tutorial](quickstart.md) when you have several continuous
measurements for the same observations: for example, traits measured on the
same individuals or outcomes measured on the same people. The tutorial fits a
Gaussian model to a response matrix and shows how to read shared and
response-specific variation.

### A continuous trait with an evolutionary tree

Use the [phylogenetic trait tutorial](vignettes/phylogenetic-gllvm.md) when
you have one continuous trait for related species and a supplied evolutionary
tree. It is a separate route because the tree describes a particular source of
similarity among species.

### Species counts across sites

Use the [community abundance tutorial](vignettes/community-abundance.md) when
your rows are species, your columns are sites, and you have environmental
measurements for those sites. Start by reproducing the documented example
before adapting it to a new dataset.

## Before you report a result

Keep the data orientation used by the chosen tutorial, record the model and
package version, and inspect whether the fit converged. Associations estimated
by a model are not, on their own, evidence of causation or direct biological
interaction. If your analysis needs uncertainty intervals, a non-Gaussian
response, or a more complex structure, read the route-specific guide first;
do not assume that a function name guarantees every combination is supported.

## Need a technical comparison?

The R package [gllvmTMB](https://github.com/itchyshin/gllvmTMB) has a broader,
formula-first interface. The [capability-parity record](gllvmtmb-parity.md)
gives detailed technical comparisons and evidence notes. It is useful when
moving an established workflow between packages, rather than as the first
place to learn what to fit.
