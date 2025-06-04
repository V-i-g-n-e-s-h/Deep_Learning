# ## Getting started

using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, select, Not, describe
using Random
using StatsPlots

data = dataset("datasets", "USArrests")
names(data) |> println

# Let's have a look at the mean and standard deviation of each feature:

describe(data, :mean, :std) |> show

# Let's extract the numerical component and coerce

X = select(data, Not(:State))
X = coerce(X, :UrbanPop => Continuous, :Assault => Continuous)

# ## PCA pipeline
#
# PCA is usually best done after standardization but we won't do it here:

PCA = @load PCA pkg = MultivariateStats

pca_mdl = Standardizer |> PCA(variance_ratio=1)
pca = machine(pca_mdl, X)
fit!(pca)
PCA
W = MLJ.transform(pca, X)

# W is the PCA'd data; here we've used default settings for PCA:

schema(W).names

# Let's inspect the fit:

r = report(pca)
cumsum(r.pca.principalvars ./ r.pca.tvar)

# In the second line we look at the explained variance with 1 then 2 PCA features and it seems that with 2 we almost completely recover all of the variance.

# ## More interesting data...

# Instead of just playing with toy data, let's load the orange juice data and extract only the columns corresponding to price data:

data = dataset("ISLR", "OJ")

feature_names = [
    :PriceCH, :PriceMM, :DiscCH, :DiscMM, :SalePriceMM, :SalePriceCH,
    :PriceDiff, :PctDiscMM, :PctDiscCH,
]

X = select(data, feature_names)
y = select(data, :Purchase)

train, test = partition(eachindex(y.Purchase), 0.7, shuffle=true, rng=1515)

using StatsBase
countmap(y.Purchase)
# ### PCA pipeline

Random.seed!(1515)

SPCA = Pipeline(
    Standardizer(),
    PCA(variance_ratio=1.0),
)

spca = machine(SPCA, X)
fit!(spca)
W = MLJ.transform(spca, X)
names(W)

# What kind of variance can we explain?

rpca = report(spca).pca
cs = cumsum(rpca.principalvars ./ rpca.tvar)


# Let's visualise this

using Plots
begin
    Plots.bar(1:length(cs), cs, legend=false, size=((800, 600)), ylim=(0, 1.1))
    xlabel!("Number of PCA features")
    ylabel!("Ratio of explained variance")
    plot!(1:length(cs), cs, color="red", marker="o", linewidth=3)
end
# So 4 PCA features are enough to recover most of the variance.

### HW TODO - Test the performance using LogisticClassifier and compare the performance on PCA features and the original set of features