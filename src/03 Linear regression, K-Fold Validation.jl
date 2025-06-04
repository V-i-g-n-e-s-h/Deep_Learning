# ## Getting started

using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, select
auto = dataset("ISLR", "Auto")
y, X = unpack(auto, ==(:MPG))
train, test = partition(eachindex(y), 0.5, shuffle=true, rng=444)

# Note the use of `rng=` to seed the shuffling of indices so that the results are reproducible.

# ### Polynomial regression

# This tutorial introduces polynomial regression in a very hands-on way. A more
# programmatic alternative is to use MLJ's `InteractionTransformer`. Run
# `doc("InteractionTransformer")` for details.

LR = @load LinearRegressor pkg = MLJLinearModels

# In this part we only build models with the `Horsepower` feature.

using Plots

begin
    plot(X.Horsepower, y, seriestype=:scatter, legend=false, size=(800, 600))
    xlabel!("Horsepower")
    ylabel!("MPG")
end

# Let's get a baseline:

lm = LR()
mlm = machine(lm, select(X, :Horsepower), y)
fit!(mlm, rows=train)
mse = rms(MLJ.predict(mlm, rows=test), y[test])^2

# Note that we square the measure to  match the results obtained in the ISL labs where the mean squared error (here we use the `rms` which is the square root of that).
xx = (Horsepower=range(50, 225, length=100) |> collect,)
yy = MLJ.predict(mlm, xx)

begin
    plot(X.Horsepower, y, seriestype=:scatter, label="Data", legend=false, size=(800, 600))
    plot!(xx.Horsepower, yy, label="Fit", legend=:topright, linewidth=3, color=:orange)
    xlabel!("Horsepower")
    ylabel!("MPG")
end

# We now want to build three polynomial models of degree 1, 2 and 3 respectively; we start by forming the corresponding feature matrix:

hp = X.Horsepower
Xhp = DataFrame(hp1=hp, hp2=hp .^ 2, hp3=hp .^ 3)

# Now we  can write a simple pipeline where the first step selects the features we want (and with it the degree of the polynomial) and the second is the linear regressor:

LinMod = Pipeline(
    FeatureSelector(features=[:hp1]),
    LR()
)

# Then we can  instantiate and fit 3 models where we specify the features each time:

LinMod.feature_selector.features = [:hp1] # poly of degree 1
lr1 = machine(LinMod, Xhp, y) # poly of degree 1 (line)
fit!(lr1, rows=train)

LinMod.feature_selector.features = [:hp1, :hp2] # poly of degree 2
lr2 = machine(LinMod, Xhp, y)
fit!(lr2, rows=train)

LinMod.feature_selector.features = [:hp1, :hp2, :hp3] # poly of degree 3
lr3 = machine(LinMod, Xhp, y)
fit!(lr3, rows=train)

# Let's check the performances on the test set

get_mse(lr) = rms(MLJ.predict(lr, rows=test), y[test])^2

@show get_mse(lr1)
@show get_mse(lr2)
@show get_mse(lr3)

# Let's visualise the models

hpn = xx.Horsepower
Xnew = DataFrame(hp1=hpn, hp2=hpn .^ 2, hp3=hpn .^ 3)

yy1 = MLJ.predict(lr1, Xnew)
yy2 = MLJ.predict(lr2, Xnew)
yy3 = MLJ.predict(lr3, Xnew)

begin
    plot(X.Horsepower, y, seriestype=:scatter, label=false, size=(800, 600))
    plot!(xx.Horsepower, yy1, label="Order 1", linewidth=3, color=:orange,)
    plot!(xx.Horsepower, yy2, label="Order 2", linewidth=3, color=:green,)
    plot!(xx.Horsepower, yy3, label="Order 3", linewidth=3, color=:red,)
    xlabel!("Horsepower")
    ylabel!("MPG")
end

# ## K-Folds Cross Validation

#
# Let's crossvalidate over the degree of the polynomial.
#
# **Note**: there's a  bit of gymnastics here because MLJ doesn't directly support a polynomial regression; see our tutorial on [tuning models](/getting-started/model-tuning/) for a gentler introduction to model tuning.
# The gist of the following code is to create a dataframe where each column is a power of the `Horsepower` feature from 1 to 10 and we build a series of regression models using incrementally more of those features (higher degree):

Xhp = DataFrame([hp .^ i for i in 1:10], :auto)

cases = [[Symbol("x$j") for j in 1:i] for i in 1:10]
r = range(LinMod, :(feature_selector.features), values=cases)

tm = TunedModel(model=LinMod, ranges=r, resampling=CV(nfolds=10), measure=rms)
#train + test => give you 10 splits of train+test
# Now we're left with fitting the tuned model

mtm = machine(tm, Xhp, y)
fit!(mtm)


rep = report(mtm)
res = rep.plotting
rep.best_model

# So the conclusion here is that the ?th order polynomial does quite well.
#
# In ISL they use a different seed so the results are a bit different but comparable.

Xnew = DataFrame([hpn .^ i for i in 1:10], :auto)
yy5 = MLJ.predict(mtm, Xnew)

begin
    plot(X.Horsepower, y, seriestype=:scatter, legend=false, size=(800, 600))
    plot!(xx.Horsepower, yy5, color=:orange, linewidth=4, legend=false)
    xlabel!("Horsepower")
    ylabel!("MPG")
end


### Effect of different features
using DataFrames
LinMod = Pipeline(
    FeatureSelector(features=[:Nmae]),
    LR()
)
names_cols = names(select(X, Not(:Name)))

cases = [[Symbol(names_cols[i]) for i in 1:j] for j in 1:lastindex(names_cols)]
r = range(LinMod, :(feature_selector.features), values=cases)

tm = TunedModel(model=LinMod, ranges=r, resampling=CV(nfolds=10), measure=rms)

# Now we're left with fitting the tuned model

mtm = machine(tm, X, y)
fit!(mtm)
rep = report(mtm)

res = rep.plotting
rep.best_model

best_models_mse_mean = mean(rep.best_history_entry.per_fold[1])^2
best_models_mse_std = std(rep.best_history_entry.per_fold[1])^2

# In this case, the best model is the one that uses all the features.

#HW TODO - find if MSE reduces further if we take and hyperparamaters tune upto 10 powers of each feature!

# Question - How can we use linear regression for classification?


