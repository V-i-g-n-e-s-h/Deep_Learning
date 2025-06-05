# Classification of MNIST dataset using an MLP and Lux.jl

#Stochastic Gradient Descent

using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote, JLD2, Plots
using CSV, DataFrames
rng = Xoshiro(1)
function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

train = CSV.read("./src/dataset/mnist/mnist_train.csv", DataFrame, header=1)
test = CSV.read("./src/dataset/mnist/mnist_test.csv", DataFrame, header=1)

function mnistloader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)   # insert trivial channel dim
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2)) #standardize the data, we divide by the largest value. you can check that it correctly odes the job.
    x4dim = meanpool((x4dim), (2, 2)) #this is being done from experience to reduce dimensionality, you can do it by trial and error for other problems also. but we make 75% efficiency just by doing this in this problem.
    x4dim = flatten(x4dim)
    # ys = permutedims(data.label) .+ 1

    yhot = onehotbatch(Vector(data.label), 0:9)  # make a 10×60000 OneHotMatrix, you can try without doing this, and see if the model learns anything.
    return DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=true)
    # return x4dim, ys
end

x1, y1 = first(mnistloader(train, 128)) #batch size is a hyper-parameter

#===== MODEL =====#

model = Chain(
    Dense(196 => 14, relu), # the number of layers, activation functions, and the number of neurons per layer are all hyper-parameters.
    Dense(14 => 14, relu),
    Dense(14 => 10),
) #MLP, FFNN, DNN ∈ ANN, Vanilla Neural Network

#===== METRICS =====#

const lossfn = CrossEntropyLoss(; logits=Val(true))

import Term: tprintln
using Term.TermMarkdown
using Markdown

tprintln(md"""
$mean(-\sum y log(ŷ) + \epsilon))$
""")

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y, 0:9)
        predicted_class = onecold(Array(first(model(x, ps, st))), 0:9)
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

#===== TRAINING =====#

train_dataloader, test_dataloader = mnistloader(train, 512), mnistloader(test, 10000) #batch size also is a hyper-parameter
ps, st = Lux.setup(rng, model) # model = arch + parameters + (optionally) state

vjp = AutoZygote() # AD backend

train_state = Training.TrainState(model, ps, st, AdamW(lambda=3e-4)) # optimizer

mkpath("./model/mnist/Lux MLP trained models")
### Lets train the model
nepochs = 100
train_accuracy, test_accraucy = 0.1, 0.1
for epoch in 1:nepochs
    ### Training Loop
    stime = time()
    for (x, y) in train_dataloader
        _, _, _, train_state = Training.single_train_step!(
            vjp, lossfn, (x, y), train_state,
        )
    end
    ttime = time() - stime

    # Collect Post Training statistics, such as accuracy
    train_accuracy = accuracy(model, train_state.parameters, train_state.states, train_dataloader) * 100
    test_accraucy = accuracy(model, train_state.parameters, train_state.states, test_dataloader) * 100

    @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch nepochs ttime train_accuracy test_accraucy

    trained_parameters, trained_states = deepcopy(train_state.parameters), deepcopy(train_state.states)
    if epoch % 5 == 0
        @save "./model/mnist/Lux MLP trained models/MLP_trained_model_$(epoch).jld2" trained_parameters trained_states
        println("saved to ", "trained_model_$(epoch).jld2")
    end
end

# HW TODO is to load trained models and use them for testing on the test dataset again and calcualte the balanced_accuracy instead of Accuracy.
using Lux, JLD2, MLUtils, OneHotArrays, CSV, DataFrames

# === Load MNIST Test Data === #
function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

function mnistloader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    x4dim = meanpool(x4dim, (2, 2))
    x4dim = flatten(x4dim)
    yhot = onehotbatch(Vector(data.label), 0:9)
    return DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=false)
end

test = CSV.read("./src/dataset/mnist/mnist_test.csv", DataFrame, header=1)
test_dataloader = mnistloader(test, 10000)  # batch size = 10000 for full eval

# === Model Architecture (must match training) === #
model = Chain(
    Dense(196 => 14, relu),
    Dense(14 => 14, relu),
    Dense(14 => 10),
)

# === Balanced Accuracy === #
function balanced_accuracy(model, ps, st, dataloader)
    st = Lux.testmode(st)
    all_preds, all_targets = Int[], Int[]
    for (x, y) in dataloader
        preds = onecold(Array(first(model(x, ps, st))), 0:9)
        targets = onecold(y, 0:9)
        append!(all_preds, preds)
        append!(all_targets, targets)
    end
    classes = 0:9
    recalls = Float64[]
    for cls in classes
        true_positives = sum((all_preds .== cls) .& (all_targets .== cls))
        actual_positives = sum(all_targets .== cls)
        if actual_positives == 0
            push!(recalls, 0.0)
        else
            push!(recalls, true_positives / actual_positives)
        end
    end
    return mean(recalls)
end

function load_model_and_evaluate(epoch::Int)
    model_path = "./model/mnist/Lux MLP trained models/MLP_trained_model_$(epoch).jld2"
    JLD2.@load model_path trained_parameters trained_states
    bal_acc = balanced_accuracy(model, trained_parameters, trained_states, test_dataloader)
    println("Balanced Accuracy on epoch $epoch model: ", round(bal_acc * 100, digits=2), "%")
end

# === Example: Evaluate Saved Models === #
for epoch in 5:5:100
    load_model_and_evaluate(epoch)
end