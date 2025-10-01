using Pkg
Pkg.add.(["DataFrames","CSV", "Statistics", "MLJ", "BetaML"])

df = CSV.read("bank-loan-dataset.csv", DataFrame)


# split data into traininig/testing sets
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=123456) # the rng is for reproducibility

# create training/testing sets
x_train = x[train, :]
y_train = y[train]
x_test = x[test, :]
y_test = y[test]

# load classifer and 
model = @load RandomForestClassifier verbosity=0 pkg=BetaML
forest = model()

rf = machine(forest, x_train, y_train)

# fit classifer
MLJ.fit!(rf)

y_hat = MLJ.predict(rf, x_test)
y_hat = MLJ.mode.(y_hat)  # convert probabilistic predictions to deterministic ones (ones with highest probability)

# metrics
accuracy = MLJ.accuracy(y_hat, y_test) # compare accuracy in predicted vs actual
println("Accuracy: $accuracy")

cm = StatisticalMeasures.ConfusionMatrices.confmat(y_hat, y_test)
display(cm)

cm = StatisticalMeasures.ConfusionMatrices.matrix(cm)

precision_setosa = cm[1,1] / sum(cm[1, :])
