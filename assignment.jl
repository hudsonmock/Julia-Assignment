using Pkg # install packages
Pkg.add.(["DataFrames","CSV", "Statistics", "MLJ", "BetaML"])

# Import modules
using CSV, DataFrames, MLJ, BetaML, Statistics
import MLJ: partition, fit!, predict

df = CSV.read("bank-loan-dataset.csv", DataFrame)

# separate features and target
x = select(df, Not(:default))  # features
y = df.default  # target
y = coerce(y, Multiclass)  # convert for MLJ

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
y_hat = MLJ.mode.(y_hat)  # convert probabilistic predictions to ones with highest probability

# metrics
accuracy_score = MLJ.accuracy(y_hat, y_test) # compare accuracy in predicted vs actual
println("Accuracy: $accuracy_score")

# confusion matrix
cm = MLJ.confusion_matrix(y_hat, y_test)
println("Confusion Matrix:")
display(cm)

# calculate precision, recall, f1_score manually
tp = sum((y_hat .== true) .& (y_test .== true))
fp = sum((y_hat .== true) .& (y_test .== false))
tn = sum((y_hat .== false) .& (y_test .== false))
fn = sum((y_hat .== false) .& (y_test .== true))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

println("Precision: $precision")
println("Recall: $recall")
println("F1-Score: $f1_score")
