using Pkg # install packages
Pkg.add.(["DataFrames","CSV", "Statistics", "MLJ", "BetaML"])

# Import the modules
using CSV, DataFrames, MLJ, BetaML, Statistics
import MLJ: partition, fit!, predict

df = CSV.read("bank-loan-dataset.csv", DataFrame)

# separate features and target
x = select(df, Not(:default))  # features
y = df.default  # target
y = coerce(y, Multiclass)  # convert for MLJ

# split our data into traininig/testing sets
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=123456) # the rng is for reproducibility

# create training/testing sets
x_train = x[train, :]
y_train = y[train]
x_test = x[test, :]
y_test = y[test]
 
model = @load RandomForestClassifier verbosity=0 pkg=BetaML
forest = model()

rf = machine(forest, x_train, y_train)

MLJ.fit!(rf)

y_hat = MLJ.predict(rf, x_test)
y_hat = MLJ.mode.(y_hat)  # convert probabilistic predictions to ones with highest probability

accuracy_score = MLJ.accuracy(y_hat, y_test) # compare accuracy in predicted vs actual
println("Accuracy: $accuracy_score")

# confusion matrix
cm = MLJ.confusion_matrix(y_hat, y_test)
println("Confusion Matrix:")
display(cm)

# calculating precision, recall, f1_score 
tp = 0  # true positives
fp = 0  # false positives
tn = 0  # true negatives
fn = 0  # false negatives

for i in 1:length(y_hat)
    global tp, fp, tn, fn
    if y_hat[i] == true
        if y_test[i] == true
            tp += 1  # predicted true, actual true
        else
            fp += 1  # predicted true, actual false
        end
    else
        if y_test[i] == false
            tn += 1  # predicted false, actual false
        else
            fn += 1  # predicted false, actual true
        end
    end
end

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

println("Precision: $precision")
println("Recall: $recall")
println("F1-Score: $f1_score")

