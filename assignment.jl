using Pkg
Pkg.add.(["DataFrames","CSV", "Statistics"])

df = CSV.read("bank-loan-dataset.csv", DataFrame)
