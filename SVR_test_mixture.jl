push!(LOAD_PATH, @__DIR__)
using SVM_Regression_Methods
using KernelFunctions
using LinearAlgebra
using Plots

# test ability to recover 2 component mixture
m, n = 1000, 20
X = randn(m, n)
y1 = sum(X, dims = 2)[:,1] .+ 0.1randn(m)
y2 = (sum(X[:,1:(n ÷ 2)], dims = 2) .- sum(X[:,((n ÷ 2)+1):end], dims = 2))[:,1]
y = [rand() < 0.4 ? y1[i] : y2[i] for i = 1:m]
ϵ = 0.1
C = 10.0
k = 2
kernel = KernelFunctions.GaussianKernel()

mixed_model =  fit_mixture(y,  X, ϵ, C, k, kernel, method = :surrogate, tol = 1.0e-5, max_iters = 1_000)
