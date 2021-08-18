using Distributed
addprocs(10)
@everywhere push!(LOAD_PATH, @__DIR__)
@everywhere using SVM_Regression_Methods
using KernelFunctions
using LinearAlgebra
using Plots
using Distributions
@everywhere using CUDA

# test ability to recover 2 component mixture
m, n = 1000, 20
X = randn(m, n)
y1 = log.(sum(exp.(X), dims=2))[:,1] .+ 0.1randn(m) # sum(X, dims = 2)[:,1] .+ 0.1randn(m)
y2 = sin.(sum(X, dims=2))[:,1] .+ 0.1randn(m) # (sum(X[:,1:(n ÷ 2)], dims = 2) .- sum(X[:,((n ÷ 2)+1):end], dims = 2))[:,1]
y_models = hcat(y1, y2)
selection = [rand() < 0.2 ? 1 : 2 for i = 1:m]
y = [y_models[i, selection[i]] for i = 1:m]

ϵ = 0.1
C = 10.0
k = 2
kernel = KernelFunctions.PolynomialKernel(degree = 2, c = 0.5)

mixed_model =  fit_mixture(y,  X, ϵ, C, k, kernel, method = :surrogate, tol = 1.0e-5, max_iters = 1_000)
probs(mixed_model)
pred = predict(mixed_model, X)
plt = plot(y)
plot!(plt, pred)

mixed_model_cu =  fit_mixture(cu(y), cu(X), convert(Float32,ϵ), convert(Float32, C), k, kernel, method = :surrogate, tol = 1.0e-5, max_iters = 1_000)
probs(mixed_model_cu)
pred_cu = predict(mixed_model_cu, X)
plt = plot(y)
plot!(plt, pred_cu)

mixed_model2 =  fit_mixture(y,  X, ϵ, C, k, kernel, method = :surrogate, max_points=500, tol = 1.0e-5, max_iters = 1_000)
probs(mixed_model2)
dm = to_dict(mixed_model2)
fdm = from_dict(MixtureModel{Univariate, Continuous, SVR_ConditionalDensity}, dm, expected_kernel=kernel)

w = [SVM_Regression_Methods.posterior_probs(mixed_model2, y[i], X[i,:]) for i = 1:m]
mean(w)
