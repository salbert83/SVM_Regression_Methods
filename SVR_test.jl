push!(LOAD_PATH, @__DIR__)
using SVM_Regression_Methods
using KernelFunctions
using LinearAlgebra

# warm up the solvers
m, n = 100, 20
X = randn(m, n)
# y = sin.(sum(X, dims=2))[:,1] .+ log.(sum(exp.(X), dims=2))[:,1] .+ 0.1randn(m)
y = (sum(X[:,1:(n ÷ 2)], dims=2) .- sum(X[:,((n ÷ 2) + 1):end], dims=2))[:,1]
ϵ = 0.01
C = 10.0*rand(m)

# Kernel
kernel = KernelFunctions.GaussianKernel()

svr1 = fit(SVM_Regression, y, X, kernel, ϵ, C, assemble_kernel = true, method = :cvx_primal)
cst1 = cost(svr1, y, X, ϵ, C)
y_pred1 = predict(svr1, X)

svr2 = fit(SVM_Regression, y, X, kernel, ϵ, C, assemble_kernel = true, method = :cvx_dual)
cst2 = cost(svr2, y, X, ϵ, C)
y_pred2 = predict(svr2, X)

svr3 = fit(SVM_Regression, y, X, kernel, ϵ, C, assemble_kernel = true, method = :surrogate)
cst3 = cost(svr3, y, X, ϵ, C)
y_pred3 = predict(svr3, X)


# benchmark

# The problem
m, n = 10_000, 20
X = randn(m, n)
# y = sin.(sum(X, dims=2))[:,1] .+ log.(sum(exp.(X), dims=2))[:,1] .+ 0.1randn(m)
y = (sum(X[:,1:(n ÷ 2)], dims=2) .- sum(X[:,((n ÷ 2) + 1):end], dims=2))[:,1]
ϵ = 0.01
C = 100.0*rand(m)

# Kernel
kernel = KernelFunctions.GaussianKernel()

# Only include these for tests where m ~ 1000
#=
svr1 = @time fit(SVM_Regression, y, X, kernel, ϵ, C, assemble_kernel = true, method = :cvx_primal)
@show cst1 = cost(svr1, y, X, ϵ, C)
y_pred1 = predict(svr1, X)
@show norm(y - y_pred1, 2)/norm(y,2)

svr2 = @time fit(SVM_Regression, y, X, kernel, ϵ, C, assemble_kernel = true, method = :cvx_dual)
@show cst2 = cost(svr2, y, X, ϵ, C)
y_pred2 = predict(svr2, X)
@show norm(y - y_pred2, 2)/norm(y,2)
=#

svr3 = @time fit(SVM_Regression, y, X, kernel, ϵ, C, assemble_kernel = true, method = :surrogate)
@show cst3 = cost(svr3, y, X, ϵ, C)
y_pred3 = predict(svr3, X)
@show norm(y - y_pred3, 2)/norm(y,2)

using JSON
testfile = "C:\\Users\\salbe\\OneDrive\\Documents\\Julia\\Data\\SVR_test.json"
testdata = Dict(:X=>X, :y=>y, :epsilon=>ϵ, :C=>C)
open(testfile, "w") do io
    JSON.print(io, testdata)
end
