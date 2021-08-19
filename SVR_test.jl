push!(LOAD_PATH, @__DIR__)
using SVM_Regression_Methods
using KernelFunctions
using LinearAlgebra
using CUDA

# warm up the solvers
m, n = 100, 20
X = randn(m, n)
y = sin.(sum(X, dims=2))[:,1] .+ log.(sum(exp.(X), dims=2))[:,1] .+ 0.1randn(m)
# y = (sum(X[:,1:(n ÷ 2)], dims=2) .- sum(X[:,((n ÷ 2) + 1):end], dims=2))[:,1]
ϵ = 0.01
C = 10.0

# Kernel
kernel = KernelFunctions.GaussianKernel()

svr1 = fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :cvx_primal)
cst1 = cost(svr1, y, X)
y_pred1 = predict(svr1, X)

svr2 = fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :cvx_dual)
cst2 = cost(svr2, y, X)
y_pred2 = predict(svr2, X)

svr3 = @time fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :surrogate)
cst3 = cost(svr3, y, X)
y_pred3 = predict(svr3, X)

svr3_cu = @time fit(SVR_ConditionalDensity, cu(y), convert(Matrix{Float32}, X), kernel, convert(Float32, ϵ), convert(Float32, C), method = :surrogate)
cst3_cu = cost(svr3_cu, y, X)
y_pred3_cu = predict(svr3_cu, X)

d = to_dict(svr3_cu)
fd  = from_dict(SVR_ConditionalDensity, d)
# benchmark

# The problem
m, n = 1_000, 20
X = randn(m, n)
y = sin.(sum(X, dims=2))[:,1] .+ log.(sum(exp.(X), dims=2))[:,1] .+ 0.1randn(m)
# y = (sum(X[:,1:(n ÷ 2)], dims=2) .- sum(X[:,((n ÷ 2) + 1):end], dims=2))[:,1]
ϵ = 0.01
C = 10.0

# Kernel
kernel = KernelFunctions.GaussianKernel()

# Only include these for tests where m ~ 1000
##=
svr1 = @time fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :cvx_primal)
@show cst1 = cost(svr1, y, X)
y_pred1 = predict(svr1, X)
@show norm(y - y_pred1, 2)/norm(y,2)

svr1_red = @time fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :cvx_primal, max_points = 1000)
@show cst1_red = cost(svr1_red, y, X)
y_pred1_red = predict(svr1_red, X)
@show norm(y - y_pred1_red, 2)/norm(y,2)

svr2 = @time fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :cvx_dual)
@show cst2 = cost(svr2, y, X)
y_pred2 = predict(svr2, X)
@show norm(y - y_pred2, 2)/norm(y,2)
##=#

svr3 = @time fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :surrogate)
@show cst3 = cost(svr3, y, X)
y_pred3 = predict(svr3, X)
@show norm(y - y_pred3, 2)/norm(y,2)

svr3_cu = @time fit(SVR_ConditionalDensity, cu(y), convert(Matrix{Float32}, X), kernel, convert(Float32, ϵ), convert(Float32, C), method = :surrogate)
@show cst3_cu = cost(svr3_cu, y, X)
y_pred3_cu = predict(svr3_cu, X)
@show norm(y - y_pred3_cu, 2)/norm(y,2)

svr3_red = @time fit(SVR_ConditionalDensity, y, X, kernel, ϵ, C, method = :surrogate, max_points = 1000)
@show cst3_red = cost(svr3_red, y, X)
y_pred3_red = predict(svr3_red, X)
@show norm(y - y_pred3_red, 2)/norm(y,2)

svr3_red_cu = @time fit(SVR_ConditionalDensity, cu(y), cu(X), kernel, convert(Float32, ϵ), convert(Float32, C), method = :surrogate, max_points = 1000)
@show cst3_red = cost(svr3_red, y, X)
y_pred3_red = predict(svr3_red, X)
@show norm(y - y_pred3_red, 2)/norm(y,2)


using JSON
testfile = "C:\\Users\\salbe\\OneDrive\\Documents\\Julia\\Data\\SVR_test.json"
testdata = Dict(:X=>X, :y=>y, :epsilon=>ϵ, :C=>C)
open(testfile, "w") do io
    JSON.print(io, testdata)
end
