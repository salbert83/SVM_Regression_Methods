# SVR calibration relying on Convex.jl

using Convex
using COSMO

function calibrate_Primal(y::AbstractVector{T}, K::AbstractMatrix{T}, ϵ::T, C::AbstractVector{T}) where {T <: Real}
    m, n = size(K)
    w = Convex.Variable(n)
    ξ = Convex.Variable(m, Positive())
    b = Convex.Variable()
    e = fill(1.0, m)

    problem = Convex.minimize(0.5dot(w, w) + 0.5sum(C[i]*ξ[i] for i = 1:m))
    problem.constraints += y - K*w - b*e <= ϵ*e + ξ
    problem.constraints += K*w + b*e - y <= ϵ*e + ξ
    solve!(problem, COSMO.Optimizer())
    println("Optimization status: $(problem.status)")
    return (w = evaluate(w), b = evaluate(b))
end

function calibrate_Dual(y::AbstractVector{T}, Kt::AbstractMatrix{T}, ϵ::T, C::AbstractVector{T}) where {T <: Real}
    n, m = size(Kt)
    λ = Convex.Variable(m)
    problem = Convex.maximize(dot(λ, y) - 0.5sumsquares(Kt*λ) - ϵ * norm(λ,1))
    problem.constraints += -0.5C <= λ
    problem.constraints += 0.5C >= λ
    problem.constraints += sum(λ) == 0.0
    solve!(problem, COSMO.Optimizer())
    println("Optimization status: $(problem.status)")
    λ_val = evaluate(λ)
    w = Kt * λ_val
    b = zero(T)
    for i = 1:m
        tol = eps(C[i])^(1/3)
        if tol < abs(λ_val[i]) < 0.5C[i] - tol
            b = y[i] - dot(Kt[:,i], w) - sign(λ_val[i]) * ϵ
            break
        end
    end
    return (w = w, b = b)
end
