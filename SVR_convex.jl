# SVR calibration relying on Convex.jl

using Convex
# using ECOS
using COSMO

function calibrate_Primal(y::AbstractVector{T}, K::AbstractMatrix{T}, ϵ::T, C::AbstractVector{T}) where {T <: Real}
    m = size(K, 1)
    w = Convex.Variable(m)
    ξ = Convex.Variable(m, Positive())
    b = Convex.Variable()
    e = fill(1.0, m)

    prob = Convex.minimize(0.5dot(w, w) + 0.5sum(C[i]*ξ[i] for i = 1:m))
    prob.constraints += y - K*w - b*e <= ϵ*e + ξ
    prob.constraints += K*w + b*e - y <= ϵ*e + ξ
    solve!(prob, COSMO.Optimizer())
    println("Optimization status: $(prob.status)")
    return (w = evaluate(w), b = evaluate(b))
end

function calibrate_Dual(y::AbstractVector{T}, K::AbstractMatrix{T}, ϵ::T, C::AbstractVector{T}) where {T <: Real}
    m = size(K, 1)
    λ = Convex.Variable(m)
    prob = Convex.maximize(dot(λ, y) - 0.5sumsquares(K*λ) - ϵ * norm(λ,1))
    prob.constraints += -0.5C <= λ
    prob.constraints += 0.5C >= λ
    prob.constraints += sum(λ) == 0.0
    solve!(prob, COSMO.Optimizer())
    println("Optimization status: $(prob.status)")
    λ_val = evaluate(λ)
    w = K * λ_val
    b = zero(T)
    for i = 1:m
        if 0 < abs(λ_val[i]) < 0.5C[i]
            b = y[i] - dot(K[i,:], w) - sign(λ_val[i]) * ϵ
            break
        end
    end
    return (w = w, b = b)
end
