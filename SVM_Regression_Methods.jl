module SVM_Regression_Methods

include("SVR_surrogate.jl")
include("SVR_convex.jl")

export SVM_Regression, fit, predict, cost

using KernelFunctions
using LinearAlgebra
using Parameters
using Statistics

@with_kw struct SVM_Regression{T <: Real}
    μ::AbstractVector{T}
    σ::AbstractVector{T}
    w::AbstractVector{T}
    b::T
    kernel::Kernel
    data::AbstractMatrix{T}
end

loss(x, ϵ) = max(abs(x) - ϵ, 0.0)

function fit(::Type{SVM_Regression}, y::AbstractVector{T}, X::AbstractMatrix{T}, kernel, ϵ::T, C::AbstractVector{T}
        ; method = :surrogate
        , assemble_kernel = true
    ) where {T <: Real}

    # Scale X appropriately
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)
    σ[σ .<= 0] .= one(T)

    X_ = (X .- μ) ./ σ

    m = size(X_, 1)
    K = if assemble_kernel || method ∈ [:cvx_primal, :cvx_dual]
            S = zeros(T, m, m)
            for i = 1:m
                for j = 1:i
                    S[i,j] = kernel(X_[i,:], X_[j,:])
                end
            end
            Symmetric(S, :L)
        else
            nothing
        end

    function apply_kernel!(x, Kx)
        if isnothing(K)
            Threads.@threads for i = 1:m
                Kx[i] = zero(T)
                for j = 1:m
                    Kx[i] += kernel(X_[i,:], X_[j,:])*x[j]
                end
            end
        else
            mul!(Kx, K, x)
        end
    end

    weights, bias = if method == :cvx_primal
            calibrate_Primal(y, K, ϵ, C)
        elseif method == :cvx_dual
            calibrate_Dual(y, K, ϵ, C)
        elseif method == :surrogate
            calibrate_surrogate(y, apply_kernel!, ϵ, C
                , maxiters = 1000
                , tol = 1.0e-6
                , CG_tol = 1.0e-5
                , max_CG_iters = 50)
        else
            throw(ArgumentError("Unsupported SVR optimization"))
        end

    return SVM_Regression(μ = μ[1,:]
                        , σ = σ[1,:]
                        , w = weights
                        , b = bias
                        , kernel = kernel
                        , data = X_)

end

function predict(svr::SVM_Regression, X::AbstractVector{T}) where {T <: Real}
    X_ = (X .- svr.μ) ./ svr.σ
    y = svr.b
    for i = 1:size(svr.data,1)
        y +=  svr.kernel(X_, svr.data[i,:]) * svr.w[i]
    end
    return y
end

function predict(svr::SVM_Regression, X::AbstractMatrix{T}) where {T <: Real}
    return [predict(svr, X[i,:]) for i = 1:size(X,1)]
end

function cost(svr::SVM_Regression, y::AbstractVector{T}, X::AbstractMatrix{T}, ϵ::T, C::AbstractVector{T}) where {T <: Real}
    y_pred = predict(svr, X)
    return 0.5dot(svr.w,svr.w) + 0.5sum(C .* loss.(y - y_pred, ϵ))
end

end
