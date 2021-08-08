using KernelFunctions
using LinearAlgebra
using Statistics

function fit(::Type{SVR_ConditionalDensity}, y::AbstractVector{T}, X::AbstractMatrix{T}, kernel, ϵ::T, C::T
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
            calibrate_Primal(y, K, ϵ, fill(C, length(y)))
        elseif method == :cvx_dual
            calibrate_Dual(y, K, ϵ, fill(C, length(y)))
        elseif method == :surrogate
            calibrate_surrogate(y, apply_kernel!, ϵ, fill(C, length(y))
                , maxiters = 1000
                , tol = 1.0e-6
                , CG_tol = 1.0e-5
                , max_CG_iters = 50)
        else
            throw(ArgumentError("Unsupported SVR optimization"))
        end

    return SVR_ConditionalDensity(μ = μ[1,:]
                        , σ = σ[1,:]
                        , w = weights
                        , b = bias
                        , kernel = kernel
                        , data = X_
                        , ϵ = ϵ
                        , C = C)

end

function predict(svr::SVR_ConditionalDensity, X::AbstractVector{T}) where {T <: Real}
    X_ = (X .- svr.μ) ./ svr.σ
    y = svr.b
    for i = 1:size(svr.data,1)
        y +=  svr.kernel(X_, svr.data[i,:]) * svr.w[i]
    end
    return y
end

function predict(svr::SVR_ConditionalDensity, X::AbstractMatrix{T}) where {T <: Real}
    return [predict(svr, X[i,:]) for i = 1:size(X,1)]
end

function cost(svr::SVR_ConditionalDensity, y::AbstractVector{T}, X::AbstractMatrix{T}) where {T <: Real}
    y_pred = predict(svr, X)
    return 0.5dot(svr.w,svr.w) + 0.5sum(svr.C .* ϵ_insensitive_loss.(y - y_pred, svr.ϵ))
end
