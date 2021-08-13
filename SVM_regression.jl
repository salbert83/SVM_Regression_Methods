using Clustering # intended for use with large data sets
using KernelFunctions
using LinearAlgebra
using Statistics

function fit(::Type{SVR_ConditionalDensity}, y::AbstractVector{T}, X::AbstractMatrix{T}, kernel, ϵ::T, C::T
        ; method = :surrogate
        , max_points = typemax(Int64)
    ) where {T <: Real}

    # Scale X appropriately
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)
    σ[σ .<= 0] .= one(T)

    X_ = (X .- μ) ./ σ

    m = size(X_, 1)

    centers = if max_points < m # identify cluster centers to represent K
        clusters = Clustering.kmeans(X_', max_points)
        clusters.centers
    else
        nothing
    end

    K = begin
            S = zeros(T, m, min(m, max_points))
            Threads.@threads for i = 1:m
                for j = 1:min(m, max_points)
                    S[i,j] = isnothing(centers) ? kernel(X_[i,:], X_[j,:]) : kernel(X_[i,:], centers[:,j])
                end
            end
            S
        end

    Kt = K'

    apply_kernel!(x, Kx) = mul!(Kx, K, x)

    apply_kernel_transpose!(x, Ktx) = mul!(Ktx, Kt, x)

    weights, bias = if method == :cvx_primal
            calibrate_Primal(y, K, ϵ, fill(C, length(y)))
        elseif method == :cvx_dual
            calibrate_Dual(y, Kt, ϵ, fill(C, length(y)))
        elseif method == :surrogate
            calibrate_surrogate(y, apply_kernel!, apply_kernel_transpose!, min(m, max_points), ϵ, fill(C, length(y))
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
                        , data = isnothing(centers) ? X_ : centers'
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

function predict(mixed::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity, C}, X::AbstractVector{T}) where {C, T<:Real}
     return sum(p * predict(svr, X) for (p, svr) ∈ zip(probs(mixed), components(mixed)))
end

function predict(mixed::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity, C}, X::AbstractMatrix{T}) where {C, T<:Real}
     return [predict(mixed, X[i,:]) for i = 1:size(X,1)]
end
