using Clustering
using Distributions
using KernelFunctions
using LinearAlgebra
using Statistics

import Distributions: pdf, logpdf

function logpdf(d::SVR_ConditionalDensity, y, x)
    x_ = (x .- μ) ./ σ
    ξ = d.b
    for j = 1:length(w)
        ξ += d.kernel(x_, d.data[j,:]) * w[j]
    end
    return log(d.C / (2.0 * (2.0 + d.ϵ * d.C))) - ϵ_insensitive_loss(ξ, d.ϵ)
end

pdf(d::SVR_ConditionalDensity, y, x) = exp(logpdf(d, y, x))

# pdf consistent with SVR penalty C/2 |ξ|_ϵ
log_cond_prob(ξ, ϵ, C) = log(C / (2.0 * (2.0 + ϵ * C))) - ϵ_insensitive_loss(ξ, ϵ)

function fit_mixture(y::AbstractVector{T}, X::AbstractMatrix{T}, ϵ::T, C::T, k, kernel
    ; method = :surrogate
    , tol = 1.0e-4
    , max_iters = 100) where {T <: Real}

    m, n = size(X)

    YX = hcat(y, X)

    μ = mean(YX, dims = 1)
    σ = std(YX, dims = 1)
    σ[σ .== zero(T)] .= one(T)

    YX_ = (YX .- μ) ./ σ

    K = begin
            M = zeros(T, m, m)
            for i = 1:m
                for j = 1:i
                    M[i,j] = kernel(YX_[i,2:end], YX_[j,2:end])
                end
            end
            Symmetric(M,:L)
        end

    apply_kernel!(x, Kx) = mul!(Kx, K, x)

    # Use K-means for initial clusters
    clusters = Clustering.kmeans(YX_', k)
    wgts = [((clusters.assignments[i] .== j) ? 1.0 : 0.0) for i = 1:m, j = 1:k]
    wgts_old = copy(wgts)
    models = Vector{SVR_ConditionalDensity}(undef, k)

    θ = zeros(T, m, k)
    b = zeros(T, k)
    p = zeros(T, k)

    for iter = 1:max_iters
        p .= (sum(wgts, dims=1) ./ m)[1,:]
        for model_idx = 1:k
            weights, bias = if method == :cvx_primal
                    calibrate_Primal(y, K, ϵ, C * wgts[:, model_idx])
                elseif method == :cvx_dual
                    calibrate_Dual(y, K, ϵ, C * wgts[:, model_idx])
                elseif method == :surrogate
                    calibrate_surrogate(y, apply_kernel!, ϵ, C * wgts[:, model_idx]
                        , maxiters = 1000
                        , tol = 1.0e-6
                        , CG_tol = 1.0e-5
                        , max_CG_iters = 50
                        , w_init = θ[:, model_idx])
                else
                    throw(ArgumentError("Unsupported SVR optimization"))
                end
            θ[:, model_idx] .= weights
            b[model_idx] = bias
        end
        # update mixture weights
        for model_idx = 1:k
            ξ = y .- K*θ[:, model_idx] .- b[model_idx]
            for i = 1:m
                wgts[i, model_idx] = log(p[model_idx]) + log_cond_prob(ξ[i], ϵ, C)
            end
        end
        wgts .-= maximum(wgts, dims = 2)
        wgts .= exp.(wgts)
        wgts ./= sum(wgts, dims = 2)
        norm_diff = norm(wgts - wgts_old, 1)
        println("iter $(iter): norm diff = $(norm_diff)")
        if norm_diff < tol * m
            break
        else
            wgts_old .= wgts
        end
    end
    for model_idx = 1:k
        models[model_idx] = SVR_ConditionalDensity(
            μ = μ[1,2:end]
            , σ = σ[1,2:end]
            , w = θ[:, model_idx]
            , b = b[model_idx]
            , kernel = kernel
            , data = YX_[:,2:end]
            , ϵ = ϵ
            , C = C
            )
    end
    return MixtureModel(models, p)
end
