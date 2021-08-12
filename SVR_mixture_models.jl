using Clustering
using Distributed
using Distributions
using KernelFunctions
using LinearAlgebra
using SharedArrays
using Statistics

import Distributions: pdf, logpdf

function fit_mixture(y::AbstractVector{T}, X::AbstractMatrix{T}, ϵ::T, C::T, k, kernel
    ; method = :surrogate
    , max_points = typemax(Int64)
    , tol = 1.0e-4
    , max_iters = 100) where {T <: Real}

    m, n = size(X)

    YX = hcat(y, X)

    μ = mean(YX, dims = 1)
    σ = std(YX, dims = 1)
    σ[σ .== zero(T)] .= one(T)
    YX_ = (YX .- μ) ./ σ

    centers = if max_points < m
        clusters = Clustering.kmeans(YX_[:,2:end]', max_points)
        clusters.centers
    else
        nothing
    end

    K = begin
            M = zeros(T, m, min(m, max_points))
            Threads.@threads for i = 1:m
                for j = 1:min(i, max_points)
                    M[i,j] = isnothing(centers) ? kernel(YX_[i,2:end], YX_[j,2:end]) : kernel(YX_[i,2:end], centers[:,j])
                end
            end
            M
        end

    Kt = K'

    apply_kernel!(x, Kx) = mul!(Kx, K, x)

    apply_kernel_transpose!(x, Ktx) = mul!(Ktx, Kt, x)

    # Use K-means for initial clusters
    clusters = Clustering.kmeans(YX_', k)
    wgts = [((clusters.assignments[i] .== j) ? 1.0 : 0.0) for i = 1:m, j = 1:k]
    models = Vector{SVR_ConditionalDensity}(undef, k)

    θ =  SharedArray(zeros(T, min(m, max_points), k))
    b = SharedArray(zeros(T, k))
    component_probs = zeros(T, k)
    offset = zeros(T, m, 1)
    ξ = zeros(T, m)
    loglikelihood = -Inf64
    loglikelihood_old = -Inf64

    for iter = 1:max_iters
        component_probs .= (sum(wgts, dims=1) ./ m)[1,:]
        @sync @distributed for model_idx = 1:k
            weights, bias = if method == :cvx_primal
                    calibrate_Primal(y, K, ϵ, C * wgts[:, model_idx])
                elseif method == :cvx_dual
                    calibrate_Dual(y, Kt, ϵ, C * wgts[:, model_idx])
                elseif method == :surrogate
                    calibrate_surrogate(y, apply_kernel!, apply_kernel_transpose!, min(m, max_points), ϵ, C * wgts[:, model_idx]
                        , maxiters = 1000
                        , tol = 1.0e-6
                        , CG_tol = 1.0e-5
                        , max_CG_iters = 50
                        , w_init = θ[:, model_idx]
                        , b_init = b[model_idx])
                else
                    throw(ArgumentError("Unsupported SVR optimization"))
                end
            θ[:, model_idx] .= weights
            b[model_idx] = bias
        end
        # update mixture weights
        for model_idx = 1:k
            mul!(ξ, K, -θ[:, model_idx])
            ξ .+= y .- b[model_idx]
            for i = 1:m
                wgts[i, model_idx] = log(component_probs[model_idx]) + log_cond_prob(ξ[i], ϵ, C)
            end
        end
        offset .= maximum(wgts, dims = 2)
        wgts .-= offset
        wgts .= exp.(wgts)
        loglikelihood =  sum(offset .+ log.(sum(wgts, dims = 2)))
        wgts ./= sum(wgts, dims = 2)

        println("iter $(iter): loglikelihood = $(loglikelihood)")
        if loglikelihood - loglikelihood_old < tol * abs(loglikelihood)
            break
        else
            loglikelihood_old = loglikelihood
        end
    end
    for model_idx = 1:k
        models[model_idx] = SVR_ConditionalDensity(
            μ = μ[1,2:end]
            , σ = σ[1,2:end]
            , w = θ[:, model_idx]
            , b = b[model_idx]
            , kernel = kernel
            , data = isnothing(centers) ? YX_[:,2:end] : centers'
            , ϵ = ϵ
            , C = C
            )
    end
    return MixtureModel(models, component_probs)
end
