using Clustering
using CUDA
using Distributed
using Distributions
using KernelFunctions
using LinearAlgebra
using SharedArrays
using Statistics

function fit_mixture(y::AbstractVector{T}, X::AbstractMatrix{T}, ϵ::T, C::T, k, kernel
    ; method = :surrogate
    , max_points = typemax(Int64)
    , tol = 1.0e-4
    , max_iters = 100) where {T <: Real}

    m, n = size(X)

    YX = hcat(convert(Vector{T}, y), convert(Matrix{T}, X))

    μ = mean(YX, dims = 1)
    σ = std(YX, dims = 1)
    σ[σ .<= zero(T)] .= one(T)
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
                for j = 1:min(m, max_points)
                    M[i,j] = isnothing(centers) ? kernel(YX_[i,2:end], YX_[j,2:end]) : kernel(YX_[i,2:end], centers[:,j])
                end
            end
            (typeof(y) <: CuArray) ?  CuArray{T}(M) : M
        end

    apply_kernel!(x, Kx) = mul!(Kx, K, x)

    apply_kernel_transpose!(x, Ktx) = mul!(Ktx, K', x)

    mK = convert(Matrix{T}, K)
    vy = convert(Vector{T}, y)

    # Use K-means for initial clusters
    clusters = Clustering.kmeans(YX_', k)
    wgts = SharedArray([((clusters.assignments[i] .== j) ? one(T) : zero(T)) for i = 1:m, j = 1:k])
    models = Vector{SVR_ConditionalDensity}(undef, k)

    θ = SharedArray(zeros(T, min(m, max_points), k))
    b = SharedArray(zeros(T, k))
    component_probs = SharedArray(zeros(T, k))
    offset = zeros(T, m, 1)
    ξ = SharedArray(zeros(T, m, k))
    loglikelihood = -Inf
    loglikelihood_old = -Inf

    for iter = 1:max_iters
        component_probs .= (sum(wgts, dims=1) ./ m)[1,:]
        @show component_probs
        @sync @distributed for model_idx = 1:k
            weights, bias = if method == :cvx_primal
                    calibrate_Primal(y, K, ϵ, C * wgts[:, model_idx])
                elseif method == :cvx_dual
                    calibrate_Dual(y, Kt, ϵ, C * wgts[:, model_idx])
                elseif method == :surrogate
                    Cvec = convert(typeof(y), C * wgts[:, model_idx])
                    calibrate_surrogate(y, apply_kernel!, apply_kernel_transpose!, min(m, max_points), ϵ, Cvec
                        , maxiters = 1000
                        , tol = 1.0e-6
                        , CG_tol = 1.0e-5
                        , max_CG_iters = 50
                        , w_init = convert(typeof(y), θ[:, model_idx])
                        , b_init = b[model_idx])
                else
                    throw(ArgumentError("Unsupported SVR optimization"))
                end
            θ[:, model_idx] .= convert(Vector{T}, weights)
            b[model_idx] = bias
            mul!(view(ξ, :, model_idx), mK, θ[:, model_idx])
            ξ[:, model_idx] .= vy .- ξ[:, model_idx] .- b[model_idx]
            for i = 1:m
                wgts[i, model_idx] = log(component_probs[model_idx]) + log_cond_prob(ξ[i, model_idx], ϵ, C)
            end
        end

        # update mixture weights
        offset .= maximum(wgts, dims = 2)
        wgts .-= offset
        wgts .= exp.(wgts)
        loglikelihood = sum(offset .+ log.(sum(wgts, dims = 2)))
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
            μ = convert(Vector{T}, μ[1,2:end])
            , σ = convert(Vector{T}, σ[1,2:end])
            , w = θ[:, model_idx]
            , b = b[model_idx]
            , kernel = kernel
            , data = isnothing(centers) ? YX_[:,2:end] : centers'
            , ϵ = ϵ
            , C = C
            )
    end

    # Reduce roundoff error
    component_probs ./= sum(component_probs)
    @show component_probs
    return MixtureModel(models, component_probs)
end
