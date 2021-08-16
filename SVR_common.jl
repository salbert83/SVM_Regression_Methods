using Distributions
using KernelFunctions
using JSON
using Parameters

import Distributions: pdf, logpdf

@with_kw struct SVR_ConditionalDensity{T <: Real} <: ContinuousUnivariateDistribution
    # Required to normalize the input
    μ::AbstractVector{T}
    σ::AbstractVector{T}

    # SVR parameters
    w::AbstractVector{T}
    b::T
    kernel::Kernel
    data::AbstractMatrix{T}

    # Additional params for density
    ϵ::T
    C::T
end

function to_dict(svr::SVR_ConditionalDensity)
    return Dict("μ" => svr.μ
        , "σ" => svr.σ
        , "w" => svr.w
        , "b" => svr.b
        , "kernel" => string(svr.kernel)
        , "data" => svr.data
        , "ϵ" => svr.ϵ
        , "C" => svr.C)
end

function from_dict(::Type{S}, data
    ; expected_kernel = nothing) where S<:SVR_ConditionalDensity

    # This part should be revisited when I have more time
    kernel_lookup = Dict("Squared Exponential Kernel (metric = Distances.Euclidean(0.0))" => GaussianKernel())

    return SVR_ConditionalDensity(
        μ = convert(Vector{Float64}, data["μ"])
        , σ = convert(Vector{Float64}, data["σ"])
        , w = convert(Vector{Float64}, data["w"])
        , b = convert(Float64, data["b"])
        , kernel = if isnothing(expected_kernel)
                    kernel_lookup[data["kernel"]]
                else
                    s = string(expected_kernel)
                    s == data["kernel"] || @warn "Expected kernel doesn't match saved kernel"
                    expected_kernel
                end
        , data = if typeof(data["data"]) <: Vector
                    hcat([convert(Vector{Float64}, c) for c ∈ data["data"]]...) # matrices stored in column major form
                else
                    convert(Matrix{Float64}, data["data"])
                end
        , ϵ = convert(Float64, data["ϵ"])
        , C = convert(Float64, data["C"])
    )
end

function to_dict(mix::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity})
    return [Dict("prob"=>p, "component"=>to_dict(c)) for (p,c) ∈ zip(probs(mix), components(mix))]
end

function from_dict(::Type{S}, data
    ; expected_kernel=nothing) where {S<:MixtureModel{Univariate, Continuous, SVR_ConditionalDensity}}
    return MixtureModel([from_dict(SVR_ConditionalDensity, d["component"], expected_kernel=expected_kernel) for d ∈ data], [d["prob"] for d ∈ data])
end

ϵ_insensitive_loss(x, ϵ) = max(abs(x) - ϵ, 0.0)

# pdf consistent with SVR penalty C/2 |ξ|_ϵ
log_cond_prob(ξ, ϵ, C) = log(C / (2.0 * (2.0 + ϵ * C))) - ϵ_insensitive_loss(ξ, ϵ)

function logpdf(d::SVR_ConditionalDensity, y, x)
    x_ = (x .- d.μ) ./ d.σ
    ξ = d.b
    for j = 1:length(d.w)
        ξ += d.kernel(x_, d.data[j,:]) * d.w[j]
    end
    return log_cond_prob(ξ, d.ϵ, d.C)
end

pdf(d::SVR_ConditionalDensity, y, x) = exp(logpdf(d, y, x))

function logpdf(mix::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity}, y, x)
    logprobs = [logpdf(d, y, x) for d ∈ components(mix)] .+ log.(probs(mix))
    m = maximum(logprobs)
    return m + log(sum(exp.(logprobs .- m)))
end

pdf(mix::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity}, y, x) = exp(logpdf(mix, y, x))

function posterior_probs(mix::MixtureModel{Univariate, Continuous, C}, y, x) where {C <: SVR_ConditionalDensity}
    logprobs = [logpdf(d, y, x) for d ∈ components(mix)] .+ log.(probs(mix))
    m = maximum(logprobs)
    w = exp.(logprobs .- m)
    return w ./ sum(w)
end
