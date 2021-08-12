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
    return Dict(:μ => svr.μ
        , :σ => svr.σ
        , :w => svr.w
        , :b => svr.b
        , :kernel => string(svr.kernel)
        , :data => svr.data
        , :ϵ => svr.ϵ
        , :C => svr.C)
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

function logpdf(mix::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity, C}, y, x) where C
    logprobs = [logpdf(d, y, x) for d ∈ components(mix)] .+ log.(probs(mix))
    m = maximum(logprobs)
    return m + log(sum(exp.(logprobs .- m)))
end

pdf(mix::MixtureModel{Univariate, Continuous, SVR_ConditionalDensity, C}, y, x) where C = exp(logpdf(mix, y, x))
