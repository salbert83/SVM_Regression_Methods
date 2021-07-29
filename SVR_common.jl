using Distributions
using KernelFunctions
using Parameters

@with_kw struct SVM_Regression{T <: Real}
    # Required to normalize the input
    μ::AbstractVector{T}
    σ::AbstractVector{T}

    # SVR parameters
    w::AbstractVector{T}
    b::T
    kernel::Kernel
    data::AbstractMatrix{T}
end

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

ϵ_insensitive_loss(x, ϵ) = max(abs(x) - ϵ, 0.0)
