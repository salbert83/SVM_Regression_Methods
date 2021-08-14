module SVM_Regression_Methods

include("SVR_common.jl")
include("SVR_surrogate.jl")
include("SVR_convex.jl")
include("SVM_regression.jl")
include("SVR_mixture_models.jl")

export SVR_ConditionalDensity, to_dict, from_dict
export fit, predict, cost, fit_mixture
export pdf, logpdf
export posterior_probs

end
