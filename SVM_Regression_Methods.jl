module SVM_Regression_Methods

include("SVR_common.jl")
include("SVR_surrogate.jl")
include("SVR_convex.jl")
include("SVM_regression.jl")
include("SVR_mixture_models.jl")

export SVR_ConditionalDensity, to_dict, from_dict
export fit, predict, cost
export pdf, logpdf, fit_mixture

end
