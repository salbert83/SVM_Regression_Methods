using LinearAlgebra

function surrogate_params(z, ϵ; τ = √eps(ϵ))
    if abs(z) > 2.0ϵ
        a = 0.5 / abs(z)
        c = 0.5abs(z) - ϵ
        return (a = a, t = zero(z), c = c)
    else
        a = 1.0 / (4.0abs(abs(z) - ϵ) + τ)
        t = if abs(z) > ϵ
                ((z > ϵ) ? 1.0 : -1.0) * (2.0ϵ - abs(z))
            else
                z
            end
        return (a = a, t = t, c = 0.0)
    end
end

function surrogate(x₀, ϵ; τ = √eps(ϵ))
    a, t, c = surrogate_params(x₀, ϵ, τ = τ)
    f(x) = a*(x - t)^2 + c
    return  f
end

function surrogate_a(z, ϵ; τ = √eps(ϵ))
    if abs(z) > 2.0ϵ
        return 0.5 / abs(z)
    else
        return one(z) / (4.0abs(abs(z) - ϵ) + τ)
    end
end

function surrogate_t(z, ϵ; τ = √eps(ϵ))
    if abs(z) > 2.0ϵ
        return zero(z)
    else
        if abs(z) > ϵ
            return ((z > ϵ) ? 1.0 : -1.0) * (2.0ϵ - abs(z))
        else
            return z
        end
    end
end

function CG!(w, apply_Q!, r_CG, d, p, Qp, CG_tol, CG_iters)
    apply_Q!(w, r_CG)
    r_CG .-= d
    rdr = dot(r_CG, r_CG)
    res_norm = sqrt(rdr)
    p .= -r_CG
    if rdr > √eps(rdr)
        for k = 1:CG_iters
            apply_Q!(p, Qp)
            pQp = dot(p, Qp)
            α = rdr / pQp
            w .+= α .* p
            r_CG .+= α .* Qp
            rdr_ = dot(r_CG,r_CG)
            if sqrt(rdr_) < CG_tol * res_norm
                break
            else
                β = rdr_/rdr
                p .*= β
                p .-= r_CG
                rdr = rdr_
            end
        end
    end
end

function calibrate_surrogate(y::AbstractVector{T}, apply_K!, apply_Kt!, n, ϵ::T, C::AbstractVector{T}
    ; maxiters = 1000
    , tol = 1.0e-6
    , min_CG_iters = 20
    , max_CG_iters = 50
    , CG_tol = 1.0e-5
    , w_init = nothing
    , b_init = nothing) where {T <: Real}

    m = length(y)
    r = similar(y)
    r_CG = similar(y, n)
    z = similar(y)
    w = if isnothing(w_init)    # If not initialized, start at 0
            w_init = similar(y, n)
            w_init .= zero(T)
            w_init
        else
            copy(w_init)
        end
    b = isnothing(b_init) ? zero(T) : b_init
    A = similar(y)
    temp_b = similar(y)
    temp_Q = similar(y)
    temp_d = similar(y)
    d = similar(y, n)
    p = similar(y, n)
    Qp = similar(y, n)
    params_a = similar(y, m)
    params_t = similar(y, m)

    function calculate_b(w, A, t)
        apply_K!(w, temp_b)
        temp_b .= y .- temp_b .- t
        return sum(A .* temp_b) / sum(A)
    end

    svr_cost = Inf64
    svr_cost_old = Inf64

    for iter = 1:maxiters
        apply_K!(w, r)
        r .= y .- r .- b
        svr_cost = 0.5dot(w,w) + 0.5sum(C .* ϵ_insensitive_loss.(r, ϵ))
        # println("svr_cost = $(svr_cost)")
        if svr_cost_old - svr_cost < max(tol, tol*svr_cost)
            println("Exiting at iteration $(iter)")
            return (w = w, b = b)
        else
            svr_cost_old = svr_cost
        end

        params_a .= surrogate_a.(r, ϵ)
        params_t .= surrogate_t.(r, ϵ)
        A .= C .* params_a
        z .= y .- params_t
        sumA = sum(A)
        z .-= sum(A .* z) / sumA

        function apply_Q!(x, Qx)
            # apply M
            apply_K!(x, temp_Q)
            temp_Q .-= sum(A .* temp_Q) ./ sumA
            # apply A
            temp_Q .*= A
            # apply M^t
            temp_Q .-= A .* sum(temp_Q) ./ sumA
            apply_Kt!(temp_Q, Qx)
            # regularization
            Qx .+= x
        end

        temp_d .= A .* z
        temp_d .-= A .* sum(temp_d) ./ sumA
        apply_Kt!(temp_d, d)

        CG!(w, apply_Q!, r_CG, d, p, Qp, CG_tol, min(min_CG_iters + iter, max_CG_iters))

        b = calculate_b(w, A, params_t)

    end

    println("Iteration limit exceeded.")
    return (w = w, b = b)
end
