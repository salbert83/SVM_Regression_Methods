using LinearAlgebra

function surrogate_params(z, ϵ; τ = √eps(ϵ))
    if abs(z) > 2.0ϵ
        a = 0.5 / abs(z)
        c = 0.5abs(z) - ϵ
        return (a = a, t = 0.0, c = c)
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

function calibrate_surrogate(y::AbstractVector{T}, apply_K!, ϵ::T, C::AbstractVector{T}
    ; maxiters = 1000
    , tol = 1.0e-6
    , max_CG_iters = 50
    , CG_tol = 1.0e-5) where {T <: Real}

    m = length(y)
    r = similar(y)
    z = similar(y)
    w = zeros(T, m)
    w_ = zeros(T, m)
    b = zero(T)
    A = similar(y)
    temp_b = similar(y)
    temp_Q = similar(y)
    temp_d = similar(y)
    d = similar(y)
    p = similar(y)
    Qp = similar(y)
    params = fill((a=zero(T), t = zero(T), c = zero(T)), m)

    function calculate_b(w, A, params)
        apply_K!(w, temp_b)
        for i = 1:m
            temp_b[i] = y[i] - temp_b[i] - params[i].t
        end
        return sum(A .* temp_b) / sum(A)
    end

    for iter = 1:maxiters
        apply_K!(w_, r)
        r .= y .- r .- b
        params .= surrogate_params.(r, ϵ, τ = √eps(ϵ))
        for i = 1:m
            A[i] = C[i] * params[i].a
            z[i] = y[i] - params[i].t
        end
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
            apply_K!(temp_Q, Qx)
            # regularization
            Qx .+= x
        end

        temp_d .= A .* z
        temp_d .-= A .* sum(temp_d) ./ sumA
        apply_K!(temp_d, d)

        # Use CG to approximately minimize the surrogate.
        w .= w_
        apply_Q!(w, r)
        r .-= d
        res_norm = norm(r)
        p .= -r
        rdr = dot(r, r)
        for k = 1:max_CG_iters
            apply_Q!(p, Qp)
            pQp = dot(p, Qp)
            α = rdr / pQp
            w .+= α .* p
            r .+= α .* Qp
            rdr_ = dot(r,r)
            if sqrt(rdr_) < CG_tol * res_norm
                break
            else
                β = rdr_/rdr
                p .*= β
                p .-= r
                rdr = rdr_
            end
        end

        b = calculate_b(w, A, params)

        update_diff = norm(w - w_)

        if update_diff < tol * norm(w)
            println("Exiting at iteration $(iter)")
            return (w = w, b = b)
        else
            w_ .= w
        end

    end

    println("Iteration limit exceeded.")
    return (w = w, b = b)
end
