"""
    bam(T, B, μ_init, Σ_init, λ_function, g)

Batch and match: fit a Gaussian via score matching.

Inputs:
    T: number of iterations to run batch and match
    B: batch size
    μ_init: initial value of mean parameter
    Σ_init: initial value of covariance parameter
    λ_function: a learning rate function of the iteration t
    g(x): function that computes ∇ log P(x)

Returns a mean vector μ and covariance matrix Σ.
"""
function bam(T, B, μ_init, Σ_init, λ_function, g; tol=1e-8)

    μ_iterates = [μ_init*1.0]
    Σ_iterates = [Σ_init*1.0]

    D = length(μ_init)

    μ_t = μ_init
    Σ_t = Σ_init
    λ_t = λ_function(0)

    ## Run iterations
    for t in 1:T

        # Sample x_t ~ q_t = N(mu_t, Sigma_t)
        x_t = rand(MultivariateNormal(μ_t, Σ_t), B)

        # D x B matrix of gradient evaluations of new batch x_t; assume B > 1 for now
        g_t = reduce(hcat, [g(x_t[:, b]) for b in 1:B])

        # Compute statistics of x_t, g_t: assume these are D x B matrices
        gbar = mean(g_t, dims=2)
        xbar = mean(x_t, dims=2)

        # Compute Γ_1, C_1 using x_t
        Γ = (g_t .- gbar) * (g_t .- gbar)' / B
        C = (x_t .- xbar) * (x_t .- xbar)' / B

        # Compute matrices involved in the quadratic matrix equation
        λ_t = λ_function(t)
        U = λ_t * (Γ + gbar * gbar' / (1+λ_t))
        V = Σ_t + λ_t*C + λ_t/(1+λ_t) * (μ_t - xbar) * (μ_t - xbar)'

        # Compute mu_1, Sigma_1 (i.e., update variational parameters)
        Σ_tt = 2 * V / (I + sqrt(I + 4*U*V)); Σ_tt = (Σ_tt + Σ_tt') ./ 2 .+ 1e-6
        μ_tt = 1/(1+λ_t) * μ_t + λ_t/(1+λ_t) * (Σ_tt * gbar + xbar); μ_tt = vec(μ_tt)

        # update parameters
        μ_t = μ_tt; Σ_t = Σ_tt
        push!(μ_iterates, μ_t); push!(Σ_iterates, Σ_t)

    end

    return μ_iterates, Σ_iterates
end

