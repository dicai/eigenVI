using Distributions
using ForwardDiff
using PyPlot

# Gaussian mixture models
"""
2-component multivariate GMM.
"""
function get_gmm2(mu1, mu2, cov1, cov2, pi1)
    D = length(mu1)
    if D > 1
        distn = MixtureModel(MultivariateNormal[
           MultivariateNormal(mu1, cov1),
           MultivariateNormal(mu2, cov2)],
                [pi1, 1-pi1])
    else
        distn = MixtureModel(Normal[
           Normal(mu1, cov1),
           Normal(mu2, cov2)],
                [pi1, 1-pi1])
    end

    P(x) = pdf(distn, x)
    logP(x) = logpdf(distn, x)
    dlogP(x) = ForwardDiff.gradient(logP, x)

    return P, logP, dlogP
end

function sample_gmm2(M, mu1, mu2, cov1, cov2, pi1)
    D = length(mu1)
    if D > 1
        distn = MixtureModel(MultivariateNormal[
           MultivariateNormal(mu1, cov1),
           MultivariateNormal(mu2, cov2)],
                [pi1, 1-pi1])
    else
        distn = MixtureModel(Normal[
           Normal(mu1, cov1),
           Normal(mu2, cov2)],
                [pi1, 1-pi1])
    end

    P(x) = pdf(distn, x)
    logP(x) = logpdf(distn, x)
    dlogP(x) = ForwardDiff.gradient(logP, x)

    return rand(distn, M)

end

"""
3-component multivariate GMM.
"""
function get_gmm3(mu1, mu2, mu3, cov1, cov2, cov3, pi1, pi2)
    if D > 1
        distn = MixtureModel(MultivariateNormal[
           MultivariateNormal(mu1, cov1),
           MultivariateNormal(mu2, cov2),
           MultivariateNormal(mu3, cov3)],
                [pi1, pi2, 1-(pi1+pi2)])
    else
        distn = MixtureModel(Normal[
           Normal(mu1, cov1),
           Normal(mu2, cov2),
           Normal(mu3, cov3)],
                [pi1, pi2, 1-(pi1+pi2)])
    end
    P(x) = pdf(distn, x)
    logP(x) = logpdf(distn, x)
    # score function
    dlogP(x) = ForwardDiff.gradient(logP, x)
    return P, logP, dlogP
end

function sample_gmm3(M, mu1, mu2, mu3, cov1, cov2, cov3, pi1, pi2)
    if D > 1
        distn = MixtureModel(MultivariateNormal[
           MultivariateNormal(mu1, cov1),
           MultivariateNormal(mu2, cov2),
           MultivariateNormal(mu3, cov3)],
                [pi1, pi2, 1-(pi1+pi2)])
    else
        distn = MixtureModel(Normal[
           Normal(mu1, cov1),
           Normal(mu2, cov2),
           Normal(mu3, cov3)],
                [pi1, pi2, 1-(pi1+pi2)])
    end

    return rand(distn, M)
end


"""
4-component multivariate GMM.
"""
function get_gmm4(mu1, mu2, mu3, mu4, cov1, cov2, cov3, cov4, pi1, pi2, pi3)
    if D > 1
        distn = MixtureModel(MultivariateNormal[
           MultivariateNormal(mu1, cov1),
           MultivariateNormal(mu2, cov2),
           MultivariateNormal(mu3, cov3),
           MultivariateNormal(mu4, cov4)],
                [pi1, pi2, pi3, 1-(pi1+pi2+pi3)])
    else
        distn = MixtureModel(Normal[
           Normal(mu1, cov1),
           Normal(mu2, cov2),
           Normal(mu3, cov3),
           Normal(mu4, cov4)],
                [pi1, pi2, pi3, 1-(pi1+pi2+pi3)])
    end
    P(x) = pdf(distn, x)
    logP(x) = logpdf(distn, x)
    # score function
    dlogP(x) = ForwardDiff.gradient(logP, x)
    return P, logP, dlogP
end

function sample_gmm4(N, mu1, mu2, mu3, mu4, cov1, cov2, cov3, cov4, pi1, pi2, pi3)
    if D > 1
        distn = MixtureModel(MultivariateNormal[
           MultivariateNormal(mu1, cov1),
           MultivariateNormal(mu2, cov2),
           MultivariateNormal(mu3, cov3),
           MultivariateNormal(mu4, cov4)],
                [pi1, pi2, pi3, 1-(pi1+pi2+pi3)])
    else
        distn = MixtureModel(Normal[
           Normal(mu1, cov1),
           Normal(mu2, cov2),
           Normal(mu3, cov3),
           Normal(mu4, cov4)],
                [pi1, pi2, pi3, 1-(pi1+pi2+pi3)])
    end
    return rand(distn, N)
end

# Rosenbrock
function get_2D_rosenbrock(a, b)
    #a = 1.; b = 1.

    # Note P here is unnormalized
    P(x) = exp(-(a * (x[2]-x[1]^2)^2+(1-x[1])^2)/b)
    # TODO: normalize P

    logP(x) = -(a * (x[2]-x[1]^2)^2+(1-x[1])^2)/b
    # score function
    dlogP(x) = ForwardDiff.gradient(logP, x)

    return P, logP, dlogP
end

# sinh-arcsinh normal distribution
"""
Generates a centered sinh-arcsinh distribution.
"""
function get_sinh_arcsinh(s, τ, Σ)
    # Assume z, s, τ are D-dimensional and broadcast
    S(z, s, τ) = sinh.((asinh.(z) .+ s) .* τ)
    C(zi, si, τi) = sqrt.(1 .+ S(zi, si, τi).^2) # this is a scalar

    log_pdf_f(z, s, τ, R) = -0.5*log((2π)^length(z) * det(R)) + sum(log.(τ .* C(z, s, τ)) - 0.5 * log.(1 .+ z .^ 2)) - 0.5 * (S(z, s, τ)' * (R \ S(z, s, τ)))
    pdf_f(z, s, τ, R) = exp(log_pdf_f(z, s, τ, R))

    logP(x) = log_pdf_f(x, s*ones(D), τ*ones(D), Σ)
    # score function
    dlogP(x) = ForwardDiff.gradient(logP, x)

    return P, logP, dlogP
end

function get_sinh_arcsinh2(s, τ, Σ)
    S(x, s, τ) = sinh.(τ .* asinh.(x) .- s)

    C(zi, si, τi) = sqrt.(1 .+ S(zi, si, τi).^2) # this is a scalar

    log_pdf_f(z, s, τ, R) = -0.5*log((2π)^length(z) * det(R)) + sum(log.(τ .* C(z, s, τ)) - 0.5 * log.(1 .+ z .^ 2)) - 0.5 * (S(z, s, τ)' * (R \ S(z, s, τ)))
    pdf_f(z, s, τ, R) = exp(log_pdf_f(z, s, τ, R))

    logP(x) = log_pdf_f(x, s, τ, Σ)
    P(x) = exp(logP(x))
    # score function
    dlogP(x) = ForwardDiff.gradient(logP, x)

    return P, logP, dlogP
end


"""
Generates funnel in D >= 2 dimensions.
"""
function get_funnel(σ2, D)
    logP(x) = logpdf(Normal(0, σ2), x[1]) + sum([logpdf(Normal(0, exp(x[1]/2)), x[d]) for d in 2:D]) + 1e-6
    P(x) = exp(logP(x))
    dlogP(x) = ForwardDiff.gradient(logP, x)
    return P, logP, dlogP
end
