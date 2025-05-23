using LinearAlgebra
using Arpack


include("bam.jl")
include("polynomials.jl")


function solve_eigenvalue_problem(A; B=nothing, arpack=false)
    # if C++ is installed, we can use ARPACK
    if arpack
        if B != nothing
            obj, α = Arpack.eigs(A, B; nev=1, which=:SM, maxiter=800, tol=1e-6)
            α /= norm(α) # rescale
        else
            obj, α = Arpack.eigs(A; nev=1, which=:SM)
        end
        return vec(α), A, obj[1]
    else
        # Compute eigenvalues and eigenvectors
        if B != nothing
            evals, evecs = eigen(A, B)
        # Most basic usage:
        else
            evals, evecs = eigen(A)
        end
        min_ind = argmin(real(evals))
        α = @view(evecs[:, min_ind])
        α /= norm(α) # rescale

        return α, A, evals[min_ind]
    end
end

function eigenVI_2D(K, X, dlogP, basis_fn, d_basis_fn; grads=false, denom=false, arpack=false, logPi=nothing)
    D, N = size(X)
    @assert D == 2
    W = zeros(K^D, K^D)

    if denom
        V = zeros(K^D, K^D)
    end

    for n in 1:N
        x = @view(X[:, n])
        # importance weight; by default is uniform
        if logPi != nothing
           logweight = -logPi(x)
        else
           logweight = -log(N)
        end
        for d in 1:D
            w_nd = zeros(K, K)
            phi = zeros(K, K)
            for k1 in 1:K, k2 in 1:K
                # assumes D == 2
                if d == 1
                    if grads
                        w_ndk = 2 * d_basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1) - dlogP[d,n] * basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1)
                    else
                        w_ndk = 2 * d_basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1) - dlogP(x)[d] * basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1)
                    end
                else
                    if grads
                        w_ndk = 2 * basis_fn(x[1], k1-1) * d_basis_fn(x[2], k2-1) - dlogP[d,n] * basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1)
                    else
                        w_ndk = 2 * basis_fn(x[1], k1-1) * d_basis_fn(x[2], k2-1) - dlogP(x)[d] * basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1)
                    end
                end

                w_nd[k1, k2] = w_ndk

                if denom
                    phi_ndk = basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1)
                    phi[k1, k2] = phi_ndk
                end
            end

            # flatten into a vector
            w = vec(w_nd)

            # sum over all dimension 1:D and samples 1:N
            W += w * w' * exp(logweight)
            if denom
                phiv = vec(phi)
                V += phiv * phiv' * exp(logweight)
            end

        end
    end

    if denom
        return solve_eigenvalue_problem(W; B=V, arpack=arpack)
    else
        return solve_eigenvalue_problem(W; B=nothing, arpack=arpack)
    end
end

"""
    eigenVI_1D(K, X, dlogP, basis_fn, d_basis_fn)

Computes α_hat, which is the eigenvector corresponding to the min eigenvalue of W.
Assumes we are working in 1D with real-valued basis-functions.

Inputs:
    K: order
    X: array of samples
    dlogP: function that computes \nabla log P(x) or contains an array of scores
    basis_fn: order-K O.N. basis function ϕ(x, K)
    d_basis_fn: derivative of order-K basis function dϕ(x, K)
    normalize: argument to divide by squared norm
    grads: if true, allows for passing in a vector of derivative evals instead of the function itself
        (used for the 2D factorized case)

Returns α_hat, the matrix W, and the minimum eigenvalue (i.e., the objective).
"""
function eigenVI_1D(K, X, dlogP, basis_fn, d_basis_fn; grads=false, denom=false, arpack=false, logPi=nothing)

    N = length(X)
    W = zeros(K, K)
    if denom
        V = zeros(K, K)
    end

    for n in 1:N
        x = X[n]
        # importance weight; by default is uniform
        if logPi != nothing
           logweight = -logPi(x)
        else
           logweight = -log(N)
        end
        # evaluate basis functions and derivatives
        Dϕ = d_basis_fn.(x, 0:(K-1))
        ϕ = basis_fn.(x, 0:(K-1))
        if grads
            w_nd = 2 * Dϕ .- dlogP[n] .* ϕ
        else
            w_nd = 2 * Dϕ .- dlogP(x) .* ϕ
        end

        W += w_nd * w_nd' * exp(logweight)

        # Denominator correction
        if denom
            V += ϕ * ϕ' * exp(logweight)
        end
    end

    if denom
        return solve_eigenvalue_problem(W; B=V, arpack=arpack)
    else
        return solve_eigenvalue_problem(W; B=nothing, arpack=arpack)
    end
end


