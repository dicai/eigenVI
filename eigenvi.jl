include("bam.jl")
include("polynomials.jl")

using Arpack

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
