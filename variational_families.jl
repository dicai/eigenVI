# 1D variational family
q(x, α, K, basis_fn) = (α' * basis_fn.(x, 0:(K-1)))^2


function q_full(x, α, K, basis_fn)
    """ 
    Constructs a 2D variational family q(x) = (∑_k α_k ϕ_k(x))^2.
    
    Inputs:
        x: input variable
        α: coefficients
        K: # of coefficients for orders 0, 1, ..., K-1.
        basis_fn: O.N. basis functions
    """
    
    ϕ = zeros(K, K)
    for k1 in 1:K, k2 in 1:K
        ϕ[k1, k2] = basis_fn(x[1], k1-1) * basis_fn(x[2], k2-1)
    end
    
    return (α' * vec(ϕ))^2
end

