using QuadGK

function construct_CDF(xmin, xmax, step, Q)
    Xrange = xmin:step:xmax
    C_table = [0.0]
    Xcurr = -Inf
    for (i, x) in enumerate(Xrange)
        C_curr = C_table[i] + quadgk(z -> Q(z), Xcurr, x, rtol=1e-3)[1]
        push!(C_table, C_curr)
        Xcurr = x
    end
    return C_table[2:end]
end


function sample_q_1D_approx(N::Int, Q::Function, zmin::Float64, zmax::Float64, num_knots::Int)
    """
    Samples from a 1D q.

    N: number of samples to take from Q
    Q: the distribution to sample from
    zmin, zmax: bounds on which to construct the CDF (I guess we could make the default recompute the quadrature)
    num_knots: how finely to construct the CDF
    """
    step=(zmax-zmin)/(num_knots-1)
    # construct array of CDF values
    CDFs = construct_CDF(zmin, zmax, step, Q)
    Zrange=zmin:step:zmax
    println(length(CDFs))

    U = rand(Uniform(0,1), N)

    Z_samples = zeros(N)
    for n in 1:N
        # find the closest value to 0
        zind = argmin(abs.(CDFs .- U[n]))
        Z_samples[n] = Zrange[zind]
    end

    return Z_samples
end

