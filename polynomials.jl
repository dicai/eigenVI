using LegendrePolynomials
using SpecialPolynomials
using ForwardDiff

""" Legendre polynomials """
# normalized Legendre polynomials (NLP)
NLP(x, l) = Pl.(x, l, norm = Val(:normalized))

# derivative of NLP
DNLP(x, l) = ForwardDiff.derivative(x -> NLP(x, l), x)

""" Hermite polynomials """
# probabilists' Hermite polynomials
HP(x, l) =  basis(ChebyshevHermite, l)(x)
DHP(x, l) = SpecialPolynomials.derivative(basis(ChebyshevHermite, l))(x)

# Normalizing constant
norm_const = l -> inv(sqrt(sqrt(2π) * factorial(l)))

# NHP
NHP(x, l) = norm_const(l) * exp(-0.25*x^2) * HP(x, l)

# derivative of NHP
function DNHP(x, l)
    l == 0 && return 0.0
    nc = norm_const(l)
    expfac = exp(-0.25*x^2)
    nc * expfac * (l * HP(x, l-1) - 0.5 * x * HP(x, l))
end

""" Big Hermite polynomials """
# Normalizing constant for large l
big_norm_const = l -> inv(sqrt(sqrt(2π) * factorial(big(l))))

# NHP
bigNHP(x, l) = big_norm_const(l) * exp(-0.25*x^2) * HP(x, l)

# derivative of NHP
function bigDNHP(x, l)
    l == 0 && return 0.0
    nc = big_norm_const(l)
    expfac = exp(-0.25*x^2)
    nc * expfac * (l * HP(x, l-1) - 0.5 * x * HP(x, l))
end
