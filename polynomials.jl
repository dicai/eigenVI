using LegendrePolynomials
using SpecialPolynomials

""" Legendre polynomials """
# normalized Legendre polynomials (NLP)
NLP(x, l) = Pl.(x, l, norm = Val(:normalized))

# derivative of NLP
DNLP(x, l) = ForwardDiff.derivative(x -> NLP(x, l), x)

""" Hermite polynomials """
# probabilists' Hermite polynomials
HP(x, l) =  basis(ChebyshevHermite, l)(x)
DHP(x, l) = derivative(basis(ChebyshevHermite, l))(x)

# normalized Hermite polynomials
NHP(x, l) = (sqrt(2*π) * factorial(l))^(-0.5) * sqrt(exp(-0.5*x^2)) * HP(x, l)

# derivative of normalized Hermite polynomials 
DNHP(x, l) = ForwardDiff.derivative(x -> NHP(x, l), x) 