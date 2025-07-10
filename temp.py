import math

# Given parameters
S = 50           # Current stock price
K = 40           # Strike price
r = 0.05         # Continuously compounded risk-free rate
T = 1.5          # Time to maturity in years
sigma = 0.15     # Volatility

# Calculate d1 and d2
d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

# Standard normal CDF using the error function
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

# Discount factor Z(t, T)
Z = math.exp(-r * T)

# Black-Scholes price
C = S * Phi(d1) - K * Z * Phi(d2)
print(f"Call option price: {C:.2f}")
