import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, chi2, expon, t, logistic

# 1. Uniform Distribution
x = np.arange(1, 7)
y = [1/6] * 6
plt.bar(x, y)
plt.title('Discrete Uniform Distribution')
plt.xlabel('Outcomes')
plt.ylabel('Probability')
plt.savefig('uniform_distribution.png')
plt.clf()

# 2. Bernoulli Distribution
x = [0, 1]
y = [0.7, 0.3]  # Example with p=0.3
plt.bar(x, y)
plt.title('Bernoulli Distribution (p=0.3)')
plt.xlabel('Outcomes')
plt.ylabel('Probability')
plt.savefig('bernoulli_distribution.png')
plt.clf()

# 3. Binomial Distribution
n, p = 10, 0.5
x = np.arange(0, n+1)
y = binom.pmf(x, n, p)
plt.bar(x, y)
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.savefig('binomial_distribution.png')
plt.clf()

# 4. Poisson Distribution
x = np.arange(0, 15)
mu = 3
y = poisson.pmf(x, mu)
plt.bar(x, y)
plt.title('Poisson Distribution (lambda=3)')
plt.xlabel('Occurrences')
plt.ylabel('Probability')
plt.savefig('poisson_distribution.png')
plt.clf()

# 5. Normal Distribution
x = np.linspace(-3, 3, 1000)
y = norm.pdf(x)
plt.plot(x, y)
plt.title('Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.savefig('normal_distribution.png')
plt.clf()

# 6. T Distribution
df = 10
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df)
plt.plot(x, y)
plt.title("Student's T Distribution (df=10)")
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.savefig('t_distribution.png')
plt.clf()

# 7. Chi-Squared Distribution
df = 5
x = np.linspace(0, 20, 1000)
y = chi2.pdf(x, df)
plt.plot(x, y)
plt.title('Chi-Squared Distribution (df=5)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.savefig('chi_squared_distribution.png')
plt.clf()

# 8. Exponential Distribution
x = np.linspace(0, 5, 1000)
y = expon.pdf(x)
plt.plot(x, y)
plt.title('Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.savefig('exponential_distribution.png')
plt.clf()

# 9. Logistic Distribution
x = np.linspace(-10, 10, 1000)
y = logistic.pdf(x)
plt.plot(x, y)
plt.title('Logistic Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.savefig('logistic_distribution.png')
plt.clf()
