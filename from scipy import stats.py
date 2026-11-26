from scipy import stats

mu = 100
sigma = 10
a = 75
b = 95

f = stats.norm.pdf(b,mu, sigma)
print('normal density = {0:6.4f}'.format(f))

cp1 = stats.norm.cdf(b, mu, sigma)
cp2 = stats.norm.cdf(a,loc=mu, scale=sigma)
p = cp1 - cp2
print('probability = {0:5.1f}%'.format(p*100))