import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.stats import poisson
from scipy.stats import norm
# You need to import the stats module
num_samples = 10000
np.random.seed(564221)
# 1a
# Write the code for generating the gs variable. 
# This is the simplest random variable of the problem 
# and can be generated independent of the others.
mean_value = 7.25
standard_deviation = 1.220238
mean_ptime = 120
std_dev_ptime = 30
gs= np.random.normal(mean_value, standard_deviation, num_samples)
#1b
#Perform the probability inrtegral transform 
# and replicate the associated plots.
mean = [0, 0, 0]
print(mean)
matrix = ([1.0, 0.6, -0.9],[0.6, 1.0, -0.5],[-0.9, -0.5, 1.0])
matrix_alt = np.array([[1.0, 0.6, -0.9, 0],[0.6, 1.0, -0.5, 0],[-0.9, -0.5, 1.0, 0],[1.0, 0, 0, 1.0]])
ak = np.random.multivariate_normal(mean, matrix, num_samples)[:,0]
pp = np.random.multivariate_normal(mean, matrix, num_samples)[:,1]
ptime = np.random.multivariate_normal(mean, matrix, num_samples)[:,2]


# 1c
# Perform the probability integral transform 
# and replicate the associated plots
ak_norm = stats.norm(0, 1).cdf(ak)
pp_norm = stats.norm(0, 1).cdf(pp)
ptime_norm = stats.norm(0, 1).cdf(ptime)

# Create histograms to visualize the transformed variables
plt.title('Histogram')
plt.hist(ak_norm, bins=30, alpha=1, label='ak_norm',color='orange')
plt.xlabel('Normal distribution of ak')
plt.ylabel('Frequency')
plt.hist(pp_norm, bins=30, alpha=1, label='pp_norm',color='purple')
plt.xlabel('Normal distribution of pp')
plt.ylabel('Frequency')
plt.hist(ptime_norm, bins=30, alpha=1, label='ptime_norm',color='pink')
plt.xlabel('Normal distribution of ptime')
plt.ylabel('Frequency')
plt.show()

# 1d
# Perform inverse transform sampling
# Define the target distribution (e.g., standard normal

ptime = norm.ppf(ptime_norm, loc=mean_ptime, scale=std_dev_ptime)
ak = poisson.ppf(ak_norm, 5)
pp = poisson.ppf(pp_norm, 15)


# Create histograms to visualize the transformed variables
plt.title('Histogram')
plt.hist(ak_norm, bins=30, alpha=1, label='ak_norm',color='orange')
plt.xlabel('Normal distribution of ak')
plt.ylabel('Frequency')
plt.hist(pp_norm, bins=30, alpha=1, label='pp_norm',color='purple')
plt.xlabel('Normal distribution of pp')
plt.ylabel('Frequency')
plt.hist(ptime_norm, bins=30, alpha=1, label='ptime_norm',color='pink')
plt.xlabel('Normal distribution of ptime')
plt.ylabel('Frequency')
plt.show()

# 1e
# Replicate the final 
# plot showcasing the correlations between the variables.

mean=[0, 0, 0, 0]
ak=np.random.multivariate_normal(mean, cov=matrix_alt, size=num_samples)[:,0]
pp=np.random.multivariate_normal(mean, cov=matrix_alt, size=num_samples)[:,1]
ptime=np.random.multivariate_normal(mean, cov=matrix_alt, size=num_samples)[:,2]
gs=np.random.multivariate_normal(mean, cov=matrix_alt, size=num_samples)[:,3]
plt.title('Histogram')
plt.hist(ak, bins=30, alpha=1, label='ak_norm',color='orange')
plt.xlabel('Normal distribution of ak')
plt.ylabel('Frequency')
plt.hist(pp, bins=30, alpha=1, label='pp_norm',color='purple')
plt.xlabel('Normal distribution of pp')
plt.ylabel('Frequency')
plt.hist(ptime, bins=30, alpha=1, label='ptime_norm',color='pink')
plt.xlabel('Normal distribution of ptime')
plt.ylabel('Frequency')
plt.hist(gs, bins=30, alpha=1, label='gs_norm',color='grey')
plt.xlabel('Normal distribution of gs')
plt.ylabel('Frequency')
plt.show()
