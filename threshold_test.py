import numpy as np

thresholds = np.loadtxt('/Users/raghav/Downloads/thresholds.txt')
fpr = np.loadtxt('/Users/raghav/Downloads/fpr.txt')
tpr = np.loadtxt('/Users/raghav/Downloads/tpr.txt')



thresholds[np.argmin(np.abs(tpr - 0.15))]
fpr[np.argmin(np.abs(tpr - 0.15))]



fpr
