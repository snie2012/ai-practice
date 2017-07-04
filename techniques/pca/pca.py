import numpy as np

def pca(dataMat, topNfeat=9999999):
	meanVals = np.mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = np.cov(meanRemoved, rowvar=0)
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))
	eigValInd = np.argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat
