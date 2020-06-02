import numpy as np


from sklearn.base import BaseEstimator, RegressorMixin

from rpy2.robjects import Matrix, numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

norm = importr("norm")

numpy2ri.activate()
pandas2ri.activate()



class EMLR(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        Z = np.hstack((y[:, np.newaxis], X))
        s = norm.prelim_norm(Z)
        thetahat = norm.em_norm(s, showits=False)
        parameters = norm.getparam_norm(s, thetahat)
        self.mu_joint = np.array(parameters[0])
        self.Sigma_joint = np.array(parameters[1])
        return(self.mu_joint,self.Sigma_joint)


    def get_beta(self, X):
        d = X.shape[1]
        indices=np.arange(1,d+1)
        mu_X = self.mu_joint[indices]
        Sigma_X = self.Sigma_joint[np.ix_(indices, indices)]
        mu_y = self.mu_joint[0]
        Sigma_yX = self.Sigma_joint[0, indices]

        beta = Sigma_yX.dot(np.linalg.inv(Sigma_X))
        beta0 = mu_y - Sigma_yX.dot(np.linalg.inv(Sigma_X)).dot(mu_X)

        return(beta0, beta)
