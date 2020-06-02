import numpy as np

class LinearRegressorNA_oracle:
    """
        Debiased Stochastic Gradient Algorithm for Linear Regression
        with missing data in the covariates.

        It computes many attributes of a linear model with coefficients
        beta = (beta_1, …, beta_d), to be passed for the minimization of
        the residual sum of squares between the observed targets,
        and the targets predicted by the linear approximation.

        Parameters
        ----------
        X_comp : numpy array of floats, size nxd
            contains the complete features (no missing data)
        X : numpy array of floats, size nxd
            contains the incomplete features
        D : numpy array of boolean, size nxd
            mask indicator of missing data
            D[i,j]=1 if X[i,j] not missing
            D[i,j]=0 if X[i,j] missing
        y : numpy array of float, size n
            the complete vector of observations
        p : numpy array of float, size d
            either, is equal to a global probability of being missing
            or, is equal to a vector of size d containing the probabilities
            of being missing for each feature
        beta_true : numpy array of size d
            contains the true parameter of the underlying linear model (=>oracle<=)
        strength : float
            regularization parameter for a Ridge penalty

    """

    def __init__(self,X_comp,X,D,y,p,beta_true,strength):
        self.X_comp = X_comp
        self.X = X
        self.D = D
        self.y = y
        self.p = p
        self.beta_true = beta_true
        self.strength = strength
        self.n_samples, self.n_features = X.shape

    def loss(self,beta):
        """Computes the quadratic loss regularized with a Ridge term
            1/(2n) || y-X beta ||^2 + lambda || beta ||^2

        """
        y, X_comp, n_samples, strength = self.y, self.X_comp, self.n_samples, self.strength
        return 0.5*np.linalg.norm(y-X_comp.dot(beta))**2/n_samples + strength*np.linalg.norm(beta)**2/2

    def get_beta_true(self):
        """Returns the value of the true parameter model"""
        return self.beta_true

    def grad_i(self,i,beta):
        """Computes the gradient of f_i without debiasing"""
        x_i = self.X[i]
        y = self.y
        return (x_i.dot(beta)-y[i])*x_i + self.strength*beta

    def grad_corrNA_i(self,i,beta):
        """Computes a DEBIASED approximation of the gradient of f_i with imputed-to-0 data"""
        x_i = self.X[i]
        xx_i = x_i**2
        y = self.y
        p = self.p
        px_i = x_i/p
        return (px_i.dot(beta)-y[i])*px_i - \
                ((1-p)/p**2)*(xx_i*beta) + \
                self.strength*beta

    def lip_max(self):
        """Computes the maximum Lipschitz constant of f_i in presence of missing values"""
        X, D, n_features, p = self.X, self.D, self.n_features, self.p
        return (((X**2).sum(axis=1))/D.sum(axis=1)).max()*n_features/np.min(p)**2

    def lip_max_theo(self):
        """Computes the maximum Lipschitz constant of f_i"""
        X_comp, p = self.X_comp, self.p
        return ((X_comp**2).sum(axis=1)+self.strength).max()/np.min(p)**2


################################################################################

#typedef void VoidCallback();

def inspector(model, verbose):
    """ Used to monitor SGD in the oracle case

    Parameters
    ----------
    model : class
            use the oracle class *LinearRegressorNA_oracle* which debiases SGD for Linear Regression with missing data in the covariates
    niter : integer
            corresponds to the number of iterations to do, i.e. the number of passes through data times the size of data
    verbose : boolean
            True for printing the quadratic loss at each iteration, False otherwise

    """
    beta_true = model.get_beta_true()
    objectives = []
    dist_it = []
    it = [0] # This is a hack to be able to modify 'it' inside the closure.
    def inspector_cl(w):
        #global beta_true
        obj = model.loss(w)
        norm_it = np.linalg.norm(w-beta_true)
        objectives.append(obj)
        dist_it.append(norm_it)
        if verbose == True:
            if it[0] == 0:
                print(' | '.join([name.center(8) for name in ["it", "obj"]]))
            if it[0] %  1 == 0: #(niter) == 0:
                print(' | '.join([("%d" % it[0]).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % norm_it).rjust(8)]))
            it[0] += 1
    inspector_cl.objectives = objectives
    inspector_cl.dist_it = dist_it
    return inspector_cl

################################################################################
################################################################################
################################################################################
##                          WITH COMPLETE DATA
################################################################################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
##############################    SGD       ###################################
#-------------------------------------------------------------------------------


def sgd(model, beta0, nepoch = 1, choice_step = 'cst', step = 1, verbose = True, callback = lambda *args, **kwargs: None, callback_modulo = 100, seed = 42):
    """Standard stochastic gradient algo

    Parameters
    ----------
    model : class
            use the oracle class *LinearRegressorNA_oracle* which debiases SGD for Linear Regression with missing data in the covariates
    beta0 : numpy array of size d
            contains the initialized parameter of the linear model
    nepoch : integer
            corresponds to the number of iterations to do, i.e. the number of passes through data times the size of data
    choice_step : "sqrt", "mu" or "cst"
            indicates the methods to use, among the following:
            "sqrt" corresponds to the decreasing step size proportional to $1/sqrt{k+1}$, where $k$ is the iteration
            "mu" corresponds to the decreasing constant step size proportional to $1/k+1$, where $k$ is the iteration
            "cst" correponds to the constant step size
    step: float
            constant part of the step size
    verbose : boolean
            True for printing the quadratic loss at each iteration, False otherwise
    callback : function
            use the function *inspector* to stock the quadratic loss at each iteration
    callback_modulo : integer
            the function *inspector* is called every *callback_modulo* iterations
    seed : integer
            random seed

    """
    np.random.seed(seed)

    n_samples = model.n_samples
    idx_samples = np.random.permutation(n_samples)

    beta = beta0.copy()
    callback(beta0)

    if choice_step == "sqrt":
        for epoch in range(nepoch):
            for idx in range(n_samples):
                step_it = 1/np.sqrt(idx+1)
                i = idx_samples[idx]
                beta -= step * step_it * model.grad_i(i,beta)
                if idx % callback_modulo == 0:
                    callback(beta)
    elif choice_step == "mu":
        for epoch in range(nepoch):
            for idx in range(n_samples):
                step_it = 1/(idx+1)
                i = idx_samples[idx]
                beta -= step * step_it * model.grad_i(i,beta)
                if idx % callback_modulo == 0:
                    callback(beta)
    else:
        for epoch in range(nepoch):
            for idx in range(n_samples):
                i = idx_samples[idx]
                beta -= step * model.grad_i(i,beta)
                if idx % callback_modulo == 0:
                    callback(beta)
    return beta

#-------------------------------------------------------------------------------
###########################    AVERAGED SGD     ################################
#-------------------------------------------------------------------------------

def avsgd(model, beta0, nepoch = 1,  step = 1, verbose = True, callback = lambda *args, **kwargs: None, callback_modulo = 100, seed = 42):
    """Averaged stochastic gradient algo

    Parameters
    ----------
    model : class
            use the oracle class *LinearRegressorNA_oracle* which debiases SGD for Linear Regression with missing data in the covariates
    beta0 : numpy array of size d
            contains the initialized parameter of the linear model
    nepoch : integer
            corresponds to the number of passes through data
    step: float
            constant part of the step size.
    verbose : boolean
            True for printing the quadratic loss at each iteration, False otherwise
    callback : function
            use the function *inspector* to stock the quadratic loss at each iteration
    callback_modulo : integer
            the function *inspector* is called every *callback_modulo* iterations
    seed : integer
        random seed

"""
    np.random.seed(seed)

    n_samples = model.n_samples
    idx_samples = np.random.permutation(n_samples)


    beta = beta0.copy()
    beta_av = beta0.copy()
    callback(beta0)
    for epoch in range(nepoch):
        for idx in range(n_samples):
            i = idx_samples[idx]
            beta -= step * model.grad_i(i,beta)
            beta_av = beta/(idx+2.)+(idx+1.)/(idx+2.)*beta_av
            if idx % callback_modulo == 0:
                callback(beta_av)
    return beta_av


################################################################################
################################################################################
################################################################################
##                          DEBIASED APPROACHES
################################################################################
################################################################################
################################################################################




#-------------------------------------------------------------------------------
########################    DEBIASED SGD APPROX      ##########################
#-------------------------------------------------------------------------------


def sgdNA(model, beta0 ,nepoch = 1 , choice_step = 'cst', step = 1, verbose = True, callback = lambda *args, **kwargs: None, callback_modulo = 100, seed = 42):
    """Standard stochastic gradient algo using a debiased gradient

    Parameters
    ----------
    model : class
            use the oracle class *LinearRegressorNA_oracle* which debiases SGD for Linear Regression with missing data in the covariates
    beta0 : numpy array of size d
            contains the initialized parameter of the linear model
    nepoch : integer
            corresponds to the number of passes through data
    choice_step : "sqrt", "mu" or "cst"
            indicates the methods to use, among the following:
            "sqrt" corresponds to the decreasing step size proportional to $1/sqrt{k+1}$, where $k$ is the iteration
            "mu" corresponds to the decreasing constant step size proportional to $1/k+1$, where $k$ is the iteration
            "cst" correponds to the constant step size
    step: float
            constant part of the step size
    verbose : boolean
            True for printing the quadratic loss at each iteration, False otherwise
    callback : function
            use the function *inspector* to stock the quadratic loss at each iteration
    callback_modulo : integer
            the function *inspector* is called every *callback_modulo* iterations
    seed : integer
        random seed

    """
    np.random.seed(seed)

    n_samples = model.n_samples
    idx_samples = np.random.permutation(n_samples)

    beta = beta0.copy()
    callback(beta0)
    if choice_step == "sqrt":
        for epoch in range(nepoch):
            for idx in range(n_samples):
                step_it = 1/np.sqrt(idx+1)
                i = idx_samples[idx]
                beta -= step * step_it * model.grad_corrNA_i(i,beta)
                if idx % callback_modulo == 0:
                    callback(beta)
    elif choice_step == "mu":
        for epoch in range(nepoch):
            for idx in range(n_samples):
                step_it = 1/(idx+1)
                i = idx_samples[idx]
                beta -= step * step_it * model.grad_corrNA_i(i,beta)
                if idx % callback_modulo == 0:
                    callback(beta)
    else:
        for epoch in range(nepoch):
            for idx in range(n_samples):
                i = idx_samples[idx]
                beta -= step * model.grad_corrNA_i(i,beta)
                if idx % callback_modulo == 0:
                    callback(beta)
    return beta


#----------------------------------------------------------------------------------------
###########################    DEBIASED AVERAGED SGD     ################################
#----------------------------------------------------------------------------------------



def avsgdNA(model, beta0 ,nepoch = 1 , step = 1, verbose = True, callback = lambda *args, **kwargs: None, callback_modulo = 100, seed = 42):
    """Averaged stochastic gradient algo using a debiased gradient

    Parameters
    ----------
    model : class
            use the oracle class *LinearRegressorNA_oracle* which debiases SGD for Linear Regression with missing data in the covariates
    beta0 : numpy array of size d
            contains the initialized parameter of the linear model
    nepoch : integer
            corresponds to the number of passes through data
    step: float
            constant part of the step size
    verbose : boolean
            True for printing the quadratic loss at each iteration, False otherwise
    callback : function
            use the function *inspector* to stock the quadratic loss at each iteration
    callback_modulo : integer
            the function *inspector* is called every *callback_modulo* iterations
    seed : integer
          random seed

    """
    np.random.seed(seed)

    n_samples = model.n_samples
    idx_samples = np.random.permutation(n_samples)
    beta = beta0.copy()
    beta_av = beta0.copy()
    callback(beta0)
    n_samples = model.n_samples
    for epoch in range(nepoch):
        for idx in range(n_samples):
            i = idx_samples[idx]
            beta -= step*model.grad_corrNA_i(i,beta)
            beta_av = beta/(idx+2.)+(idx+1.)/(idx+2.)*beta_av
            if idx % callback_modulo == 0:
                callback(beta_av)
    return beta_av
