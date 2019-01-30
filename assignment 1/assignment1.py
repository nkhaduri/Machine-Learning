def featureNormalization(X):
    """ Feature Normalization """
    # your code here
    return (X - np.mean(X))/np.std(X)

def normalEquation(X, y):
    # Normal Equations
    return np.linalg.pinv(X.transpose().dot(X)).dot(np.transpose(X)).dot(y).ravel()

def computeCost(X, y, theta=[[0], [0]]):
    """ Computing Cost (for Multiple Variables) """
    return np.sum((X.dot(theta) - y) * (X.dot(theta) - y)) / (2 * X.shape[0])

def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    """ Gradient Descent (for Multiple Variables) """
    # J_history 
    J_history = []
    for k in range(num_iters):
        theta -= (alpha * (X.dot(theta) - y).transpose().dot(X) / X.shape[0]).transpose()
        J_history.append(computeCost(X, y, theta))
        
    return(theta, np.asarray(J_history))

def sigmoid(z):
    return 1 / (1 + np.float64(np.exp(-z)))

def costFunction(theta, X, y):
    """ Logistic Regression Cost """
    s = sigmoid(X.dot(theta))
    return (-(np.log(s).transpose().dot(y) + np.log(1 - s).transpose().dot(1 - y)) / X.shape[0])[0]

def gradient(theta, X, y):
    """ Logistic Regression Gradient """
    return ((X.transpose().dot(sigmoid(X.dot(theta).reshape(-1,1)) - y)) / X.shape[0]).ravel()

def predict(theta, X, threshold=0.5):
    """ Logistic Regretion predict """
    return (sigmoid(X.dot(theta)) >= threshold).astype(int)

def costFunctionReg(theta, reg, *args):
    """ Regularized Logistic Regression Cost """
    X = args[0]
    y = args[1]
    return costFunction(theta, X, y) + reg * (theta.transpose().dot(theta) - theta[0]**2) / (2 * X.shape[0]) 

def gradientReg(theta, reg, *args):
    """ Regularized Logistic Regression Gradient """
    X = args[0]
    y = args[1]
    grad = gradient(theta, X, y)
    theta[0] = 0.
    return grad + reg * theta / X.shape[0]