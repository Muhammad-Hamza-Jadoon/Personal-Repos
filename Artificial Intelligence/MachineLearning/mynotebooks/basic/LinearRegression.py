import numpy as np
import copy, math
import pandas as pd
import matplotlib.pyplot as plt



def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """

    # m = X.shape[0]
    # cost = 0.0
    # for i in range(m):                                
    #     f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
    #     cost = cost + (f_wb_i - y[i])**2       #scalar
    # total_cost = cost / (2 * m)                      #scalar  
    
    m = x.shape[0] 
    pred = np.dot(x, w) + b
    total_cost = (1/(2*m))*sum((pred - y)**2)

    return total_cost




def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """

    # m,n = X.shape           #(number of examples, number of features)
    # dj_dw = np.zeros((n,))
    # dj_db = 0.

    # for i in range(m):                             
    #     err = (np.dot(X[i], w) + b) - y[i]   
    #     for j in range(n):                         
    #         dj_dw[j] = dj_dw[j] + err * X[i, j]    
    #     dj_db = dj_db + err                        
    # dj_dw = dj_dw / m                                
    # dj_db = dj_db / m  
    
    m = x.shape[0]
    pred = np.dot(x, w) + b -y
    dj_dw = np.dot(x.T, pred)/m
    dj_db = np.mean(pred)    
        
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing