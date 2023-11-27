import numpy as np

"""
This certain has been written to handle both system first order ODE and Single higher order ODE, only care that needs to be taken is
properly initialize the array of functions. The variable f should be given in such a way that it has two arguments the first argument
being the independent variable and the second is an array of dependent variables. 
"""

def Runge_Katta_system(a, b, h, m, f, s): 

    N = int( (b - a) / h )
    X = np.zeros(N)
    Y = np.zeros([ N , m ])

    for k in range(0, N):
        X[k] = a + k* h
    
    # Adding the initial conditions
    Y[0] = s
    k1 = np.zeros(m)
    k2 = np.zeros(m)
    k3 = np.zeros(m)
    k4 = np.zeros(m)

    for i in range(0, N - 1):

        for j in range(0,m):
            k1[j] = h * ( f[j]( X[i] , Y[i]) )

        for j in range(0,m):
            k2[j] = h * ( f[j]( X[i] + h/2 , Y[i] + k1/2  ) )

        for j in range(0,m):
            k3[j] =  h * ( f[j]( X[i] + h/2 , Y[i] + k2/2 ) )

        for j in range(0,m):
            k4[j] =  h * (f[j]( X[i] + h , Y[i] + k3 ) )
        
        Y[i + 1] = Y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6 
    
    return X, Y
