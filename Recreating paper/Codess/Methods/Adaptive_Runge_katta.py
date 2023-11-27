import numpy as np

"""
We will perform the Runge katta method just for 1 iteration as, we will doing it often in adaptive runge katta, hence it will useful to have 
this function. 
"""
def one_iter_RK4(x, y, h, f):
    # x is the independent variable, y is the array of dependent variables, h is step size, f is the function

    N = len(y)

    k1 = np.zeros(N)
    k2 = np.zeros(N)
    k3 = np.zeros(N)
    k4 = np.zeros(N)

    
    for j in range(0, N):
        k1[j] = h * ( f[j](x , y) )

    for j in range(0, N):
        k2[j] = h * ( f[j]( x + h/2 , y + k1/2  ) )

    for j in range(0, N):
        k3[j] =  h * ( f[j]( x + h/2 , y + k2/2 ) )

    for j in range(0, N):
        k4[j] =  h * (f[j]( x + h , y + k3 ) )
        
    Y = y + (k1 + 2* k2 + 2* k3 + k4) / 6
    X = x + h

    return X, Y

def Adaptive_Runge_katta(a, b, h_ini, m, f, s, delta):

    X = [a]
    Y = [s]
    H = [h_ini]

    while X[-1] <= b :

        temp_x, temp_y = one_iter_RK4(X[-1], Y[-1], H[-1], f)
        temp_h = one_iter_RK4(temp_x, temp_y, H[-1], f)[1]

        temp_2h = one_iter_RK4(X[-1], Y[-1], 2*H[-1], f)[1]

        e = np.linalg.norm(temp_2h - temp_h)
        rho = 30* H[-1]* delta/ e

        # if next h is greater than 2* h

        if rho >= 16 :
            X.append(X[-1] + H[-1])
            X.append(X[-1] + 2*H[-1])
            Y.append(temp_h)
            Y.append(temp_2h)
            H.append(2* H[-1])

        elif rho >= 1:
            X.append(X[-1] + H[-1]) 
            X.append(X[-1] + 2*H[-1])
            Y.append(temp_h)
            Y.append(temp_2h)
            H.append(H[-1]* rho**(1/4))

        else:
            H[-1] = H[-1]* rho**(1/4)

    return X, Y, H

            







    