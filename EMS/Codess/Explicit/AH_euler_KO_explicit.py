import numpy as np
from findiff import FinDiff
import matplotlib.pyplot as plt


def round_of_to_lowest_hundered(x):
    return (x // 100)* 100


def Apparant_horizon_KO(p_given):
    # Fixing Constants

    """
    Using first order approximation for first derivative
    """
    M = 1
    Q = 0.9  
    b = 200
    p = p_given
    a = 0.5
    n = 2**(12) # Defining the number of grid points
    D = 0.039  # Dissipation coefficient for Kriess oliger dissipation scheme

    err = 10**(-4)
    Z = np.linspace(err, 1 - err, n)
    zstep = 1 / (n)
    Y = np.zeros(4)

    Y[3] = 2* M

    # Performing finite difference using first order difference (Eulers Method)

    for i in range(0, 3):
        c1 = Q**2 / (M* Z[2 - i]**2)
        c2 = 4* a**2* M* Z[2 - i]**4 / (p**4 * (Z[2 - i] - 1)**6 )

        int1 = np.exp(- 4* Z[2 - i]**2 / (p**2* (Z[2 - i] - 1)**2 ))
        int2 = np.exp(- 2* Z[2 - i]**2 / (p**2* (Z[2 - i] - 1)**2 ))

        term1 = c1* np.exp( - a**4* b* int1 ) 
        term2 = c2* int2
        term = term1 + term2    
        Y[2 - i] =  Y[3 - i]  - zstep * term

    R = np.zeros(n)
    R = M* Z/(1 - Z)

    # Using Fourth order finite difference by borrowing first 3 points from the Euler Method data
    Y4 = np.zeros(n)

    # Defining the initial conditions from the Eulers Method 
    Y4[n - 1] = Y[3]
    Y4[n - 2] = Y[2]
    Y4[n - 3] = Y[1]
    Y4[n - 4] = Y[0]

    for i in range(0, n - 4):
        c1 = Q**2 / (M* Z[n - i - 5]**2)
        c2 = 4* a**2* M* Z[n - i - 5]**4 / (p**4 * (Z[n - i - 5] - 1)**6 )

        int1 = np.exp(- 4* Z[n - i - 5]**2 / (p**2* (Z[n - i - 5] - 1)**2 ))
        int2 = np.exp(- 2* Z[n - i - 5]**2 / (p**2* (Z[n - i - 5] - 1)**2 ))

        term1 = c1* np.exp( - a**4* b* int1 ) 
        term2 = c2* int2
        term = term1 + term2     

        Y4[n - i - 5] = ( (3 - 12* D* zstep )* Y4[n - i - 1] + (-16 + 48* D* zstep )* Y4[n - i - 2] + (36 + 72* D* zstep)* Y4[n - i - 3] + \
                            (-48 + 48* D* zstep)* Y4[n - i - 4] + 12* zstep* term) / (-25 + 12* D* zstep)


    # Uncomment to see the location of apparent horizon in a graph 

    plt.plot(Z[: 4*n//5], Y4[: 4*n//5])
    plt.plot(Z[: 4*n//5], R[: 4*n//5])
    plt.show()
    """
    plt.plot(Z[n//5: 4*n//5], np.log10(abs(Y4[n//5: 4*n//5] - Y[n//5: 4*n//5]) / Y4[n//5: 4*n//5]))
    plt.show()
    """

    """
    Uncomment to see the value where the initial apparant horizon is located 
    """
    ind_cutoff = 0

    for i in range(0, n):
        if abs(Y4[i] - R[i]) <= 5* 10**(-4):
            print(f" Value of z = {Z[i]} \n Value of r =  {R[i]} \n value of y = {Y4[i]} \n Coressponding indice {i}")
            ind_cutoff = i
            break
            

    cuttoff = round_of_to_lowest_hundered(ind_cutoff)
    m = n - cuttoff

    # Defining Alpha at t = 0
    alpha0 = np.ones(m)
    alpha = [alpha0] # This is the list which we will evolve forward in time

    # Defining Pi at t = 0
    pi0 = np.zeros(m)
    pi = [pi0]

    # Defining y in the required range at t = 0
    y0 = Y4[n-m:]

    # Defining r in the required range 
    r = R[n-m:]

    # Defining z in the required range
    z = Z[n-m:]

    # Defining Phi in the required range at t = 0

    phi0 = a* np.exp(- z**2 / (p**2 * (1 - z)**2 ))
    phi = [phi0]

    # Defining Capital Phi as it plays a major role at t = 0
    dz = z[1] - z[0]
    d_dz = FinDiff(0, dz, 1, acc = 4)

    Capital_phi0 = d_dz(phi0) / d_dz(r)
    Capital_phi = [Capital_phi0]

    # Defining Zeta in the required range at t = 0

    zeta0 = np.sqrt(y0/r)
    zeta = [zeta0]

    # Defining the time interval

    N = 300 # No of time points we are evolving the system
    t_ini = 0
    t_final = 0.3
    t = np.linspace(t_ini, t_final, N)
    dt = (t_final - t_ini) / N # stepsize

    for i in range(0, N):

        # Differential equation 1

        tempzeta = zeta[-1] + dt* ( (r* alpha[-1] / zeta[-1])* (pi[-1] + Capital_phi[-1]* zeta[-1])* ( pi[-1]*zeta[-1] + Capital_phi[-1]) )

        tempphi = phi[-1] + dt* alpha[-1]* (pi[-1] + Capital_phi[-1]* zeta[-1])

        temp1 = d_dz( (pi[-1]* zeta[-1] + Capital_phi[-1])* alpha[-1]* r**2 ) / d_dz(r)
        temp2 = alpha[-1]* Q**2 * 4* b* np.exp(- b* phi[-1]**4 )* phi[-1]**3 / (2* r**4)

        temppi = pi[-1] + dt* ( temp1/r**2 - temp2 ) 

        # adding the value of the metric/ matter function at next time step
        zeta.append(tempzeta)
        pi.append(temppi)
        phi.append(tempphi)
        

        # Doing the same process for Capital phi
        temp_Capital_phi = d_dz(pi[-1]) / d_dz(r)
        Capital_phi.append(temp_Capital_phi)

        # Now we need to solve for the constraint alpha at every point in time, looking into the equation (5) in paper and doing some simplification 
        # we find that it better to use ln(alpha) than alpha for solving the equation or else the constraint will become implicit which is a problem
        # let l be the new variable instead of alpha, which will solve for the first 4 points using eulers method

        l = np.zeros(4)
        l[-1] = 0
        for j in range(0, 3):
            l_term = - d_dz(r)[2 - j]* r[2 - j]* pi[-1][2 - j]* Capital_phi[-1][2 - j] / ( zeta[-1][2 - j] )
            l[2 - j] = l[3 - j] - zstep* l_term

        # Continuing the solution using fourth order finite difference for better stability

        l4 = np.zeros(m)

        l4[m - 1] = l[3]
        l4[m - 2] = l[2]
        l4[m - 3] = l[1]
        l4[m - 4] = l[0]

        for j in range(0, m - 4):
            l4_term = - d_dz(r)[m - j - 5]* r[m - j - 5]* pi[-1][m - j - 5]* Capital_phi[-1][m - j - 5] / ( zeta[-1][m - j - 5] )
            
            l4[m - j - 5] = ((3 - 12* D* zstep )* l4[m - j - 1] + (-16 + 48* D* zstep )* l4[m - j - 2] + (36 + 72* D* zstep)* l4[m - j - 3] + \
                            (- 48 + 48* D* zstep)* l4[m - j - 4] + 12* zstep* l4_term) / (-25 + 12* D* zstep)
            
        # Adding the latest value of alpha to the array
        alpha.append(np.exp(l4))   
        print(i)

    # Now we using the time evolved data from these simulation we will try to plot the evolution of apparant horizon 

    apparant_horizon_phi = np.zeros(N)
    apparant_horizon_radius = np.zeros(N)

    for i in range(0, N):
        for j in range(0, m):
            if abs(zeta[i][j] - 1) <= 10**(-3):
                apparant_horizon_phi[i] = phi[i][j]
                apparant_horizon_radius[i] = r[j]
                break


    plt.plot(t, apparant_horizon_radius)
    plt.show()
    plt.plot(t, apparant_horizon_phi)
    plt.show()

    return t, apparant_horizon_radius, apparant_horizon_phi

