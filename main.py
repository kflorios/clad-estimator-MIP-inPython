import cplex
import sys
import numpy as np
import pandas as pd
import math
from docplex.mp.progress import *
def CladCompute():
    #Main Program: Computes CLAD by defining a MILP and calling milp.py
    X,y,w = readXyw()
    X,mu,sigma = standardizeX(X,y)
    c,A,b,M = definecAbM(X,y,w)
    lb,ub,Aeq,beq,n,p,best = definelbub(X,y)
    x,cost,feasible,time = milp_cplex(c,A,b,Aeq,beq,lb,ub,n,p,M,X,y, best)
    estimatesNorm=x
    value=cost
    status=feasible
    runtime=time
    quality=status
    estimatesRaw=denormalizeEstimates(estimatesNorm,mu,sigma)
    estimates=estimatesRaw
    return value,estimates,time,quality


def readXyw():
    #Reads X,y,w of given CLAD problem
    #X = np.loadtxt("X_200_4.txt")
    X = np.loadtxt("X_601_5.txt")
    X=X[:,1:]
    #y = np.loadtxt("ys_200_4.txt")
    y = np.loadtxt("ys_601_5.txt")
    y=y[:,1:]
    #w = np.loadtxt("w_numeric200.txt")
    #w=w[:,1:]
    w=np.ones(np.size(X,0))
    return X,y,w

def standardizeX(X,y):
    #Standardizes X
    df = pd.DataFrame.from_records(X)
    mu = df.mean(axis=0)
    sigma = df.std(axis=0)
    testX = (X - np.tile(mu,(np.size(X,0),1))) / np.tile(sigma,(np.size(X,0),1))
    p = np.size(X,1)
    for j in range(p):
        if math.isnan(testX[0,j]):
            X[:,j] = X[:,j]
        else:
            X[:,j] = testX[:,j]
    return X,mu,sigma

def definecAbM(X,y,w):
    #Defines c,A,b for milp.py
    n=np.size(X,0)
    p=np.size(X,1)
    c3=np.tile(0,(1,n))  #gammas integer
    c1=np.tile(0,(1,p))    #betas
    c2=np.tile(0,(1,n))      #phis
    c4=np.tile(1,(1,n))        #sm's
    c5=np.tile(1,(1,n))          #sp's
    c=np.hstack((c3,c1,c2,c4,c5))
    d = 50
    #d=20
    #d=5
    M = np.zeros(n)
    for i in range(n):
        M[i] = np.abs(X[i, 0])*d + sum(np.abs(X[i, j]) * d for j in range(1, p))
    A1=np.zeros((n,4*n+p))
    for i in range(n):      #constraint 3c in pdf ssrn
        A1[i,range(n)]=0  #gammas
        for j in range(p):
            A1[i,(n+j)]=X[i,j]  #betas
        A1[i,(n+p):(2*n+p-1)]=-1*(range(n)==i)  #phis
        A1[i,(2*n+p):(3*n+p-1)]=0              #sm's
        A1[i,(3*n+p):(4*n+p-1)]=0               #sp's
    b1=np.zeros(n)
    A2=np.zeros((n,4*n+p))
    for i in range(n):       # constraint 3e in pdf ssrn
        A2[i,range(n)]=M[i]*(range(n)==i)  #gammas
        for j in range(p):
            A2[i,(n+j)]=X[i,j]  #betas
        A2[i,(n+p):(2*n+p-1)]=1*(range(n)==i)    #phis
        A2[i,(2*n+p):(3*n+p-1)]=0    #sm's
        A2[i,(3*n+p):(4*n+p-1)]=0     #sp's
    b2=np.zeros(n)
    for i in range(n):
        b2[i]=M[i]
    A3=np.zeros((n,4*n+p))
    for i in range(n):    #constraint 3f in pdf ssrn
        A3[i,range(n)]=-M[i]*(range(n)==i)  #gammas
        A3[i,(n):(n+p-1)]=0  #betas
        A3[i,(n+p):(2*n+p-1)]=1*(range(n)==i)  #phis
        A3[i,(2*n+p):(3*n+p-1)]=0  #sm's
        A3[i,(3*n+p):(4*n+p-1)]=0  #sp's
    b3=np.zeros(n)
    for i in range(n):
        b3[i]=0  # for left censoring at 0
    A=np.concatenate((A1,A2,A3),axis=0)
    b=np.concatenate((b1,b2,b3))
    return c,A,b,M

def definelbub(X,y):
    #Defines lb,ub for milp.py
    d = 50
    #d=20
    #d=10
    #d=5
    n=np.size(X,0)
    p=np.size(X,1)
    lb1=np.tile(0,(1,n))  #gammas
    lb2=np.tile(-d,(1,p))  #betas
    #lb3=np.tile(-Inf,(1,n))  #phis
    lb3=np.tile(0,(1,n))       #phis, left censoring at (0)
    lb4=np.tile(0,(1,n))      #sm's
    lb5=np.tile(0,(1,n))       #sp's
    lb=np.hstack((lb1,lb2,lb3,lb4,lb5))
    BIGNUM=np.inf
    ub1=np.tile(1,(1,n))  #gammas
    ub2=np.tile(d,(1,p))   #betas
    ub3=np.tile(BIGNUM,(1,n))  #phi's
    ub4=np.tile(BIGNUM,(1,n))   #sm's
    ub5=np.tile(BIGNUM,(1,n))    #sp's
    ub=np.hstack((ub1,ub2,ub3,ub4,ub5))
    Aeq=np.zeros((n,4*n+p))
    for i in range(n):
        Aeq[i,range(n)]=0  #gammas
        Aeq[i,(n):(n+p-1)]=0  #betas
        Aeq[i,(n+p):(2*n+p-1)]=-1*(range(n)==i)  #phis
        Aeq[i,(2*n+p):(3*n+p-1)]=1*(range(n)==i)  #sm's
        Aeq[i,(3*n+p):(4*n+p-1)]=-1*(range(n)==i)  #sp's
    beq=np.zeros(n)
    for i in range(n):
        beq[i]=-y[i]
    best=np.inf
    return lb,ub,Aeq,beq,n,p,best

def milp_cplex(c,A,b,Aeq,beq,lb,ub,n,p,M, X,y,best):
    # Solves a mixed integer lp using cplex 20.1
    # c: is objective function coefficients A: is constraint matrix
    # b: is constraint vector
    # lb: lower bound ub: upper bound n: number of 0-1 variables
    # best: is best solution so far
    # Note this uses the Python/Cplex interface documented at
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_20.1.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html
    # Also, it assumes first n variables must be integer.
    # The MIP equations of the CLAD estimator are available at
    # Bilias, Y., Florios, K., & Skouras, S.(2019).Exact computation
    # of Censored Least Absolute Deviations estimator.
    # Journal of Econometrics, 212(2), 584 - 606.
    # Written by Kostas Florios, January 2, 2021
    #
    #
    # echo %PYTHONPATH%
    # C:\Program Files\IBM\ILOG\CPLEX_Studio201\cplex\python\3.7\x64_win64
    # (setup a conda environment called "cplex")
    # conda create --name=cplex python=3.7.6
    # conda activate cplex
    # run the proper install procedure in that environment (see ibm link above)
    # also pip install numpy and pip install pandas in the conda environment
    # python main.py
    from docplex.mp.model import Model
    M = M
    model=Model("clad")
    y = y.flatten()
    R1 = range(n)
    R2 = range(p)
    R3 = range(n)
    R4 = range(n)
    R5 = range(n)
    idx1  = [(i) for i in R1]
    idx2 = [(j) for j in R2]
    idx3 = [(i) for i in R3]
    idx4 = [(i) for i in R4]
    idx5 = [(i) for i in R5]
    d = 50
    #d=20
    #d=10
    #d=5
    gamma = model.binary_var_dict(idx1, None)
    beta = model.continuous_var_dict(idx2,lb=-d,ub=d)
    phi  = model.continuous_var_dict(idx3,lb=0,ub=np.inf)  #left censoring at zero (0)
    sm   = model.continuous_var_dict(idx4,lb=0,ub=np.inf)
    sp   = model.continuous_var_dict(idx5,lb=0,ub=np.inf)
    for i in R1:
        model.add_constraint(phi[i] >= model.sum(X[i,j]*beta[j] for j in R2) )
    for i in R1:
        model.add_constraint(phi[i] >= 0)
    for i in R1:
        model.add_constraint(phi[i] <= model.sum(X[i,j]*beta[j] for j in R2) + M[i]*(1-gamma[i]) )
    for i in R1:
        model.add_constraint(phi[i] <= 0 + M[i]*gamma[i])
    for i in R1:
        model.add_constraint(y[i] - phi[i] + sm[i] - sp[i] ==0)
    for j in R2:
        model.add_constraint(beta[j] >= -d)
        model.add_constraint(beta[j] <= +d)
    model.total_inside_obj = model.sum(sm[i] for i in R1) + model.sum(sp[i] for i in R1)
    model.add_kpi(model.total_inside_obj, "inside cost")
    model.minimize(model.total_inside_obj)
    model.print_information()
    model.export_as_lp("model.lp")
    #cplex options, experiment with these
    model.context.cplex_parameters.emphasis.mip=3
    #
    #monitoring progress
    #https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/progress.ipynb
    #from docplex.mp.progress import *
    # connect a listener to the model
    model.add_progress_listener(TextProgressListener())
    ok=model.solve(clean_before_solve=True)
    print(ok)
    #x=ok.get_value_list([beta[0],beta[1],beta[2],beta[3],beta[4]])
    x = ok.get_value_list(list(beta[j] for j in range(p)))
    deviation=ok.objective_value
    feasible=ok.solve_details.status
    time=ok.solve_details.time
    return x,deviation,feasible,time

def denormalizeEstimates(estimatesNorm,mu,sigma):
    #denormalized estimatesNorm obtained by Cplex MIP to estimatesRaw
    #which are meaningful to the user
    #quick and dirty implementation, based on GAMS and Fortran Analogues
    p = len(estimatesNorm)
    betaNorm=estimatesNorm
    betaRaw = np.zeros(p)
    betaHelp = np.zeros(p)
    for j in range(p):
        if (sigma[j] != 0):
            betaHelp[j] = betaNorm[j]/sigma[j]
        if (sigma[j] ==0):
            for jj in range(p):
                if (sigma[jj] != 0):
                    betaHelp[j] = betaHelp[j] - betaNorm[jj]*mu[jj]/sigma[jj]
                else:
                    jj0=jj
            betaHelp[j] = betaHelp[j]+betaNorm[jj0]
    for j in range(p):
        #betaRaw[j] = betaHelp[j]/betaHelp[0]
        betaRaw[j] = betaHelp[j]
    estimatesRaw=betaRaw
    return estimatesRaw


if __name__ == "__main__":

    #CladCompute()
    value, estimates, time, quality = CladCompute()
    print(value)
    print(estimates)
    print(time)
    print(quality)