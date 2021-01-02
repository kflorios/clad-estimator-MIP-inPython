# clad-estimator-MIP-inPython
Exact computation of Censored Least Absolute Deviations estimator with Mixed Integer Programming via Python code

Use handy Python code from the file main.py in order to exactly compute the CLAD estimator with MIP. 
The main code sets up the matrices A,b,c,Aeq,beq,lb,ub of the MIP model,
and then relies on a MIP solver to solve it.
We can use CPLEX with the docplex interface. All functions are supplied on top of the single file main.py.

The dataset is read in readXyw function via the files X.txt and ys.txt which can be adopted as desired. 
Currently, left censoring at zero is supposed, as is in most applications of CLAD.
In order to have a more flexible modeling approach, readers are suggested to consult the GAMS version
of the same model in https://www.gams.com/modlib/libhtml/clad.htm.

The code is generic. It supports any reasonable value for sample size, N, and number of predictors, p.
It also prints out the progress of the solution process during the run.

Feedback for the Python code at cflorios@central.ntua.gr, cflorios@aueb.gr.

In case you have trouble using the docplex interface, do not hesitate to contact me for support.

This is a translation of my own Matlab code available in another repository of mine (https://github.com/kflorios/clad-estimator-mip).

For completeness, I supply the Matlab manual here too.

The expected result is

value = 826.9999999999984
estimates = (5.7, -3.,  -6.,  -3.,  37.5)
time = 35.60899999999674
quality = integer optimal, tolerance


Suggested publication:  

Bilias, Y., Florios, K., & Skouras, S. (2019). Exact computation of Censored Least Absolute Deviations estimator.
Journal of Econometrics, 212(2), 584-606.

https://www.sciencedirect.com/science/article/abs/pii/S030440761930140X
