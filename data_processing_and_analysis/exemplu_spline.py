
import scipy.interpolate as si
import numpy as np
import matplotlib.pyplot as plt

nt = 3 # the number of knots
x = np.linspace(0,24,10)
y = np.ones((x.shape)) # the function we wish to interpolate (very trivial in this case)
t = np.linspace(x[0],x[-1],nt)
t = t[1:-1]

tt = np.linspace(0.0, 24, 100)
y_rep = si.splrep(x, y, t=t, k=2)

y_i = si.splev(tt, y_rep)

spl = np.zeros((100,))

plt.figure()
for i in range(nt+1):
    vec = np.zeros(nt+1)
    vec[i] = 1.0
    y_list = list(y_rep)
    y_list[1] = vec.tolist()
    y_i = si.splev(tt, y_list) # your basis spline function
    spl = spl + y_i*y_rep[1][i] # the interpolated function
    plt.plot(tt, y_i)



    plt.show()