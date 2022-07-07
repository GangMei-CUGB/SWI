"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np


def gen_testdata():
    data = np.load("test.npz")
    t, x= data["t"], data["x"].T
    x= data["x"].T
    x=np.ravel(x)
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + 0.0015844 * dy_x - 0.26933 * dy_xx


def boundary_on(x,on_boundary):
    return np.isclose(x[0],0)

def boundary_down(x,on_boundary):
    return on_boundary and np.isclose(x[0],-100)

geom = dde.geometry.Interval(-100, 0)
timedomain = dde.geometry.TimeDomain(0, 2400)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)



bc1 = dde.DirichletBC(geomtime, lambda x: 0.337, boundary_on )

bc2 = dde.DirichletBC(geomtime, lambda x: 0.188, boundary_down)


ic = dde.IC(
    geomtime, lambda x:0.188, lambda _, on_initial: on_initial
)


data = dde.data.TimePDE(
    geomtime, pde, [bc1, bc2, ic], num_domain=25400, num_boundary=800, num_initial=1600
)

net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
print('losshistory:',losshistory)
print('train_state',train_state)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


X= gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
np.savetxt("light.dat", np.hstack((X, y_pred)))