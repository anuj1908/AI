import numpy as np
def activfun_sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
x=np.array([[1,1,0,1],[0,0,0,0],[0,1,1,1],[1,1,1,1],[0,1,0,1],[0,1,0,0],[1,0,0,1],[0,1,0,0]])
y= np.array([[0,0,1,0,1,0,1,0]]).T
np.random.seed(1)
w0=2*np.random.random((4,4))-1
w1=2*np.random.random((4,1))-1
for n in range(140):
    l_input = x
    l1 = activfun_sigmoid(l_input.dot(w0))
    l_output = activfun_sigmoid(l1.dot(w1))
    l_output_error = y-l_output
    l_output_delta = l_output_error
activfun_sigmoid(l_output, True)
l1_error = l_output_delta.dot(w1.T)
l1_delta = l1_error*activfun_sigmoid(l1, deriv=True)
w1 = w1+2*l1.T.dot(l_output_delta)
w0 = w0+2*l_input.T.dot(l1_delta)
print("Output after training")
print(l_output)
print("loss:\n"+str(np.mean(np.square(y-l_output))))
