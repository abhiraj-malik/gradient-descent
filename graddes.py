
#
import numpy as np
x = np.random.randn(15, 1)
y = 2*x + np.random.randn()


w = 0.0
b = 0.0
learning_rate = 0.01

def descent(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    number = x.shape[0]
    # loss = (y-(wx+b))**2
    for xi, yi in zip(x,y):
        dldw += -2*xi*(y-(w*xi+b))
        dldb += -2*(y-(w*xi+b))

    w = w-learning_rate*(1/number)*dldw
    b = w-learning_rate*(1/number)*dldb
    return w,b

for epoch in range(500):
    w,b = descent(x,y,w,b,learning_rate)
    y1 = w*x + b
    loss = np.divide(np.sum((y - y1)**2, axis=0), x.shape[0])
    print(f'{epoch} Loss is {loss}, parameters w:{w}, b:{b}')
print(x,y)



