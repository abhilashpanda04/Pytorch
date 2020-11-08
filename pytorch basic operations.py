import torch
torch.__version__
torch.cuda.is_available()
torch.cuda.get_device_name()

import numpy as np

a=np.random.rand(9)
arr=np.array(a)
arr.dtype


tensors=torch.from_numpy(arr)

tensors
tensors[:2]
tensors[3:4]

tensor_arr=torch.tensor(arr)
tensor_arr
arr1=np.random.rand(10)
arr2=np.random.rand(10)


b=tensor_arr=torch.tensor(arr2)
a=tensor_arr=torch.tensor(arr1)

torch.add(a,b)
torch.add(a,b,out=c)
c=torch.zeros(4)
c.dtype
c
torch.add(a,b).sum()

#dot product and multiplication

a=torch.tensor([3,4,5])
b=torch.tensor([7,8,9])

a.mul(b)

a.dot(b)

x=torch.tensor([[1,2,3],[3,4,5]])
y=torch.tensor([[3,7],[9,3],[4,5]])
torch.matmul(x,y)

torch.mm(x,y)
x@y
x=torch.tensor(4.0,requires_grad=True)
x
y=x**2
y

##back propagation
y.backward()

print(x.grad)

l=[[2.,3.,5.,6.,7.],[7.,6.,4.,5.,6.],[7.,.6,.5,.3,.4]]

torch_input=torch.tensor(l,requires_grad=True)

torch_input
y=torch_input**3+torch_input**2

y
z=y.sum()
z

z.backward()

torch_input.grad
