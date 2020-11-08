import torch
import pandas as pd

#reading the dataset
df=pd.read_csv("diabetes.csv")

df.head()

df.isnull().sum()

import seaborn as sns

import numpy as np

# df["Outcome"]=np.where(df["Outcome"]==1,"Diabetic","No-Diabetic")
#
# sns.pairplot(df,hue="outcome")

from sklearn.model_selection import train_test_split

X=df.drop("Outcome",axis=1).values
y=df["Outcome"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#import all necessary libraries

import torch.nn as nn
import torch.nn.functional as F


#dependent feature should be always float tensor
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


y_train

df.shape

class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x


#instanstiate the model
torch.manual_seed(20)

model=ANN_Model()

model.parameters

##define backward propogation

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


epochs=600
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss)
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


import matplotlib.pyplot as plt
# %matplotlib.inline


plt.plot(range(epochs),final_losses)
plt.ylabel('loss')
plt.xlabel('epoch')

predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax())
        print(y_pred.argmax().item())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
cm

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")



from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,predictions)
score

torch.save(model,'diabetes.pt')
