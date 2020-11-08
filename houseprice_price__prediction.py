import torch
import pandas as pd

df=pd.read_csv('houseprice.csv',usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                         "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()

df.shape

df.head(10)

df.info()
for i in df.columns:
    print("column name {} and unique values are {}".format(i,len(df[i].unique())))

import datetime
a=datetime.datetime.now().year

df["Total years"]=a-df["YearBuilt"]

df["Total years"]
df.drop("YearBuilt",axis=1,inplace=True)

df.columns

cat_features=["MSSubClass", "MSZoning", "Street", "LotShape"]
out="SalePrice"

from sklearn.preprocessing import LabelEncoder
lbl_encoder={}
lbl_encoder["MSSubClass"]=LabelEncoder()
lbl_encoder["MSSubClass"].fit_transform(df["MSSubClass"])

for feature in cat_features:
    lbl_encoder[feature]=LabelEncoder()
    df[feature]=lbl_encoder[feature].fit_transform(df[feature])

df.head()
import numpy as np
cat_features=np.stack([df["MSSubClass"],df["MSZoning"],df["Street"],df["LotShape"]],1)
cat_features
cat_features=torch.tensor(cat_features,dtype=torch.int64)

cont_features=[]

for i in df.columns:
    if i in ["MSSubClass","MSZoning","Street","LotShape","SalePrice"]:
        pass
    else:
        cont_features.append(i)

cont_features

#stacking continneous variable to tensor
cont_values=np.stack([df[i].values for i in cont_features],axis=1)
cont_values=torch.tensor(cont_values,dtype=torch.float)
cont_values


cont_values.dtype

y=torch.tensor(df["SalePrice"].values,dtype=torch.float).reshape(-1,1)



cat_features.shape,cont_values.shape,y.shape
df.shape

#embedding

cat_dims=[len(df[cols].unique()) for cols in ["MSSubClass", "MSZoning", "Street", "LotShape"]]

cat_dims

#the output dims should be set base on input dims
embedding_dim=[(x,min(50,(x+1)//2)) for x in cat_dims]

#preprocessing step
embedding_dim


import torch.nn as nn
import torch.functional as F
embed_representation=nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim])
embed_representation
cat_features
cat_featuresz=cat_features[:4]
cat_featuresz

pd.set_option('display.max_rows', 500)
embedding_val=[]
for i,e in enumerate(embed_representation):
    embedding_val.append(e(cat_features[:,i]))

embedding_val

z=torch.cat(embedding_val,1)
z
dropout=nn.Dropout(.4)

final_embed=dropout(z)
final_embed
class FeedForwardNN(nn.Module):
    def __init__(self,embedding_dim,n_cont,out_sz,layers,p=0.5):
        super().__init__()
        self.embeds=nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim])
        self.emb_drop=nn.Dropout(p)
        self.bn_cont=nn.BatchNorm1d(n_cont)

        layerlist=[]
        n_emb=sum((out for inp,out in embedding_dim))
        n_in=n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in,i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            layerlist.append
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))

        self.layers=nn.Sequential(*layerlist)
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

len(cont_features)

torch.manual_seed(100)
model=FeedForwardNN(embedding_dim,len(cont_features),1,[100,50],p=0.4)


model

loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
df.shape

cont_values.shape

batch_size=1200
test_size=int(batch_size*0.15)
train_catagorical=cat_features[:batch_size-test_size]
test_catagorical=cat_features[batch_size-test_size:batch_size]
train_cont=cont_values[:batch_size-test_size]
test_cont=cont_values[batch_size-test_size:batch_size]
y_train=y[:batch_size-test_size]
y_test=y[batch_size-test_size:batch_size]

len(train_catagorical),len(test_catagorical),len(train_cont),len(test_cont),len(y_train),len(y_test)

epochs=6000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model(train_catagorical,train_cont)
    loss=torch.sqrt(loss_function(y_pred,y_train))
    final_losses.append(loss)
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


import matplotlib.pyplot as plt
plt.plot(range(epochs), final_losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch');
y_pred=""
with torch.no_grad():
    y_pred=model(test_catagorical,test_cont)
    loss=torch.sqrt(loss_function(y_pred,y_test))
    print('RMSE: {}'.format(loss))


data_verify=pd.DataFrame(y_test.tolist(),columns=["Test"])
data_predicted=pd.DataFrame(y_pred.tolist(),columns=["Prediction"])

data_predicted



final_output=pd.concat([data_verify,data_predicted],axis=1)
final_output['Difference']=final_output['Test']-final_output['Prediction']
final_output.head()

torch.save(model,"houseprice.pt")

torch.save(model1.state_dict(),"houseweight.pt")
emb_size=[(15,8),(5,3),(2,1),(4,2)]
model1=FeedForwardNN(emb_size,5,1,[100,50],p=0.4)

model1.load_state_dict(torch.load("houseweight.pt"))


model1.eval()
