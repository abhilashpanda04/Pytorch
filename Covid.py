import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rc
from pylab import rcParams
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from torch import nn,optim
import torch.nn.functional as F

%config InlineBackend.figure_format='retina'
%matplotlib inline

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)




import wget
url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
wget.download(url)

df=pd.read_csv("time_series_covid19_confirmed_global.csv")

df=df.iloc[:,4:]
df

df.isnull().sum()

daily_cases=df.sum(axis=0)
daily_cases.head()

daily_cases.index=pd.to_datetime(daily_cases.index)

daily_cases.head

plt.plot(daily_cases)
plt.title("cummulative daily cases")


daily_cases=daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

daily_cases.head()

plt.plot(daily_cases)


daily_cases.shape

#preprocessing
test_data_size=115
train_data=daily_cases[:-test_data_size]
test_data=daily_cases[-test_data_size:]
train_data.head()

test_data.head()

train_data.shape
test_data.shape

scaler=MinMaxScaler()
scaler=scaler.fit(np.expand_dims(train_data,axis=1))
train_data=scaler.transform(np.expand_dims(train_data,axis=1))
test_data=scaler.transform(np.expand_dims(test_data,axis=1))

def sliding_windows(data,seq_lenght):
    xs=[]
    ys=[]
    for i in range(len(data)-seq_lenght-1):
        x=data[i:(i+seq_lenght)]
        y=data[i+seq_lenght]
        xs.append(x)
        ys.append(y)

    return np.array(xs),np.array(ys)

seq_length=5

X_train,y_train=sliding_windows(train_data,seq_length)
X_test,y_test=sliding_windows(test_data,seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()



X_train.shape

y_train.shape

X_train[:2]
y_train[:2]


class covidpred(nn.Module):
    def __init__(self,input_dim,hidden_dim,seq_length,num_layer=2):
        super(covidpred,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.seq_lenght=seq_length
        self.num_layer=num_layer

        self.lstm=nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layer,
        dropout=0.5)

        self.linear=nn.Linear(in_features=hidden_dim,out_features=1)
    def reset_hidden_state(self):
        self.hidden=(
            torch.zeros(self.num_layer,self.seq_lenght,self.hidden_dim),
            torch.zeros(self.num_layer,self.seq_lenght,self.hidden_dim))
    def forward(self,input):
        lstm_out,self.hidden=self.lstm(input.view(len(input),self.seq_lenght,-1),self.hidden)
        y_pred=self.linear(lstm_out.view(self.seq_lenght,len(input),self.hidden_dim)[-1])
        return y_pred

    def train_model(model,train_data,train_labels,test_data=None,test_labels=None):
        loss_fn=torch.nn.MSELoss(reduction="sum")
        optimiser=torch.optim.Adam(model.parameters(),lr=1e-3)
        num_epochs=180
        train_hist=np.zeros(num_epochs)
        test_hist=np.zeros(num_epochs)

        for t in range(num_epochs):
            model.reset_hidden_state()
            y_pred=model(X_train)
            loss=loss_fn(y_pred.float(),y_train)

            if test_data is not None:
                with torch.no_grad():
                    y_test_pred = model(X_test)
                    test_loss = loss_fn(y_test_pred.float(), y_test)
                    test_hist[t] = test_loss.item()
                if t % 10 == 0:
                    print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
            elif t % 10 == 0:
                    print(f'Epoch {t} train loss: {loss.item()}')
            train_hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return model.eval(), train_hist, test_hist

model=covidpred(1,512,seq_length=seq_length,num_layer=2)
model,train_hist,test_hist = train_model(model,X_train,y_train,X_test,y_test)

plt.plot(train_hist,label='Training loss')

plt.plot(test_hist,label='Test loss')

plt.ylim((0,5))

plt.legend();

#predicting the daily cases

with torch.no_grad():
    test_seq=X_test[:1]
    preds=[]
    for _ in range(len(X_test)):
        y_test_pred=model(test_seq)
        pred=torch.flatten(y_test_pred).item()
        preds.append(pred)

        new_seq=test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()


true_cases=scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(),axis=0)).flatten()

predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

plt.plot(
  daily_cases.index[:len(train_data)],
  scaler.inverse_transform(train_data).flatten(),
  label='Historical Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
  true_cases,
  label='Real Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
  predicted_cases,
  label='Predicted Daily Cases'
)

plt.legend();


scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))

all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))

all_data.shape

X_all, y_all = sliding_windows(all_data, seq_length)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model = covidpred(1, 512,seq_length=seq_length,num_layer=2)
model, train_hist, _ = train_model(model, X_all, y_all)

DAYS_TO_PREDICT = 30

with torch.no_grad():
  test_seq = X_all[:1]
  preds = []
  for _ in range(DAYS_TO_PREDICT):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()

daily_cases.index[-1]


predicted_index = pd.date_range(
  start=daily_cases.index[-1],
  periods=DAYS_TO_PREDICT + 1,
  closed='right'
)

predicted_cases = pd.Series(
  data=predicted_cases,
  index=predicted_index
)

plt.plot(predicted_cases, label='Predicted Daily Cases')
plt.legend();
