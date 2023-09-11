# 資料集解釋:delphi對美國40個州每天在fb上進行Covid-19抽樣調查，從2020/04開始共進行67~68天(所以每一個州1的總數是67或68個)，問卷包含有無covid-19相關症狀、陽性與否、有無戴口罩、自覺心理健康程度等，最後推估該州可能有多少百分比的人擁有那些屬性，我們使用的資料的各項屬性即為delphi最後的推估結果
# 任務目標:依照每個州過去3天的調查結果以及過去兩天的陽性比例 推估 第三天該州的陽性比例 (所以資料會重複，某一列第二天的資料即是下一列第一天的資料)

tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import csv
import os
import random

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed=random.randint(0,1000000)
# myseed=955517
print(f'seed is: {myseed}')
torch.backends.cudnn.deterministic = True # 在Pytorch中使用GPU時，底層的CuDNN庫負責加速神經網路的運算。CuDNN庫在每次運行前會自動選擇一種算法來執行，然而由於算法涉及到隨機性，同一段代碼在多次運行中可能會得到不同的結果，因此設為True表示運行CuDNN庫之前，會固定CuDNN庫的隨機數種子，返回的算法將是確定的，即預設算法。如果再設置porch的隨機種子為固定值的話，可以保證每次運行網路的時候相同輸入的輸出是固定的。
torch.backends.cudnn.benchmark = False # 若設成True會讓程式在開始時花費一點額外時間，為整個網路的每個卷積層搜索最適合它的卷積實現算法，進而實現網路的加速。適用場景是網路結構固定（不是動態變化的），網路的輸入形狀（包括 batch size，圖片大小，輸入的通道）是不變的，其實也就是一般情況下都比較適用。反之，如果卷積層的設置一直變化，將會導致程式不停地做優化，反而會耗費更多的時間。
np.random.seed(myseed) # 為numpy的random generator設定隨機種子
random_state=myseed 
torch.manual_seed(myseed) # 固定在cpu上運算的隨機數種字
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(myseed) # 固定在gpu上運算的隨機數種字

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# with open("covid.train.csv") as file:
#     data=list(csv.reader(file))
#     # data=np.array(data)[1:][:,1:].astype(float)

# filter=list(range(40))+[57,75]
# print(data[:,filter])
# print(data[:,-1])

# dataP=pd.read_csv("covid.train.csv")
# print(dataP.columns)
# print(dataP.shape)
# print(dataP.head(3))
# print(dataP[["tested_positive.2","shop"]])

class COVID19Dataset(Dataset):
    def __init__(self,
                 path,
                 mean,
                 std,
                 mode='train',
                 target_only=False
                 ):
        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not config["target_only"]:
            feats = list(range(93))
        else:
            feats = list(range(40))+[75, 57, 42, 60, 78, 43, 61, 79, 40, 58, 76, 41, 59, 77] # list(range(40))+[57,75]

        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats] 
            
            # if mode == 'train':
            #     indices = [i for i in range(len(data)) if i % 10 != 0]
            # elif mode == 'dev':
            #     indices = [i for i in range(len(data)) if i % 10 == 0]
            
            indices_tr, indices_dev = train_test_split([i for i in range(data.shape[0])], test_size=config["test_size"], random_state=random_state) # 輸入要切分的資料、test(or val)的比例、隨機種子(確保每次拆的結果都一樣)
            if self.mode == 'train':
                indices = indices_tr
            elif self.mode == 'dev':
                indices = indices_dev # scikit-learn的分割資料集套件可以隨機分，若用原本程式分就沒有隨機了
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        if mode == 'train': # 因為照理說test、validation的資料集應該要使用train的資料來做正規化，所以要拆開來計算
            self.mean=self.data[:, 40:].mean(dim=0, keepdim=True)
            self.std=self.data[:, 40:].std(dim=0, keepdim=True)
        else:
            self.mean=mean
            self.std=std
        self.data[:, 40:] = (self.data[:, 40:] - self.mean) / self.std
        
        self.dim = self.data.shape[1]

        print(f'Finished reading the {mode}ing set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.dim})')
    
    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

# print(COVID19Dataset(tr_path).__getitem__(0)) # 看training的第一筆資料
# print(COVID19Dataset(tr_path).__len__()) # 因為有90%的資料拿去validation 所以training是2430

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False, mean=None, std=None):
    dataset = COVID19Dataset(path, mean, std, mode=mode, target_only=target_only)
    if mode=="train": # 把train算出來的mean、std留下來
        mean=dataset.mean
        std=dataset.std
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False, 
        num_workers=n_jobs, pin_memory=True)                     
    return dataloader, mean, std

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__() 

        self.net = nn.Sequential(     # 簡單的資料集不用太複雜的network，用兩層，neurons也不用太多
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1) # 因為輸出會是[[...], [...],...]，所以把第2個維度(長度是1，因為最後對每個x只輸出一個y)squeeze掉
    
    def cal_loss(self, pred, target, lambda_L2=0.001):
        L2_regularization=0
        for w in model.parameters():
            L2_regularization += torch.norm(w, p=2)
        return torch.sqrt(self.criterion(pred, target)) + L2_regularization*lambda_L2 # 改用平方根誤差，loss降比較快

# model=NeuralNet(10)
# x=torch.randn(10,10)
# print(model(x))
    
def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                          
        for x, y in tr_set:                     
            optimizer.zero_grad()               
            x, y = x.to(device), y.to(device) 
            pred = model(x)                    
            mse_loss = model.cal_loss(pred, y, config["lambda_L2"]) 
            mse_loss.backward()                 
            optimizer.step()                    
            loss_record['train'].append(mse_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print(f'Saving model (epoch = {epoch+1 :4d}, loss = {min_mse :.4f})')
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break

    print(f'Finished training after {epoch} epochs')
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()                                
    total_loss = 0
    for x, y in dv_set:                         
        x, y = x.to(device), y.to(device)       
        with torch.no_grad():                   
            pred = model(x)                     
            mse_loss = model.cal_loss(pred, y, config['lambda_L2'])  
        total_loss += mse_loss.detach().cpu().item() * len(x)  
    total_loss = total_loss / len(dv_set.dataset)              

    return total_loss

def test(tt_set, model, device):
    model.eval()                                
    preds = []
    for x in tt_set:                            
        x = x.to(device)                       
        with torch.no_grad():                  
            pred = model(x)                     
            preds.append(pred.detach().cpu())  
    preds = torch.cat(preds, dim=0).numpy() # 輸出會是9個tensor，每個tensor有270筆pred，所以可直接沿著columns接在一起
    return preds

config = {
    'n_epochs': 10000,                
    'batch_size': 270,              
    'optimizer': 'SGD',             
    'optim_hparas': {                
        'lr': 0.001,                
        'momentum': 0.9             
    },
    'early_stop': 500,               
    "lambda_L2": 0.001,
    'save_path': 'models/model.pth',
    "target_only": True,
    "test_size":0.2
}

def plot_learning_curve(loss_record, title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])] 
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train') 
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev') 
    plt.ylim(0.0, 3.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title(f'Learning curve of {title}')
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

device = get_device()                
os.makedirs('models', exist_ok=True)

tr_set, tr_mean, tr_std = prep_dataloader(tr_path,'train', config['batch_size'], target_only=config["target_only"])
dv_set, mean_none, std_none = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=config["target_only"], mean=tr_mean, std=tr_std)
tt_set, mean_none, std_none = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=config["target_only"], mean=tr_mean, std=tr_std)

model = NeuralNet(tr_set.dataset.dim).to(device)
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  
model.load_state_dict(ckpt) 
plot_pred(dv_set, model, device)  

def save_pred(preds, file):
    print(f'Saving results to {file}')
    with open(file, 'w', newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive']) 
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)
# print(preds)
save_pred(preds, 'pred.csv')