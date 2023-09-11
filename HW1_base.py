tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'   # path to testing data

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])] # //表只留下商數 ::x表取每個第xth的值
    figure(figsize=(6, 4)) # 畫布大小
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train') # Loss_record['train']長度6975=775(epoch)*9 tab:red是深紅的意思
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev') # Loss_record['dev']長度775=即每個epoch都去算，另外最後model是存在第574個epoch，即從第575個epoch開始連續200個都沒辦法讓dev_mse < min_mse
    plt.ylim(0.0, 3.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title(f'Learning curve of {title}')
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
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
    plt.scatter(targets, preds, c='r', alpha=0.5) # alpha是透明度
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

class COVID19Dataset(Dataset): # COVID19Dataset繼承所有Dataset的屬性和能力
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp)) # list後面放iterable的參數即可把資料list化
            data = np.array(data[1:])[:, 1:].astype(float) # 把str改成float
        
        if not target_only:
            feats = list(range(93)) # 刪除id後共93個columns，用feats控管想要輸入的變數量
        else:
            # Using 40 states & 2 tested_positive features (indices = 57 & 75)，其他columns都不要了
            feats = list(range(40))+[57,75]
            pass # pass表不做任何事，程式繼續執行，常配合TODO使用，用在還沒想到要寫甚麼時提醒自己回來寫，因為空著會error所以寫個pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats] # 只留下想要的columns
            self.data = torch.FloatTensor(data) # 把data轉成floattensor格式(單精度浮點張量)，因為是testing所以全部都切成data
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1] # 取最後數來第一筆
            data = data[:, feats] 
            
            # Splitting training data into train & dev sets
            if mode == 'train': # 只切training set的10%當validation set
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        # \ 表換行符號，程式會接續進行，通常用在程式碼太長為了提升易讀性時
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True) # dim=0表示沿著第一維做mean、std

        self.dim = self.data.shape[1] # 把shape中的第二個值(即維度)存到dim變數中

        print(f'Finished reading the {mode} set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.dim})')

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
    
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False): # n_jobs表示要平行讀取的資料數
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False, # 當是traing的時候才shuffle，其他則否 # 要不要放棄最後一個batch，當他小於batch size時
        num_workers=n_jobs, pin_memory=True) # 要不要用cuda加速的記憶體                         # Construct dataloader
    return dataloader

class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__() # 對NeuralNet的parent class初始化

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1) # 把大小是1的維度刪除(不重要的維度)，x是input tensor，通過network算出output，當output是single dimentions時很常用
    
    def cal_loss(self, pred, target, lambda_L2=0.001): # 對prediction和target算loss
        ''' Calculate loss '''
        L2_regularization=0
        for w in model.parameters():
            L2_regularization += torch.norm(w, p=2) ** 2 # 做L2正規化，可以用norm=2(每個weight平方相加開根號)再平方回去，因為我們不是要算norm不用把結果開根號
        return self.criterion(pred, target) + L2_regularization*lambda_L2

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(  # 使用getattr()的寫法，使我們能動態選擇optimizer，因為config['optimizer']是字串，所以不能直接寫torch.optim.config['optimizer']
        model.parameters(), **config['optim_hparas'])  # **是Unpacking Operator(開箱運算子)，用於dictionary，把字典中的鍵值對解包當作參數傳遞給optimizer constructor

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader (tr_set是一個dataloader) 本處每個for迴圈會跑2430/270=9次(dataset/batchsize)
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y, config["lambda_L2"])  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item()) # 更新次數=loss_record長度=training epochs * dataset/batchsize =776*9 (每一次迴圈更新一次，每個epoch跑9個迴圈) =7000

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print(f'Saving model (epoch = {epoch+1 :4d}, loss = {min_mse :.4f})') # 4d:整數空4格，.4f:浮點數保留小數點後4位，epoch從0開始算所以表達的時候要加1
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path，model.state_dict()會回傳字典物件，裡面存了model的狀態(學好的參數、layer等)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print(f'Finished training after {epoch} epochs') # epoch不用+1 因為while後面就加了
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y, config['lambda_L2'])  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss 因為mse_loss是計算一個batch的平均loss，所以要乘上batch size才能還原全部的loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/ 在現有工作區創建一個資料夾，exist_ok=True，若已存在也也不會跳錯誤
target_only = True                    # Using 40 states & 2 tested_positive features

# 把hyper-parameters都蒐集成一個dictionary
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    "lambda_L2": 0.001,
    'save_path': 'models/model.pth'  # your model will be saved here
}

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model # 因為前面訓練的模型已經不需要了所以刪除
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model checkpoint，map_location='cpu'是把載入的模組放在cpu，ckpt是一個state_dic
model.load_state_dict(ckpt) # 把train好的weight丟入model中
plot_pred(dv_set, model, device)  # Show prediction on the validation set

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print(f'Saving results to {file}')
    with open(file, 'w', newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive']) # 寫入id, tested_positive當表頭
        for i, p in enumerate(preds): # enumerate會自動將iterable的資料型態編號，因此可配合for迴圈一起取出編號和值
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv

