import pickle
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_data(df,chunk_size=1):
    X = []
    for i in tqdm(df.index.unique()):
        x = torch.FloatTensor(df.loc[i].values)
        size = x.shape[0]
        if chunk_size > 1:
            size = int(size/chunk_size)
        x = torch.chunk(x,chunk_size)
        X.extend(x)
    X = torch.stack(X, 0)
    return X

def load_dataset(dataset_name = "lorenz",file_path=r'.\data',chunk_size=1):

    with open(os.path.join(file_path,f"{dataset_name}/{dataset_name}_train_inputs.pickle"), "rb") as handle:
        train_df = pickle.load(handle)

    with open(os.path.join(file_path,f"{dataset_name}/{dataset_name}_test_inputs.pickle"), "rb") as handle:
        test_df = pickle.load(handle)

    X_train = load_data(train_df,chunk_size)
    X_test = load_data(test_df,chunk_size)

    return X_train,X_test


class differential_dataset(Dataset):

    def __init__(self,X,horizon):

        self.X = X
        self.horizon = horizon
        self.D = X.shape[-1]
        self.T = X.shape[1]-self.horizon
        self.mu = torch.tensor([torch.mean(X[:,:,i]) for i in range(self.D)])
        self.std = torch.tensor([torch.std(X[:,:,i]) for i in range(self.D)])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is int:
            idx = [idx]

        start = torch.randint(low=0,high=self.T+1,size=(len(idx),))
        windows = torch.tensor([list(range(i,i+self.horizon)) for i in start]).unsqueeze(-1).repeat(1,1,self.D)
        x = torch.gather(self.X[idx],1,windows).squeeze()

        return x

if __name__ == "__main__":
    X_train,X_test = load_dataset(chunk_size=1)
    print(X_train.shape)
    print(X_test.shape)

    dataset = differential_dataset(X_train,10)
    print("hello:")


