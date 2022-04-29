import torch.optim

from models import Lusch
from data_generator import load_dataset,differential_dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    koopman_dim = 64
    hidden_dim = 500
    input_dim = 3
    delta_t = 0.01


    epochs = 300
    lr = 1e-3
    horizon = 72;
    batch_size = 128
    load_chkpt = True
    chkpt_filename = "fixed_matrix"
    start_epoch = 1
    device="cuda"

    n = 10

    model = Lusch(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    X_train, X_test = load_dataset(chunk_size=1)

    X_train_recon = X_train[:, :-horizon, :]; X_test_recon = X_test[:, :-horizon, :]
    X_forecast_train = X_train[:, -horizon:, :] ; X_forecast_test = X_test[:, -horizon:, :]

    train_dl = DataLoader(differential_dataset(X_train_recon, horizon), batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon, horizon), batch_size=batch_size)

    model.mu = train_dl.dataset.mu.to(device)
    model.std = train_dl.dataset.std.to(device)

    save_every = 5

    if load_chkpt:
        print("LOAD CHECKPOINTS")
        state_dicts = torch.load(chkpt_filename+".pth")
        model.load_state_dict(state_dicts['model'])


    with torch.inference_mode():
        model.eval()

        # x_recon_hat = model.recover(        model.embed(X_test_recon[[n],:,:].to(device))     ).cpu().squeeze(0)
        x_recon_hat = model(X_test_recon[[n],:,:].to(device)).cpu().squeeze(0)

        # print(F.mse_loss(model(X_test_recon.cuda()),X_test_recon.cuda()))
        x_ahead_hat = model.recover(model.koopman_operator(model.embed(X_test_recon[[n],[-1],:].to(device).unsqueeze(0)),horizon)).cpu().squeeze(0)



        mpl.use('Qt5Agg')
        plt.figure(figsize=(20, 10))
        #     for i in range(3):

        plt.plot(np.arange(X_test_recon.shape[1]), X_test_recon[n, :, :], '--')
        plt.plot(np.arange(x_recon_hat.shape[0]), x_recon_hat)
        plt.plot(X_train_recon.shape[1] + np.arange(horizon), x_ahead_hat.cpu(), 'r.')

        plt.xlabel("Time (n)", fontsize=20)
        plt.ylabel("State", fontsize=20)
        plt.legend(["x", "y", "z", "$x_{reconstructed}$", "$y_{reconstructed}$", "$z_{reconstructed}$", "Forecasted"],
                   fontsize=20)
        plt.show()



        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X_test[n, :, 0], X_test[n, :, 1], X_test[n, :, 2], 'k-')  # c=np.linspace(0,1,Time_Length))
        ax.plot3D(x_recon_hat[:, 0], x_recon_hat[:, 1], x_recon_hat[:, 2], 'b*')
        ax.plot3D(x_ahead_hat[:, 0], x_ahead_hat[:, 1], x_ahead_hat[:, 2], 'rx')
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel(r'$Z$', fontsize=20)
        plt.legend(["Actual", "Reconstruction", "Forecasted"])
        plt.show()