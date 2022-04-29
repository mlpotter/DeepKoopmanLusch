import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def koopman_loss(x,model,Sp,T,alpha1=2,alpha2=1e-10):
    # Sp < T
    MAX_T = max(Sp,T)

    encoder_x = model.embed(x)
    recover_x = model.recover(encoder_x)


    koopman_stepped = model.koopman_operator(encoder_x[:,[0],:],MAX_T)
    recover_koopman = model.recover(koopman_stepped[:,:(Sp-1),:])


    reconstruction_inf_loss = torch.norm(x-recover_x,p=float('inf'),dim=[-2,-1]).mean()
    prediction_inf_loss = torch.norm(x[:,1:Sp,:]-recover_koopman,p=float('inf'),dim=[-2,-1]).mean()


    lin_loss = F.mse_loss(encoder_x[:,1:T,:],koopman_stepped[:,:(T-1),:])
    pred_loss = F.mse_loss(recover_koopman,x[:,1:Sp,:],)
    reconstruction_loss = F.mse_loss(recover_x,x)
    inf_loss = reconstruction_inf_loss + prediction_inf_loss

    loss = alpha1*(pred_loss + reconstruction_loss) + lin_loss + alpha2*inf_loss
    return loss

def prediction_loss(x_recon,x_ahead,model):
    # Sp < T
    with torch.inference_mode():
        model.eval()
        Y = model.koopman_operator(model.embed(x_recon[:,[-1],:]),x_ahead.shape[1])
        prediction_loss = F.mse_loss(x_ahead,model.recover(Y))

    return prediction_loss

