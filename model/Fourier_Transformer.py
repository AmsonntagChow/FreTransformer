import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from layers.FTrans import FTrans

class FTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ftrans = FTrans(configs)
        self.configs = configs
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.pre_length = configs.pre_length
        self.enc_in = configs.enc_in
        self.seq_length = configs.seq_length
        self.frequency_size = self.embed_size
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.w21 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size, self.embed_size))
        self.b21 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w22 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size, self.embed_size))
        self.b22 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w23 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size,self.embed_size))
        self.b23 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size ))
        self.w24 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size, self.embed_size))
        self.b24 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.fc = nn.Sequential(
                nn.Linear(self.enc_in*self.embed_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.enc_in)
            )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y
   

    def forward(self, x, dec_inp):
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        

        # FFT
        x = torch.fft.rfft(x, dim=2, norm='ortho') # conduct fourier transform along time dimension

        # x = x.reshape(B, N, L//2+1)

        #
        #B*N*L ==> B*L*N
        x = x.permute(0, 2, 1)

        #B*L*N ==>L_*N
        # x = x.reshape(-1, N)

        # fourier Transformer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = FTrans(self.configs).to(device)
        
        dec_inp = torch.fft.rfft(dec_inp, dim=1, norm='ortho')
        pre = self.configs.pre_length//2+1
        label = self.configs.label_len//2+1
        dec_inp = torch.zeros_like(dec_inp[:, -pre:, :])
        dec_inp = torch.cat([dec_inp[:, :label, :], dec_inp], dim=1).to(device)
        # x = model(x,None, dec_inp, None)
        x = self.ftrans(x, None, dec_inp, None)

        x = x.permute(0, 2, 1)
        # x = x.reshape(B, N, L//2+1)
        x = torch.fft.irfft(x, n=L, dim=2, norm="ortho")


        x = x.permute(0, 2, 1)
        return x

    def viz_adj(self, x):
        x = x.permute(0, 2, 1, 3) # [B, L, N, D]
        x = x[:, :, 170:200, :]
        x = torch.mean(x, dim=1)
        adp = torch.mm(x[0], x[0].T)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp * (1 / np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(data=df, cmap="Oranges")
        plt.savefig("./emb" + '.pdf')
