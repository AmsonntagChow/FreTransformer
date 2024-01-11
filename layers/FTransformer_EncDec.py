import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_fgcn=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_fgcn = d_fgcn or 4 * d_model
        self.d_model = d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_fgcn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_fgcn, out_channels=d_model, kernel_size=1)
        self.scale = 0.02
        self.sparsity_threshold = 0.01
        self.w41 = nn.Parameter(
            self.scale * torch.randn(2, d_model, d_fgcn))
        self.b41 = nn.Parameter(self.scale * torch.randn(2, d_fgcn))
        self.w42 = nn.Parameter(
            self.scale * torch.randn(2, d_fgcn, d_model))
        self.b42 = nn.Parameter(self.scale * torch.randn(2, d_model))
        self.linear1 = nn.Linear(d_model, d_fgcn)
        self.linear2 = nn.Linear(d_fgcn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        # dropout_mask = (torch.rand_like(new_x.real) < self.dropout.p).float() / (1.0 - self.dropout.p)
        # x_real = new_x.real * dropout_mask
        # x_imag = new_x.imag * dropout_mask
        x_real = self.dropout1(new_x.real)
        x_imag = self.dropout1(new_x.imag)

        x_dropout = torch.complex(x_real, x_imag)

        x = x + x_dropout

        x_real = self.norm1(x.real)
        x_imag = self.norm1(x.imag)
        
        x = torch.complex(x_real, x_imag)

        y = x

        y1_real = (
                torch.einsum('...i,io->...o', y.real, self.w41[0]) - \
                torch.einsum('...i,io->...o', y.imag, self.w41[1]) + \
                self.b41[0]
        )

        y1_imag = (
                torch.einsum('...i,io->...o', y.imag, self.w41[0]) + \
                torch.einsum('...i,io->...o', y.real, self.w41[1]) + \
                self.b41[1]
        )
        y1 = torch.stack([y1_real, y1_imag], dim=-1)
        y1 = F.softshrink(y1, lambd=self.sparsity_threshold)
        y1 = torch.view_as_complex(y1)
        # y_real = self.activation(self.linear1(y.real))
        # y_imag = self.activation(self.linear1(y.imag))

        # dropout_mask1 = (torch.rand_like(y_real) < self.dropout.p).float() / (1.0 - self.dropout.p)
    
        # y_real = y_real * dropout_mask1
        # y_imag = y_imag * dropout_mask1
        y2_real = self.dropout2(y1.real)
        y2_imag = self.dropout2(y1.imag)



        # y_real = self.linear2(y_real)
        # y_imag = self.linear2(y_imag)
        y3_real = (
                torch.einsum('...i,io->...o', y2_real, self.w42[0]) - \
                torch.einsum('...i,io->...o', y2_imag, self.w42[1]) + \
                self.b42[0]
        )

        y3_imag = (
                torch.einsum('...i,io->...o', y2_imag, self.w42[0]) + \
                torch.einsum('...i,io->...o', y2_real, self.w42[1]) + \
                self.b42[1]
        )

        # dropout_mask2 = (torch.rand_like(y_real) < self.dropout.p).float() / (1.0 - self.dropout.p)


        # y_real = y_real * dropout_mask2
        # y_imag = y_imag * dropout_mask2
        y4_real = self.dropout3(y3_real)
        y4_imag = self.dropout3(y3_imag)
        
        y = torch.complex(y4_real,y4_imag)
        

        z_real = self.norm2(x.real + y.real)
        z_imag = self.norm2(x.imag + y.imag)

        z = torch.complex(z_real,z_imag)

        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        return z, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x_real = self.norm(x.real)
            x_imag = self.norm(x.imag)
            x = torch.complex(x_real,x_imag)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_fgcn=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_fgcn = d_fgcn or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_fgcn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_fgcn, out_channels=d_model, kernel_size=1)
        self.scale = 0.02
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, d_model, d_fgcn))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, d_fgcn))
        self.w4 = nn.Parameter(
            self.scale * torch.randn(2, d_fgcn, d_model))
        self.b4 = nn.Parameter(self.scale * torch.randn(2, d_model))
        self.sparsity_threshold = 0.01
        self.linear1 = nn.Linear(d_model, d_fgcn)
        self.linear2 = nn.Linear(d_fgcn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x_ = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0]

        # dropout_mask = (torch.rand_like(x_.real) < self.dropout.p).float() / (1.0 - self.dropout.p)
        # x_real = x_.real * dropout_mask
        # x_imag = x_.imag * dropout_mask
        x_real = self.dropout1(x_.real)
        x_imag = self.dropout1(x_.imag)

        x_ = torch.complex(x_real, x_imag)
        x = x + x_
        # x = x + self.dropout(self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask,
        #     tau=tau, delta=None
        # )[0])

        x_real = self.norm1(x.real)
        x_imag = self.norm1(x.imag)
        x = torch.complex(x_real, x_imag)
        # x = self.norm1(x)

        x_ = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0]

        # dropout_mask = (torch.rand_like(x_.real) < self.dropout.p).float() / (1.0 - self.dropout.p)
        # x_real = x_.real * dropout_mask
        # x_imag = x_.imag * dropout_mask

        x_ = torch.complex(x_real, x_imag)
        x = x + x_
        # x = x + self.dropout(self.cross_attention(
        #     x, cross, cross,
        #     attn_mask=cross_mask,
        #     tau=tau, delta=delta
        # )[0])

        x_real = self.norm1(x.real)
        x_imag = self.norm1(x.imag)
        
        x = torch.complex(x_real, x_imag)

        y = x


        # y_real = self.activation(self.linear1(y.real))
        # y_imag = self.activation(self.linear1(y.imag))
        y1_real = (
                torch.einsum('...i,io->...o', y.real, self.w3[0]) - \
                torch.einsum('...i,io->...o', y.imag, self.w3[1]) + \
                self.b3[0]
        )

        y1_imag = (
                torch.einsum('...i,io->...o', y.imag, self.w3[0]) + \
                torch.einsum('...i,io->...o', y.real, self.w3[1]) + \
                self.b3[1]
        )
        y1 = torch.stack([y1_real, y1_imag], dim=-1)
        y1 = F.softshrink(y1, lambd=self.sparsity_threshold)
        y1 = torch.view_as_complex(y1)

        # dropout_mask1 = (torch.rand_like(y_real) < self.dropout.p).float() / (1.0 - self.dropout.p)
    
        # y_real = y_real * dropout_mask1
        # y_imag = y_imag * dropout_mask1
        y2_real = self.dropout2(y1.real)
        y2_imag = self.dropout2(y1.imag)

        # y_real = self.linear2(y_real)
        # y_imag = self.linear2(y_imag)
        y3_real = (
                torch.einsum('...i,io->...o', y2_real, self.w4[0]) - \
                torch.einsum('...i,io->...o', y2_imag, self.w4[1]) + \
                self.b4[0]
        )

        y3_imag = (
                torch.einsum('...i,io->...o', y2_imag, self.w4[0]) + \
                torch.einsum('...i,io->...o', y2_real, self.w4[1]) + \
                self.b4[1]
        )

        # dropout_mask2 = (torch.rand_like(y_real) < self.dropout.p).float() / (1.0 - self.dropout.p)


        # y_real = y_real * dropout_mask2
        # y_imag = y_imag * dropout_mask2
        y4_real = self.dropout3(y3_real)
        y4_imag = self.dropout3(y3_imag)

        y = torch.complex(y4_real,y4_imag)

        # y = torch.complex(y_real,y_imag)
        

        z_real = self.norm2(x.real + y.real)
        z_imag = self.norm2(x.imag + y.imag)

        z = torch.complex(z_real,z_imag)

        # y = x = self.norm2(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        return z


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x_real = self.norm(x.real)
            x_imag = self.norm(x.imag)
            x = torch.complex(x_real,x_imag)

        if self.projection is not None:
            x_real = self.projection(x.real)
            x_imag = self.projection(x.imag)
            x = torch.complex(x_real,x_imag)
        return x
