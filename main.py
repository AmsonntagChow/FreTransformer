
import argparse
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import Dataset_Fin
from model.Fourier_Transformer import FTransformer
import time
import os
import numpy as np
import pandas as pd
from utils.utils import save_model, load_model, evaluate
import random
import matplotlib.pyplot as plt

fix_seed = 9999
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='=.=.=.=.=.=.===.=')
parser.add_argument('--data', type=str, default='Fin', help='data set')

parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.00013, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.5)


# Phase 12
parser.add_argument('--start_train', type=str, default="2020-11-18")
parser.add_argument('--end_train', type=str, default="2021-09-18")
parser.add_argument('--start_vali', type=str, default="2021-09-19")
parser.add_argument('--end_vali', type=str, default="2021-11-18")
parser.add_argument('--start_backtest', type=str, default="2021-11-19")
parser.add_argument('--end_backtest', type=str, default="2022-05-01")


parser.add_argument('--device', type=str, default='cuda:0', help='device')

##FTransformer below

parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--number_frequency', type=int, default='1', help='number of frequency')
parser.add_argument('--seq_length', type=int, default=20, help='input length')
parser.add_argument('--pre_length', type=int, default=20, help='predict length')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_fgcn', type=int, default=32, help='dimension of fcn')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--c_out', type=int, default=6, help='output size')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--label_len', type=int, default=48, help='start token length')

args = parser.parse_args()
print(f'Training configs: {args}')

# create output dir
result_train_file = os.path.join('output', args.data, 'train')
result_test_file = os.path.join('output', args.data, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

# data set
data_parser = {
    'ECG':{'root_path':'data/ECG_data.csv', 'type':'1'},
    'Fin':{'root_path':'data/^GSPC_candles_D.csv', 'type':'1'},
    'COVID':{'root_path':'data/covid.csv', 'type':'1'},
}

# data process
if args.data in data_parser.keys():
    data_info = data_parser[args.data]

train_set = Dataset_Fin(root_path=data_info['root_path'], flag='train', seq_len=args.seq_length, pre_len=args.pre_length, type=data_info['type'], start_train=args.start_train, end_train= args.end_train, start_vali= args.start_vali, end_vali = args.end_vali, start_backtest= args.start_backtest, end_backtest= args.end_backtest)
test_set = Dataset_Fin(root_path=data_info['root_path'], flag='test', seq_len=args.seq_length, pre_len=args.pre_length, type=data_info['type'], start_train=args.start_train, end_train= args.end_train, start_vali= args.start_vali, end_vali = args.end_vali, start_backtest= args.start_backtest, end_backtest= args.end_backtest)
val_set = Dataset_Fin(root_path=data_info['root_path'], flag='val', seq_len=args.seq_length, pre_len=args.pre_length, type=data_info['type'], start_train=args.start_train, end_train= args.end_train, start_vali= args.start_vali, end_vali = args.end_vali, start_backtest= args.start_backtest, end_backtest= args.end_backtest)

train_dataloader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

test_dataloader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

val_dataloader = DataLoader(
    val_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FTransformer(args).to(device)
my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

forecast_loss = nn.MSELoss(reduction='mean').to(device)
# forecast_loss = nn.SmoothL1Loss(reduction='mean').to(device)
# forecast_loss = nn.L1Loss(reduction='mean').to(device)

total_params = 0
for name, parameter in model.named_parameters():
    if not parameter.requires_grad: continue
    param = parameter.numel()
    total_params += param
print(f"Total Trainable Params: {total_params}")

all_gradients = {name: [] for name, _ in model.named_parameters()}

def get_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradients[name] = parameter.grad.data.cpu().numpy()
    return gradients

def validate(model, vali_loader):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        # decoder input
        dec_inp = y
        # dec_inp = torch.zeros_like(y[:, -args.pre_length:, :]).float()
        # dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        ##here for Transformer
        forecast = model(x, dec_inp)
        # forecast = model(x)
        # y = y.permute(0, 2, 1).contiguous().to(device)
        loss = forecast_loss(forecast, y)
        # print(f"Loss before backward: {loss.item()}")
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.array(preds)
    trues = np.array(trues)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)    

    
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f};MSE {score[3]:7.9f};ic {score[4]:7.9f};rank_ic {score[5]:7.9f}.train.')
    model.train()
    return loss_total/cnt

def test():
    result_test_file = 'output/Fin/train'
    model = load_model(result_test_file, 99)
    model.eval()
    preds = []
    trues = []
    for index, (x, y) in enumerate(test_dataloader):
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        # decoder input
        dec_inp = torch.zeros_like(y[:, -args.pre_length:, :]).float()
        dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        ##here for Transformer
        forecast = model(x, dec_inp)
        # print(y.shape)##
        # y = y.permute(0, 2, 1).contiguous().to(device)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.array(preds)
    trues = np.array(trues)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    # preds_reshaped = preds.reshape(-1, 5)
    # print(preds_reshaped.shape)
    # trues_reshaped = trues.reshape(-1, preds.shape[-1])
    # t = test_dataloader.dataset
    # t.inverse()
    # df1 = pd.DataFrame(p)
    # df2 = pd.DataFrame(t)
    # df1.to_csv('output/Fin/preds.csv')
    # df2.to_csv('output/Fin/trues.csv')
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f};MSE {score[3]:7.9f};ic {score[4]:7.9f};rank_ic {score[5]:7.9f}.test.')

if __name__ == '__main__':

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    train_losses = []
    val_losses = []
#train
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_gradients = {name: [] for name, _ in model.named_parameters()}
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            cnt += 1

            y = y.float().to(device)
            x = x.float().to(device)

            # decoder input
            dec_inp = y

            ##here for Fre-Transformer
            forecast = model(x, dec_inp)[0]
            
            
            # forecast = model(x)
            # y = y.permute(0, 2, 1).contiguous().to(device)
            loss = forecast_loss(forecast, y)
            # print(f"Loss before backward: {loss.item()}")
            loss.backward()
            # for name, parameter in model.named_parameters():
            #     if parameter.grad is not None:
            #         print(f"Gradient for {name} is not None")  # This should print
            #     else:
            #         print(f"Gradient for {name} is None")
            batch_gradients = get_gradients(model)
            # print(batch_gradients)
            for name, grad in batch_gradients.items():
                epoch_gradients[name].append(grad)
            
            my_optim.step()
            loss_total += float(loss)
        
        for name in all_gradients.keys():
            all_gradients[name].append(np.mean(epoch_gradients[name], axis=0))

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_dataloader)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))
        train_losses.append(loss_total / cnt)
        val_losses.append(val_loss)
        save_model(model, result_train_file, epoch)
    
    #test
    test()
    



    #loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.clf()
    # #gradient plot
    # print(all_gradients.keys())
    # #dict_keys(['embeddings', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias'])
    # param_name_to_plot = 'fc.0.weight' 
    # print(all_gradients[param_name_to_plot])
    # # Make sure to stack along the first dimension (each array represents an epoch)
    # param_gradients = np.stack(all_gradients[param_name_to_plot], axis=0)

    # # Calculate the mean along the second axis (collapse all the parameter gradients)
    # mean_gradient_magnitudes = np.mean(np.abs(param_gradients), axis=1)

    # # Plot the mean gradient magnitude for each epoch
    # plt.figure(figsize=(10, 5))  # Create a new figure to avoid overlaps
    # plt.plot(mean_gradient_magnitudes, label=f'Gradient Magnitude for {param_name_to_plot}')
    # plt.title('Mean Gradient Magnitudes Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Gradient Magnitude')
    # plt.legend()
    # plt.savefig("gradient_plot_FreT.png")
    # plt.close()  # Close the plot to avoid conflicts if plotting again later


