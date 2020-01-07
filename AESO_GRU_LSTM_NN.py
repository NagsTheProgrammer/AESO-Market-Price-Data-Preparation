# Framework based on Gated Recurrent Unit (GRU) with PyTorch by Gabriel Loye, July 22, 2019

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook
# from ipywidgets import IntProgress
from sklearn.preprocessing import MinMaxScaler
import csv
import winsound

# Define data root directory
offline_total = "/Data Subsets/Offline/Total Offline/"
offline_large = "/Data Subsets/Offline/Large (gt 400 MW)/"
offline_medium = "/Data Subsets/Offline/Medium (gt 100 MW)/"
offline_wo_other = "/Data Subsets/Offline/Offline excl Other/"

online_total = "/Data Subsets/Online/Total Online/"
online_large = "/Data Subsets/Online/Large (gt 400 MW)/"
online_medium = "/Data Subsets/Online/Medium (gt 100 MW)/"
online_wo_other = "/Data Subsets/Online/Online excl Other/"

# choose data subset
subset = online_large
if subset == offline_large:
    category = "Offline"
elif subset == online_large:
    category = "Online"
data_dir = "D:/1. Programming/Pycharm/AESO GRU Predictor" + subset
os.chdir("./Prepped Data/")

# choose test percentage
testPercentage = 0.42

# Visualise how our data looks
if category == "Offline":
    pd.read_csv(data_dir + "2019-8-14_Battle 3_offline.csv").head()
elif category == "Online":
    pd.read_csv(data_dir + "2016-10-17_Keephills 1_online.csv").head()

# The scaler objects will be stored in this dictionary so that our output test data from the model can be re-scaled during evaluation
label_scalers = {}

train_x = []
test_x = {}
test_y = {}

for file in tqdm_notebook(os.listdir(data_dir)):
    # print(file)
    # # Skipping the files we're not using
    # if file[-4:] != ".csv" or file == "pjm_hourly_est.csv":
    #     continue

    # Store csv file in a Pandas DataFrame
    df = pd.read_csv('{}/{}'.format(data_dir, file), parse_dates=[0])
    # Processing the time data into suitable input formats
    df['hour'] = df.apply(lambda x: x['Date Time'].hour, axis=1)
    df['dayofweek'] = df.apply(lambda x: x['Date Time'].dayofweek, axis=1)
    df['month'] = df.apply(lambda x: x['Date Time'].month, axis=1)
    df['dayofyear'] = df.apply(lambda x: x['Date Time'].dayofyear, axis=1)
    df = df.sort_values("Date Time").drop("Date Time", axis=1)

    # Scaling the input data
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(df.values)
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
    label_scalers[file] = label_sc

    # Define lookback period and split inputs/labels
    lookback = 12
    inputs = np.zeros((len(data) - lookback, lookback, df.shape[1]))
    labels = np.zeros(len(data) - lookback)

    for i in range(lookback, len(data)):
        inputs[i - lookback] = data[i - lookback:i]
        labels[i - lookback] = data[i, 0]
    inputs = inputs.reshape(-1, lookback, df.shape[1])
    labels = labels.reshape(-1, 1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(testPercentage * len(inputs))
    if len(train_x) == 0:
        train_x = inputs[:-test_portion]
        train_y = labels[:-test_portion]
    else:
        train_x = np.concatenate((train_x, inputs[:-test_portion]))
        train_y = np.concatenate((train_y, labels[:-test_portion]))
    test_x[file] = (inputs[-test_portion:])
    test_y[file] = (labels[-test_portion:])

batch_size = 48
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#  good till here

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=5, model_type="GRU", drop_prob = 0.2):
    # Setting common hyperparameters
    temp = iter(train_loader)
    input_dim = next(temp)[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.time()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.time() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE


def returnArrayVals(inputArray):
    temp_array = []

    length = len(inputArray)

    for i in range(length):
        temp_array.append(inputArray[i][0])

    return temp_array

def writeToCSV(predictedList, actualList, filename, sMAPE):
    path = "/1. Programming/Pycharm/AESO GRU Predictor/Output Data/" + category + "/"
    filename = path + filename

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    with open(filename, mode='w', newline='') as outputFile:
        outputWriter = csv.writer(outputFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        outputWriter.writerow(["Actual", "Predicted", "sMAPE = {}".format(sMAPE)])
        for e in range(len(actualList)):
            outputWriter.writerow([actualList[e], predictedList[e]])


# output param
model = "both"
hp = "lr"
if model == "both":
    if hp == "":
        predictionArray = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    else:
        predictionArray = [2, 2, 2, 2]

epoch = 4
lr = 0.001
lr_gru = 0.001
lr_lstm = 0.001
dp_gru = 0.3
dp_lstm = 0.2
dp = 0.2
if hp == "lr":
    lr = [0.0001, 0.001, 0.01, 0.1]
elif hp == "epoch":
    epoch = [5, 10, 15, 20]
elif hp == "dp":
    dp = [0.1, 0.2, 0.3, 0.4]
filename = [model + "_" + category + "_" + "_Large_lr[0.0001].csv", model + "_" + category + "_" + "_Large_lr[0.001].csv", model + "_" + category + "_" + "_Large_lr[0.01].csv", model + "_" + category + "_" + "_Large_lr[0.1].csv"]



# plot figure
plt.figure(figsize=(14,10))

for x in range(4):
    # if hp == "lr":
    #     filename[x] = model + "_" + category + "_" + "_Large_" + hp + "[{}].csv".format(lr[x])
    # elif hp == "epoch":
    #     filename[x] = model + "_" + category + "_" + "_Large_" + hp + "[{}].csv".format(epoch[x])
    # elif hp == "dp":
    #     filename[x] = model + "_" + category + "_" + "_Large_" + hp + "[{}].csv".format(dp[x])
    # else:
    #     filename[x] = model + "_" + category + "_" + "_Large_" + "epoch[{}]".format(epoch) + "lr[{}]".format(lr) + ".csv"

    if model == "lstm":
        if hp == "lr":
            trainingModel = train(train_loader, lr[x], EPOCHS = epoch, model_type="LSTM")
        elif hp == "epoch":
            trainingModel = train(train_loader, lr, EPOCHS = (epoch[x]), model_type="LSTM")
        else:
            trainingModel = train(train_loader, lr, EPOCHS = epoch, model_type="LSTM")
    elif model == "gru":
        if hp == "lr":
            trainingModel = train(train_loader, lr[x], EPOCHS = epoch, model_type="GRU")
        elif hp == "epoch":
            trainingModel = train(train_loader, lr, EPOCHS = (epoch[x]), model_type="GRU")
        else:
            trainingModel = train(train_loader, lr, EPOCHS = epoch, model_type="GRU")
    elif model == "both":
        if hp == "lr":
            trainingModel_gru = train(train_loader, lr[x], EPOCHS = epoch, model_type="GRU")
            trainingModel_lstm = train(train_loader, lr[x], EPOCHS = epoch, model_type="LSTM,")
            outputs_gru, targets, sMAPE_gru = evaluate(trainingModel_gru, test_x, test_y, label_scalers)
            outputs_lstm, targets, sMAPE_lstm = evaluate(trainingModel_lstm, test_x, test_y, label_scalers)
            plt.subplot(2, 2, x + 1)
            plt.plot(outputs_gru[predictionArray[x]][:], color="g", label="GRU (lr = {})".format(lr[x]))
            plt.plot(outputs_lstm[predictionArray[x]][:], color="r", label="LSTM (lr = {})".format(lr[x]))
        elif hp == "dp":
            trainingModel_gru = train(train_loader, lr, EPOCHS = epoch, model_type="GRU", drop_prob = dp[x])
            trainingModel_lstm = train(train_loader, lr, EPOCHS  =epoch, model_type="LSTM,", drop_prob = dp[x])
            outputs_gru, targets, sMAPE_gru = evaluate(trainingModel_gru, test_x, test_y, label_scalers)
            outputs_lstm, targets, sMAPE_lstm = evaluate(trainingModel_lstm, test_x, test_y, label_scalers)
            plt.subplot(2, 2, x + 1)
            plt.plot(outputs_gru[predictionArray[x]][:], color="g", label="GRU (dp = {})".format(dp[x]))
            plt.plot(outputs_lstm[predictionArray[x]][:], color="r", label="LSTM (dp = {})".format(dp[x]))
        elif hp == "epoch":
            trainingModel_gru = train(train_loader, lr, EPOCHS = epoch[x], model_type="GRU", drop_prob = dp)
            trainingModel_lstm = train(train_loader, lr, EPOCHS = epoch[x], model_type="LSTM,", drop_prob = dp)
            outputs_gru, targets, sMAPE_gru = evaluate(trainingModel_gru, test_x, test_y, label_scalers)
            outputs_lstm, targets, sMAPE_lstm = evaluate(trainingModel_lstm, test_x, test_y, label_scalers)
            plt.subplot(2, 2, x + 1)
            plt.plot(outputs_gru[predictionArray[x]][:], color="g", label="GRU (dp = {})".format(dp[x]))
            plt.plot(outputs_lstm[predictionArray[x]][:], color="r", label="LSTM (dp = {})".format(dp[x]))
        else:
            trainingModel_gru = train(train_loader, lr_gru, EPOCHS = epoch, model_type="GRU", drop_prob = dp_gru)
            trainingModel_lstm = train(train_loader, lr_lstm, EPOCHS = epoch, model_type="LSTM,", drop_prob = dp_lstm)
            print("GRU SMAPE")
            outputs_gru, targets, sMAPE_gru = evaluate(trainingModel_gru, test_x, test_y, label_scalers)
            print("LSTM SMAP")
            outputs_lstm, targets, sMAPE_lstm = evaluate(trainingModel_lstm, test_x, test_y, label_scalers)
            plt.subplot(2, 2, x + 1)
            plt.plot(outputs_gru[predictionArray[x]][:], color="g", label="GRU (SMAPE = {0:.2f}%)".format(sMAPE_gru*100))
            plt.plot(outputs_lstm[predictionArray[x]][:], color="r", label="LSTM (SMAPE = {0:.2f}%)".format(sMAPE_lstm*100))

    if model != "both":
        outputs, targets, sMAPE = evaluate(trainingModel, test_x, test_y, label_scalers)
        temp = returnArrayVals(outputs)
        temp_target = returnArrayVals(targets)

        plt.subplot(2, 2, x+1)
        if hp == "epoch":
            plt.plot(outputs[predictionArray[x]][:], color="g", label="Predicted (epoch = {})".format(epoch[x]))
        elif hp == "lr":
            plt.plot(outputs[predictionArray[x]][:], color="g", label="Predicted (lr = {})".format(lr[x]))
        else:
            plt.plot(outputs[predictionArray[x]][:], color="g", label="Predicted (SMAPE = {0:.2f}%)".format(sMAPE*100))

    plt.plot(targets[predictionArray[x]][:], color="b", label="Actual")
    plt.ylabel('Pool Price ($/MW)')
    # if x == 3:
    plt.xlabel('Predicted Data Points')
    plt.legend()



    # writeToCSV(outputs[predictionArray[x]], targets[predictionArray[x]], filename[x], sMAPE)

    # axes = plt.gca()
    # axes.set_xlim([300, 400])
    # axes.set_ylim([-5, 1000])

# plt.plot(temp_target, color="b", label="Actual")
# plt.plot(temp0, color="g", label="Predicted (lr = 0.0001)")
# plt.plot(temp1, color="r", label="Predicted (lr = 0.001)")
# plt.plot(temp2, color="m", label="Predicted (lr = 0.01)")
# plt.plot(temp3, color="c", label="Predicted (lr = 0.1)")
# # writeToCSV(temp, temp_target, filename[x], sMAPE)
#
# plt.ylabel('Electricity Price ($/MW)')
# plt.xlabel('Successive Data Points Scaled to Range (300, 400)')
# plt.legend()
#
# axes = plt.gca()
# axes.set_xlim([300, 400])
# axes.set_ylim([-5, 1000])



# #figure 1
#
# lr = 0.0001
# if (model == "lstm"):
#     lstm_model = train(train_loader, lr, model_type="LSTM")
#     lstm_outputs, targets, lstm_sMAPE = evaluate(lstm_model, test_x, test_y, label_scalers)
#     temp_lstm = returnArrayVals(lstm_outputs)
#     temp_target = returnArrayVals(targets)
# else:
#     gru_model = train(train_loader, lr, model_type="GRU")
#     gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)
#     temp_gru = returnArrayVals(gru_outputs)
#     temp_target = returnArrayVals(targets)
#
# plt.subplot(2,2,1)
# if (model == "lstm"):
#     plt.plot(temp_lstm, "-o", color="g", label="Predicted")
#     writeToCSV(temp_lstm, temp_target, filename[0])
# else:
#     plt.plot(temp_gru, "-o", color="g", label="Predicted")
#     writeToCSV(temp_gru, temp_target, filename[0])
# plt.plot(temp_target, color="b", label="Actual")
# plt.ylabel('Electricity Price ($/MW) for lr = 0.0001 (MW)')
# plt.legend()
#
# axes = plt.gca()
# axes.set_xlim([300, 420])
# axes.set_ylim([-5, 1000])
#
#
#
# #figure 2
#
# lr = 0.001
# if (model == "lstm"):
#     lstm_model = train(train_loader, lr, model_type="LSTM")
#     lstm_outputs, targets, lstm_sMAPE = evaluate(lstm_model, test_x, test_y, label_scalers)
#     temp_lstm = returnArrayVals(lstm_outputs)
#     temp_target = returnArrayVals(targets)
# else:
#     gru_model = train(train_loader, lr, model_type="GRU")
#     gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)
#     temp_gru = returnArrayVals(gru_outputs)
#     temp_target = returnArrayVals(targets)
#
# plt.subplot(2,2,2)
# if (model == "lstm"):
#     plt.plot(temp_lstm, "-o", color="g", label="Predicted")
#     writeToCSV(temp_lstm, temp_target, filename[1])
# else:
#     plt.plot(temp_gru, "-o", color="g", label="Predicted")
#     writeToCSV(temp_gru, temp_target, filename[1])
# plt.plot(temp_target, color="b", label="Actual")
# plt.ylabel('Electricity Price ($/MW) for lr = 0.001 (MW)')
# plt.legend()
#
# axes = plt.gca()
# axes.set_xlim([300, 420])
# axes.set_ylim([-5, 1000])
#
#
#
# #figure 3
#
# lr = 0.01
# if (model == "lstm"):
#     lstm_model = train(train_loader, lr, model_type="LSTM")
#     lstm_outputs, targets, lstm_sMAPE = evaluate(lstm_model, test_x, test_y, label_scalers)
#     temp_lstm = returnArrayVals(lstm_outputs)
#     temp_target = returnArrayVals(targets)
# else:
#     gru_model = train(train_loader, lr, model_type="GRU")
#     gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)
#     temp_gru = returnArrayVals(gru_outputs)
#     temp_target = returnArrayVals(targets)
#
# plt.subplot(2,2,3)
# if (model == "lstm"):
#     plt.plot(temp_lstm, "-o", color="g", label="Predicted")
#     writeToCSV(temp_lstm, temp_target, filename[2])
# else:
#     plt.plot(temp_gru, "-o", color="g", label="Predicted")
#     writeToCSV(temp_gru, temp_target, filename[2])
# plt.plot(temp_target, color="b", label="Actual")
# plt.ylabel('Electricity Price ($/MW) for lr = 0.01 (MW)')
# plt.legend()
#
# axes = plt.gca()
# axes.set_xlim([300, 420])
# axes.set_ylim([-5, 1000])
#
#
#
# #figure 4
#
# lr = 0.1
# if (model == "lstm"):
#     lstm_model = train(train_loader, lr, model_type="LSTM")
#     lstm_outputs, targets, lstm_sMAPE = evaluate(lstm_model, test_x, test_y, label_scalers)
#     temp_lstm = returnArrayVals(lstm_outputs)
#     temp_target = returnArrayVals(targets)
# else:
#     gru_model = train(train_loader, lr, model_type="GRU")
#     gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)
#     temp_gru = returnArrayVals(gru_outputs)
#     temp_target = returnArrayVals(targets)
#
# plt.subplot(2,2,4)
# if (model == "lstm"):
#     plt.plot(temp_lstm, "-o", color="g", label="Predicted")
#     writeToCSV(temp_lstm, temp_target, filename[3])
# else:
#     plt.plot(temp_gru, "-o", color="g", label="Predicted")
#     writeToCSV(temp_gru, temp_target, filename[3])
# plt.plot(temp_target, color="b", label="Actual")
# plt.ylabel('Electricity Price ($/MW) for lr = 0.1 (MW)')
# plt.legend()
#
# axes = plt.gca()
# axes.set_xlim([300, 420])
# axes.set_ylim([-5, 1000])


frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

plt.show()

# change axis range

# axes = plt.gca()
# axes.set_xlim([400, 500])
# axes.set_ylim([-5, 1000])