import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats

import math
import torch
import torch.nn as nn
import numpy as np
import neptune.new as neptune

torch.manual_seed(37)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(True)

###########################
### Aritra Implementation
###########################
class PermutedGruCell(nn.Module):
    def __init__(self, hidden_size, bias):
        super().__init__()
        """
        For each element in the input sequence, each layer computes the following
        function:
        Gru Math
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} +  (W_{hn}(r_t*  h_{(t-1)})+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}
        """
        self.hidden_size = hidden_size
        self.W_ir = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_iz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_in = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.reset_parameters()


    def reset_parameters(self):
        for w in self.parameters():
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def forward(self, x, lower, hidden=None):
        # x is B, input_size
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size).to(device)
        W_ir = self.W_ir * lower.to(device)
        W_hr = self.W_hr * lower.to(device)
        W_iz = self.W_iz * lower.to(device)
        W_hz = self.W_hz * lower.to(device)
        W_in = self.W_in * lower.to(device)
        W_hn = self.W_hn * lower.to(device)

        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        r_t = sigmoid(torch.matmul(x, W_ir) + torch.matmul(hidden, W_hr)).to(device)
        z_t = sigmoid(torch.matmul(x, W_iz) + torch.matmul(hidden, W_hz)).to(device)
        n_t = tanh(torch.matmul(x, W_in) + torch.matmul(r_t * hidden, W_hn)).to(device)
        hy = (hidden * z_t + (1.0 - z_t) * n_t).to(device)
        # print("hy: \n", hy)
        return hy

def is_permuation_matrix(x):
     x = np.asanyarray(x.to("cpu"))
     print("Is it a permutation matrix?")
     print("x.ndim: ", x.ndim)
     print("x.shape[0] == x.shape[1]: ", x.shape[0] == x.shape[1])
     print("(x.sum(axis=0) == 1.).all(): ", (x.sum(axis=0) == 1.).all())
     print("(x.sum(axis=1) == 1.).all(): ", (x.sum(axis=1) == 1.).all())
     print("((x == 1.) | (x == 0.)).all()): ", ((x == 1.) | (x == 0.)).all())
     return (x.ndim == 2 and x.shape[0] == x.shape[1] and
             (x.sum(axis=0) == 1.).all() and 
             (x.sum(axis=1) == 1.).all() and
             ((x == 1.) | (x == 0.)).all())

class PermutationMatrix(nn.Module):
    def __init__(self, input_size, temperature=100, unroll=1000):
        super().__init__()
        self.unroll, self.temperature = unroll, temperature
        self.matrix = nn.Parameter(torch.empty(input_size, input_size)).to(device)
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(5))
        self.lower = torch.tril(torch.ones(input_size, input_size)).to(device)

    def forward(self, verbose=False):
        # print("matrix debug: ")
        # print(self.matrix)
        # print("isNan: ", torch.sum(torch.isnan(self.matrix)))
        matrix_shape = self.matrix.shape[0]
        max_row = torch.max(self.matrix, dim=1).values.reshape(matrix_shape, 1).to(device)
        ones = torch.ones(matrix_shape).reshape(1, matrix_shape).to(device)
        matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix))).to(device)
        # matrix = torch.exp(self.temperature * (self.matrix - torch.matmul(max_row, ones)))
        # print(matrix)
        for _ in range(self.unroll):
            matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
        output_lower = torch.matmul(torch.matmul(matrix, self.lower), matrix.t()).t().to(device)
        ideal_matrix_order = matrix.data.argmax(dim=1, keepdim=True)
        new_matrix = torch.zeros_like(matrix)
        new_matrix.scatter_(
            1, ideal_matrix_order, torch.ones_like(ideal_matrix_order).float().to(device)
        )
        causal_order = [(idx, int(d[0])) for idx, d in enumerate(ideal_matrix_order)]
        causal_order.sort(key=lambda x: x[1])
        causal_order = [d[0] for d in causal_order]
        if verbose:
            row_sum = round(float(torch.median(torch.sum(matrix, dim=1)[0])), 2)
            col_sum = round(float(torch.median(torch.sum(matrix, dim=0)[0])), 2)
            row_max = round(float(torch.median(torch.max(matrix, dim=1)[0])), 2)
            col_max = round(float(torch.median(torch.max(matrix, dim=0)[0])), 2)
            print(
                "Median Row Sum: {}, Col Sum: {} Row Max: {} Col Max: {}".format(
                    row_sum, col_sum, row_max, col_max
                )
            )
            # print("Permutation Matrix\n", matrix.data.numpy().round(1))
            # print(
            #     "Permuted Lower Triangular Matrix\n",
            #     output_lower.t().data.numpy().round(1),
            # )
            print("Ideal Permutation Matrix\n", new_matrix.data)
            is_permuted = is_permuation_matrix(new_matrix.data)
            print("is_permuted?: ", is_permuted)
            print(
                "Ideal Lower Triangular\
                    matrix\n",
                torch.matmul(torch.matmul(new_matrix, self.lower), new_matrix.t()),
            )
            # print("Causal Order\n", causal_order)

        return output_lower


class PermutedGru(nn.Module):
    def __init__(
        self,
        hidden_size,
        bias=False,
        num_layers=1,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.cell = PermutedGruCell(hidden_size=hidden_size, bias=False)
        self.batch_first = batch_first
        self.permuted_matrix = PermutationMatrix(hidden_size)

    def forward(self, input_, lengths=None, hidden=None):
        # input_ is of dimensionalty (T, B, hidden_size, ...)
        # lenghths is B,
        dim = 1 if self.batch_first else 0
        lower = self.permuted_matrix(verbose=False) # Verbose Flag
        outputs = []
        for x in torch.unbind(input_, dim=dim):  # x dim is B, I
            hidden = self.cell(x, lower, hidden)
            outputs.append(hidden.clone())

        hidden_states = torch.stack(outputs)  # T, B, H
        last_states = []
        if lengths is None:
            lengths = [len(input_)] * len(input_[0])
        for idx, l in enumerate(lengths):
            last_states.append(hidden_states[l - 1, idx, :])
        last_states = torch.stack(last_states)
        return hidden_states, last_states


class PermutedDKT(nn.Module):
    def __init__(self, n_constructs):
        super().__init__()
        self.gru = PermutedGru(n_constructs, batch_first=False)
        self.n_constructs = n_constructs
        self.output_layer = nn.Linear(1, 1)

    def forward(self, construct_input, labels):
        # Input shape is T, B
        # Input[i,j]=k at time i, for student j, construct k is attended
        # label is T,B 0/1
        T, B = construct_input.shape
        input = torch.zeros(T, B, self.n_constructs).to(device)
        # mask = labels.apply_(lambda x: 0 if x == 0 else 1)
        mask = torch.zeros(labels.shape).to(device)
        for i, row in enumerate(labels):
            for j, col in enumerate(row):
                if labels[i][j] != 0:
                    mask[i][j] = 1
        input.scatter_(2, construct_input.unsqueeze(2), labels.unsqueeze(2).float())
        # print("Non-zero element in input: ", torch.count_nonzero(input))
        labels = torch.clamp(labels, min=0)
        hidden_states, _ = self.gru(input)

        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        # print("init state: ", init_state)
        # print("hidden state: ", hidden_states)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[1:, :, :]
        # output = self.output_layer(shifted_hidden_states.unsqueeze(3)).squeeze(3)
        # print("Shifted hidden states: ", shifted_hidden_states)
        # print("DEBUG SHAPE: ", shifted_hidden_states.shape)
        output = torch.sigmoid(self.output_layer(shifted_hidden_states.unsqueeze(3)).squeeze(3))
        output = torch.gather(output, 2, construct_input.unsqueeze(2)).squeeze(2)
        pred = (output >= 0.5).float()
        # acc = torch.mean((pred == labels).float())
        acc = torch.mean((pred == torch.clamp(labels, min=0).float()).float())
        # cc_loss = nn.BCEWithLogitsLoss()
        cc_loss = nn.BCELoss(reduction='none')
        # cc_loss = nn.BCELoss()
        # loss = cc_loss(output, labels.float())
        loss = cc_loss(output, torch.clamp(labels, min=0).float())
        loss = loss * mask

        return loss, acc

###########################
### Create TrainingDataset
###########################
class TrainingDataset(Dataset):
    def __init__(self, constructs, labels, tot_construct_set):
        self.constructs = constructs
        self.labels = labels
        # Put number of construct here?
        self.n_constructs = len(tot_construct_set)
        self.unique_construct_list = list(tot_construct_set)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        construct = self.constructs[idx]
        label = self.labels[idx]
        sample = {"Construct": construct, "Label": label}
        return sample

def createDataset(filename: str):
    lessons_df = pd.read_csv(filename)
    checkin_df = lessons_df[lessons_df['Type'] == 'Checkin']
    simple_df = checkin_df.iloc[:, [2, 5, 9]] 
    simple_df.loc[simple_df["IsCorrect"] == 0, "IsCorrect"] = -1 

    num_of_questions = stats.mode(simple_df["UserId"]).count[0]

    tot_construct_set = set()
    tot_construct_list = list()
    tot_label_list = list()
    
    for user, user_info in simple_df.groupby('UserId'):

        constructs = user_info["ConstructId"].values.tolist() # [C]
        labels = user_info["IsCorrect"].values.tolist() # [C]

        tot_construct_set.update(constructs)

        num_of_constructs = len(constructs)
        pad_needed = num_of_questions - num_of_constructs # [P = Q - C]

        constructs += [0] * pad_needed # [Q]
        labels += [0] * pad_needed # [Q]

        tot_construct_list.append(constructs)
        tot_label_list.append(labels)
    tot_construct_set.add(0)
    tot_serialized_construct_list = list()
    unique_construct_list = list(tot_construct_set)
    # print(unique_construct_list)
    for constructs in tot_construct_list:
        # print("constructs: ", constructs)
        serialized_constructs = list(map(lambda x: unique_construct_list.index(x), constructs))
        tot_serialized_construct_list.append(serialized_constructs)
    TD = TrainingDataset(tot_serialized_construct_list, tot_label_list, tot_construct_set)

    return TD

def main():
    print("start")
    training_set = createDataset('data/Task_3_dataset/checkins_lessons_checkouts_training.csv')
    # training_set = createDataset('data/Task_3_dataset/tmp_training.csv')
    dkt = PermutedDKT(n_constructs=training_set.n_constructs).to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     dkt = nn.DataParallel(dkt)
    # dkt = dkt.to(device)

    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(dkt.parameters(), lr=learning_rate)
    n_epochs = 100
    best_loss = 100.0
    best_accuracy = 0.0
    best_epoch = 0.0
    run = neptune.init(
    project="phdprojects/challenge",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MTYwYjA3Zi01NmNhLTQ4YWMtOWFmMy0zMjdmZDliOGE4YzAifQ==",
    capture_hardware_metrics = False
    )
    for epoch in range(n_epochs): # loop over the dataset multiple times
        print("Epoch ", epoch)
        train_loss=[]
        train_accuracy=[]
        for i, data in enumerate(training_loader, 0):
            constructs = data['Construct']
            labels = data['Label']
            optimizer.zero_grad()
            loss, acc = dkt(torch.stack(constructs).to(device), torch.stack(labels).to(device))
            train_accuracy.append(acc)
            train_loss.append(loss.mean().item())
            loss.mean().backward()
            optimizer.step()
        if (sum(train_loss)/len(train_loss) < best_loss):
            # print("========"*10)
            best_loss = sum(train_loss)/len(train_loss)
            best_accuracy = sum(train_accuracy)/len(train_accuracy)
            best_epoch = epoch
            run['best_loss'] = best_loss
            run['best_accuracy'] = best_accuracy
            run['best_epoch'] = best_epoch
            torch.save(dkt.state_dict(), './model/best_dkt.pt') 
            # print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.mean().item():.4f}, Train Accuracy: {sum(train_accuracy)/len(train_accuracy):.2f}")
            # print("========"*10)
        run['loss'].log(sum(train_loss)/len(train_loss))
        run['accuracy'].log(sum(train_accuracy)/len(train_accuracy))

    torch.save(dkt.state_dict(), './model/final_dkt.pt') 




if __name__ == "__main__":
    main()