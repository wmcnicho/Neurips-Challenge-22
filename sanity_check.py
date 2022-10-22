import math
import numpy as np
import pudb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import neptune.new as neptune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(2)

class GroundTruthPermutedGruCell(nn.Module):
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
        # self.W_ir = torch.rand(hidden_size, hidden_size)
        # self.W_hr = torch.rand(hidden_size, hidden_size)
        # self.W_iz = torch.rand(hidden_size, hidden_size)
        # self.W_hz = torch.rand(hidden_size, hidden_size)
        # self.W_in = torch.rand(hidden_size, hidden_size)
        # self.W_hn = torch.rand(hidden_size, hidden_size)

        W_ir = torch.empty(hidden_size, hidden_size, dtype = torch.double)
        W_hr = torch.empty(hidden_size, hidden_size, dtype = torch.double)
        W_iz = torch.empty(hidden_size, hidden_size, dtype = torch.double)
        W_hz = torch.empty(hidden_size, hidden_size, dtype = torch.double)
        W_in = torch.empty(hidden_size, hidden_size, dtype = torch.double)
        W_hn = torch.empty(hidden_size, hidden_size, dtype = torch.double)
        b_ir = torch.randn(hidden_size, dtype = torch.double)
        b_hr = torch.randn(hidden_size, dtype = torch.double)
        b_iz = torch.randn(hidden_size, dtype = torch.double)
        b_hz = torch.randn(hidden_size, dtype = torch.double)
        b_in = torch.randn(hidden_size, dtype = torch.double)
        b_hn = torch.randn(hidden_size, dtype = torch.double)
        nn.init.kaiming_normal_(W_hn, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(W_ir, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(W_hr, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(W_iz, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(W_hz, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(W_in, a=math.sqrt(hidden_size), mode='fan_out')

    def forward(self, x, lower, hidden=None):
        # x is B, input_size
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size).to(device)
        W_ir = self.W_ir * lower
        W_hr = self.W_hr * lower
        W_iz = self.W_iz * lower
        W_hz = self.W_hz * lower
        W_in = self.W_in * lower
        W_hn = self.W_hn * lower
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        mask = torch.zeros(self.hidden_size, dtype = torch.double)
        r_t = sigmoid(torch.matmul(x, W_ir) + self.b_ir * mask + torch.matmul(hidden, W_hr) + self.b_hr )
        z_t = sigmoid(torch.matmul(x, W_iz) + self.b_iz * mask + torch.matmul(hidden, W_hz) + self.b_hz )
        n_t = tanh(torch.matmul(x, W_in) +  self.b_in * mask + r_t * (torch.matmul(hidden, W_hn) + self.b_hn))
        hy = hidden * z_t + (1.0 - z_t) * n_t

        return hy

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
        W_ir = self.W_ir * lower
        W_hr = self.W_hr * lower
        W_iz = self.W_iz * lower
        W_hz = self.W_hz * lower
        W_in = self.W_in * lower
        W_hn = self.W_hn * lower
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        r_t = sigmoid(torch.matmul(x, W_ir) + torch.matmul(hidden, W_hr))
        z_t = sigmoid(torch.matmul(x, W_iz) + torch.matmul(hidden, W_hz))
        n_t = tanh(torch.matmul(x, W_in) + torch.matmul(r_t * hidden, W_hn))
        hy = hidden * z_t + (1.0 - z_t) * n_t
        return hy


class PermutationMatrix(nn.Module):
    def __init__(self, input_size, temperature=100, unroll=20):
        super().__init__()
        self.unroll, self.temperature = unroll, temperature
        self.matrix = nn.Parameter(torch.empty(input_size, input_size))
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(5))
        self.lower = torch.tril(torch.ones(input_size, input_size))

    def forward(self, verbose=False):
        matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
        for _ in range(self.unroll):
            matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
        output_lower = torch.matmul(torch.matmul(matrix, self.lower), matrix.t()).t()
        ideal_matrix_order = matrix.data.argmax(dim=1, keepdim=True)
        new_matrix = torch.zeros_like(matrix)
        new_matrix.scatter_(
            1, ideal_matrix_order, torch.ones_like(ideal_matrix_order).float()
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
            print("Permutation Matrix\n", matrix.data.numpy().round(1))
            print(
                "Permuted Lower Triangular Matrix\n",
                output_lower.t().data.numpy().round(1),
            )
            print("Ideal Permutation Matrix\n", new_matrix.data)
            print(
                "Ideal Lower Triangular\
                    matrix\n",
                torch.matmul(torch.matmul(new_matrix, self.lower), new_matrix.t()),
            )
            print("Causal Order\n", causal_order)

        return output_lower

class GroundTruthPermutationMatrix(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.matrix = torch.eye(input_size)
        self.lower = torch.tril(torch.ones(input_size, input_size))

    def forward(self, verbose=False):
        output_lower = torch.matmul(torch.matmul(self.matrix, self.lower), self.matrix.t()).t()
        # matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
        
        ideal_matrix_order = self.matrix.data.argmax(dim=1, keepdim=True)
        causal_order = [(idx, int(d[0])) for idx, d in enumerate(ideal_matrix_order)]
        causal_order.sort(key=lambda x: x[1])
        causal_order = [d[0] for d in causal_order]
        print("Causal order: ", causal_order)
       
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
        lower = self.permuted_matrix(verbose=False)
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

class GroundTruthPermutedGru(nn.Module):
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
        self.permuted_matrix = GroundTruthPermutationMatrix(hidden_size)

    def forward(self, input_, lengths=None, hidden=None):
        # input_ is of dimensionalty (T, B, hidden_size, ...)
        # lenghths is B,
        W = torch.randn(input_.shape[2]) * input_.shape[0]
        b = torch.randn(1)
        dim = 1 if self.batch_first else 0
        lower = self.permuted_matrix(verbose=False)
        outputs = []
        labels = []
        # for x in torch.unbind(input_, dim=dim):  # x dim is B, I
        #     hidden = self.cell(x, lower, hidden)
        #     labels.append(torch.sigmoid(hidden).clone()*x)
        #     # print("features: \n", x)
        #     # print("labels: \n", torch.sigmoid(hidden).clone()*x)
        #     outputs.append(hidden.clone())
        for t, x in enumerate(torch.unbind(input_, dim=dim)):  # x dim is B, I
            hidden = self.cell(x, lower, hidden)
            labels.append(torch.sigmoid(hidden*W+b).clone()*x)
        labels = torch.stack(labels)
        ans = torch.zeros(labels.shape[0], labels.shape[1])
        for i, xx in enumerate(labels):
            for j, yy in enumerate(xx):
                ans[i][j] = torch.max(yy)
        ans = (ans >= 0.5).int()
        
        return ans

class PermutedDKT(nn.Module):
    def __init__(self, n_concepts):
        super().__init__()
        self.gru = PermutedGru(n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        self.output_layer = nn.Linear(1, 1)

    def forward(self, concept_input, labels):
        # Input shape is T, B
        # Input[i,j]=k at time i, for student j, concept k is attended
        # label is T,B 0/1
        # T, B = concept_input.shape
        B, T = concept_input.shape
        input = torch.zeros(T, B, self.n_concepts)
        concept_input_t = concept_input.t()
        labels_t = labels.t()
        # print("input shape: ", input.shape)
        # print("concept_input_t shape: ", concept_input_t.shape)
        # print("labels_t shape: ", labels_t.shape)
        input.scatter_(2, concept_input_t.unsqueeze(2), labels_t.unsqueeze(2).float())
        labels = torch.clamp(labels, min=0)
        hidden_states, _ = self.gru(input)

        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[:-1:, :, :]
        output = self.output_layer(shifted_hidden_states.unsqueeze(3)).squeeze(3)
        output = torch.gather(output, 2, concept_input_t.unsqueeze(2)).squeeze(2).t()
        # print("output: \n", output)
        pred = (output > 0.0).float()
        acc = torch.mean((pred == labels).float())

        cc_loss = nn.BCEWithLogitsLoss()
        loss = cc_loss(output, labels.float())

        return loss, acc

class GroundTruthPermutedDKT(nn.Module):
    def __init__(self, n_concepts):
        super().__init__()
        self.gru = GroundTruthPermutedGru(n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        # self.output_layer = nn.Linear(1, 1)

    def forward(self, concept_input):
        # Input shape is T, B
        # Input[i,j]=k at time i, for student j, concept k is attended
        # label is T,B 0/1
        B, T = concept_input.shape
        input = torch.zeros(T, B, self.n_concepts)
        concept_input_t = concept_input.t()
        input.scatter_(2, concept_input_t.unsqueeze(2), 1)
        labels = self.gru(input)

        return labels.t()

class TrainingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        # self.unique_constructs = features.unique()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        sample = {"Features": features, "Labels": label}
        return sample

def createDataset(features, labels):
    training_set = TrainingDataset(features, labels)
    return training_set    

def sink_horn(matrix, temperature=100, unroll=20, verbose=False):
    p_matrix = torch.exp(temperature * (matrix - torch.max(matrix)))
    lower = torch.tril(torch.ones(matrix.shape[0], matrix.shape[1]))
    for _ in range(unroll):
        p_matrix = p_matrix / torch.sum(p_matrix, dim=1, keepdim=True)
        p_matrix = p_matrix / torch.sum(p_matrix, dim=0, keepdim=True)
    output_lower = torch.matmul(torch.matmul(p_matrix, lower), p_matrix.t()).t()
    ideal_matrix_order = p_matrix.data.argmax(dim=1, keepdim=True)
    new_matrix = torch.zeros_like(p_matrix)
    new_matrix.scatter_(
        1, ideal_matrix_order, torch.ones_like(ideal_matrix_order).float()
    )
    causal_order = [(idx, int(d[0])) for idx, d in enumerate(ideal_matrix_order)]
    causal_order.sort(key=lambda x: x[1])
    causal_order = [d[0] for d in causal_order]
    if verbose:
        row_sum = round(float(torch.median(torch.sum(p_matrix, dim=1)[0])), 2)
        col_sum = round(float(torch.median(torch.sum(p_matrix, dim=0)[0])), 2)
        row_max = round(float(torch.median(torch.max(p_matrix, dim=1)[0])), 2)
        col_max = round(float(torch.median(torch.max(p_matrix, dim=0)[0])), 2)
        print(
            "Median Row Sum: {}, Col Sum: {} Row Max: {} Col Max: {}".format(
                row_sum, col_sum, row_max, col_max
            )
        )
        print("Permutation Matrix\n", p_matrix.data.numpy().round(1))
        print(
            "Permuted Lower Triangular Matrix\n",
            output_lower.t().data.numpy().round(1),
        )
        print("Ideal Permutation Matrix\n", new_matrix.data)
        print(
            "Ideal Lower Triangular\
                p_matrix\n",
            torch.matmul(torch.matmul(new_matrix, lower), new_matrix.t()),
        )
        print("Causal Order\n", causal_order)

        return causal_order

def test():
    global SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # C (constructs), Q (questions), S (students)
    # C, Q, S = 10, 20, 50
    C, Q, S = 5, 10, 10
    print(f"# of constructs: {C}\n# of questions: {Q}\n# of students: {S}")

    gt_dkt = GroundTruthPermutedDKT(n_concepts=C)
    features = torch.randint(0, C, (S, Q))
    labels = gt_dkt(features)

    print("========="*10)
    print("Data generation")
    print("Features: \n", features)
    print("Labels: \n", labels)
    print("========="*10)
    perm_dataset = True
    # gt_perm = torch.randperm(C)
    gt_perm = list(np.random.permutation(C))
    if perm_dataset:
        print("Ground truth permutation: ", gt_perm)
        for i, xx in enumerate(features):
            for j, yy in enumerate(xx):
                features[i][j] = gt_perm[features[i][j]]
    training_set = createDataset(features, labels)
        # print("Features: \n", features)
    
    dkt = PermutedDKT(n_concepts=C).to(device)

    training_loader = DataLoader(training_set, batch_size=4, shuffle=False)
    optimizer = torch.optim.Adam(dkt.parameters(), lr=0.01)
    #0.01
   
    n_epochs = 500
    best_loss = 100.0
    best_accuracy = 0.0
    best_epoch = 0.0
    # run = neptune.init(
    #     project="phdprojects/challenge",
    #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MTYwYjA3Zi01NmNhLTQ4YWMtOWFmMy0zMjdmZDliOGE4YzAifQ==",
    #     capture_hardware_metrics = False
    # )
    for epoch in range(n_epochs): # loop over the dataset multiple times
        # print("Epoch ", epoch)
        train_loss=[]
        train_accuracy=[]
        for i, data in enumerate(training_loader, 0):
            b_construct = data['Features']
            b_label = data['Labels']
            optimizer.zero_grad()
            loss, acc = dkt(b_construct, b_label)
            train_accuracy.append(acc)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        if (sum(train_loss)/len(train_loss) < best_loss):
            best_loss = sum(train_loss)/len(train_loss)
            best_accuracy = sum(train_accuracy)/len(train_accuracy)
            best_epoch = epoch
            # run['best_loss'] = best_loss
            # run['best_accuracy'] = best_accuracy
            # run['best_epoch'] = best_epoch
            # torch.save(dkt.state_dict(), './model/best_dkt.pt') 
            torch.save(dkt, './model/best_dkt.pt')
        # print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {sum(train_accuracy)/len(train_accuracy):.2f}")
        # run['loss'].log(sum(train_loss)/len(train_loss))
        # run['accuracy'].log(sum(train_accuracy)/len(train_accuracy))

        torch.save(dkt, './model/final_dkt.pt')

    best_dkt = torch.load('./model/best_dkt.pt')
    causal_order = []
    for name, param in best_dkt.named_parameters():
        if (name == "gru.permuted_matrix.matrix"):
            # torch.set_printoptions(threshold=10_000)
            causal_order = sink_horn(param, verbose=True)
    if perm_dataset: 
        print("Ground truth permutation:\n", gt_perm)
    return causal_order == gt_perm
if __name__ == "__main__":
    for i in range(500):
        print("========"*10)
        SEED = i
        is_true_order = test()
        if is_true_order:
            print("SEED: ", SEED)
        print("========"*10)