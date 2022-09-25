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

        self.W_ir = torch.empty(hidden_size, hidden_size)
        self.W_hr = torch.empty(hidden_size, hidden_size)
        self.W_iz = torch.empty(hidden_size, hidden_size)
        self.W_hz = torch.empty(hidden_size, hidden_size)
        self.W_in = torch.empty(hidden_size, hidden_size)
        self.W_hn = torch.empty(hidden_size, hidden_size)
        self.b_ir = torch.randn(hidden_size)
        self.b_hr = torch.randn(hidden_size)
        self.b_iz = torch.randn(hidden_size)
        self.b_hz = torch.randn(hidden_size)
        self.b_in = torch.randn(hidden_size)
        self.b_hn = torch.randn(hidden_size)
        nn.init.kaiming_normal_(self.W_hn, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_ir, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_hr, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_iz, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_hz, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_in, a=math.sqrt(hidden_size), mode='fan_out')

    def forward(self, x, lower, mask, hidden=None):
        # x is B, input_size
        if hidden is None:
            # hidden = torch.zeros(x.size(0), self.hidden_size).to(device)
            hidden = torch.randn(x.size(0), self.hidden_size).to(device)
        W_ir = self.W_ir * lower
        W_hr = self.W_hr * lower
        W_iz = self.W_iz * lower
        W_hz = self.W_hz * lower
        W_in = self.W_in * lower
        W_hn = self.W_hn * lower
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        # print(f"before h:\n {hidden}")
        r_t = sigmoid(torch.matmul(x, W_ir) + self.b_ir * mask + torch.matmul(hidden, W_hr) + self.b_hr)
        z_t = sigmoid(torch.matmul(x, W_iz) + self.b_iz * mask + torch.matmul(hidden, W_hz) + self.b_hz)
        n_t = tanh(torch.matmul(x, W_in) +  self.b_in * mask + r_t * (torch.matmul(hidden, W_hn) + self.b_hn))
        hy = hidden * z_t + (1.0 - z_t) * n_t
        # print(f"after h:\n {hy}")

        return hy

class GroundTruthPermutationMatrix(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.matrix = torch.eye(input_size)
        self.lower = torch.tril(torch.ones(input_size, input_size))
        # self.matrix = self.matrix.to(torch.double)
        # self.lower = self.lower.to(torch.double)
    def forward(self, verbose=False):
        # output_lower = torch.matmul(torch.matmul(self.matrix, self.lower), self.matrix.t()).t()
        output_lower = torch.matmul(torch.matmul(self.matrix, self.lower), self.matrix.t())
        # matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
        
        ideal_matrix_order = self.matrix.data.argmax(dim=1, keepdim=True)
        causal_order = [(idx, int(d[0])) for idx, d in enumerate(ideal_matrix_order)]
        causal_order.sort(key=lambda x: x[1])
        causal_order = [d[0] for d in causal_order]
        print("Causal order: ", causal_order)
       
        return output_lower

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
        self.cell = GroundTruthPermutedGruCell(hidden_size=hidden_size, bias=False)
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
        # print("Debug student minds: \n")
        for t, x in enumerate(torch.unbind(input_, dim=dim)):  # x dim is S(B)
            # print("======" * 10)
            # print(f"time step: {t}")
            # print(f"construct: {x.argmax()}")
            hidden = self.cell(x, lower, x, hidden)
            labels.append(torch.sigmoid(hidden*W+b).clone()*x)
        labels = torch.stack(labels)
        ans = torch.zeros(labels.shape[0], labels.shape[1])
        for i, xx in enumerate(labels):
            for j, yy in enumerate(xx):
                ans[i][j] = torch.max(yy)
        ans = (ans >= 0.5).int()
        ans = torch.where(ans==0, ans-1, ans)
        return ans

class GroundTruthPermutedDKT(nn.Module):
    def __init__(self, n_concepts):
        super().__init__()
        self.gru = GroundTruthPermutedGru(n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        # self.output_layer = nn.Linear(1, 1)

    def forward(self, concept_input):
        # Input shape is Q(T), S(B)
        # Input[i,j]=k at time i, for student j, concept k is attended
        # label is T,B 0/1
        B, T = concept_input.shape
        input = torch.zeros(T, B, self.n_concepts)
        concept_input_t = concept_input.t()
        input.scatter_(2, concept_input_t.unsqueeze(2), 1)
        labels = self.gru(input)

        return labels.t()

