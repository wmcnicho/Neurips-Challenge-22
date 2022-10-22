from hashlib import new
import math
# import pudb
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PermutationMatrix(nn.Module):
    def __init__(self, input_size, temperature=100, unroll=20):
        super().__init__()
        self.unroll, self.temperature = unroll, temperature
        self.matrix = nn.Parameter(torch.empty(input_size, input_size), requires_grad=True) # permutation matrix
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(5))
        self.lower = torch.tril(torch.ones(input_size, input_size))  # lower triangular matrix
    def forward(self, verbose=False):
        matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
        for _ in range(self.unroll):
            matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
        output_lower = torch.matmul(torch.matmul(matrix, self.lower), matrix.t()).t()
        ideal_matrix_order = matrix.data.argmax(dim=1, keepdim=True) # index of max value in each row of the matrix
        new_matrix = torch.zeros_like(matrix) 
        new_matrix.scatter_(
            1, ideal_matrix_order, torch.ones_like(ideal_matrix_order).float()
        ) # ideal p matrix
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
            n_t = \tanh(W_{in} x_t + b_{in} +  (W_{hn}(r_t *  h_{(t-1)})+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}
        """
        self.hidden_size = hidden_size
        self.W_ir = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.W_iz = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.W_in = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            # print(w.requires_grad)

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


class PermutedDKT(nn.Module):
    def __init__(self, n_concepts):
        super().__init__()
        self.gru = PermutedGru(n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        self.output_layer = nn.Linear(1, 1)
        # for param in self.output_layer.parameters():
        #     print(param.requires_grad)
    def forward(self, concept_input, labels):
        # Input shape is T, B
        # Input[i,j]=k at time i, for student j, concept k is attended
        # label is T,B 0/1
        T, B = concept_input.shape # T is the number of timesteps and B is the batch_size
        input = torch.zeros(T, B, self.n_concepts)
        input.scatter_(2, concept_input.unsqueeze(2), labels.unsqueeze(2).float())
        hidden_states, _ = self.gru(input)
        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[1:, :, :]
        output = self.output_layer(shifted_hidden_states.unsqueeze(3)).squeeze(3)
        output = torch.gather(output, 2, concept_input.unsqueeze(2)).squeeze(2)
        # print("output", output)
        pred = (output > 0.0).float()
        # print(pred)
        return pred


# def main():
#     dkt = PermutedDKT(n_concepts=5)
#     learning_rate = 0.001
#     XX = 10
#     # concept_input = torch.randint(0, 5, (4, 2))
#     # labels = torch.randint(0, 2, (4, 2)) * 2 - 1

#     concept_input_list = torch.randint(0, 5, (XX, 4, 2))
#     labels_list = torch.randint(0, 2, (XX, 4, 2)) * 2 - 1

#     # print("concept:", concept_input)
#     # print("labels:", labels)
#     optimizer = torch.optim.SGD(dkt.parameters(), lr=learning_rate)
#     cc_loss = nn.BCEWithLogitsLoss()
    
#     for i in range(XX):
#         print("Training ", i, "th")
#         y_pred = dkt(concept_input_list[i], labels_list[i])
#         optimizer.zero_grad()
#         y = torch.clamp(labels_list[i], min=0).float()
#         loss = cc_loss(y_pred, y)
#         acc = torch.mean((y_pred == y).float())
#         loss.requires_grad = True
#         loss.backward()
#         optimizer.step()
#         print("loss:", loss)
#         print("acc", acc)
#         print("\n")

def main():
    dkt = PermutedDKT(n_concepts=5)

    T = 4 # number of questions
    B = 2 # number of students

    concept_input = torch.randint(0, 5, (T, B))
    labels = torch.randint(0, 2, (T, B)) * 2 - 1

    print("Input: \n")
    print(concept_input)
    # XX = 10
    # concept_input_list = torch.randint(0, 5, (XX, 4, 2))
    # labels_list = torch.randint(0, 2, (XX, 4, 2)) * 2 - 1

    # Define hyperparameters
    # n_epochs = 100
    # lr=0.01
    learning_rate = 0.001

    # Define Loss, Optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(dkt.parameters(), lr=learning_rate)

    dkt.train()

    niter = 1

    running_loss = 0.0
    for _ in range(0, niter):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        y_pred = dkt(concept_input, labels)
        loss = loss_fn(y_pred, torch.clamp(labels, min=0).float())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        # print statistics
        print("-" * 10)
        running_loss += loss.item()
        for name, val in dkt.named_parameters():
            # print("name: ", name, "\nval: ", val.data)
            if name == "gru.permuted_matrix.matrix":
                print("Permutation Matrix P: \n", val.data)


if __name__ == "__main__":
    main()
