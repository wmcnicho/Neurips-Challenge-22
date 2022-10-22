import math
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.W_ir = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_iz = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_in = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def forward(self, x, lower, hidden=None):
        # x is B, input_size
        # Question: Why isn't self.W_ir being updated for every timestep?
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
    def __init__(self, input_size, temperature, unroll):
        super().__init__()
        self.unroll, self.temperature = unroll, temperature
        self.matrix = nn.Parameter(torch.empty(input_size, input_size, device=device))
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(5))
        self.lower = torch.tril(torch.ones(input_size, input_size, device=device))

    def forward(self, epoch, verbose=False):
        # TODO: update temperature and unroll
        temperature = ((epoch//10)+1)*self.temperature
        unroll = ((epoch//10)+1)*self.unroll

        # NOTE: For every element of the matrix subtract with the max value, multiply by the temperature and make it exponential
        print(self.matrix)
        
        matrix_shape = self.matrix.shape[0]

        max_row = torch.max(self.matrix, dim=1).values.reshape(matrix_shape, 1)
        ones = torch.ones(matrix_shape, device=device).reshape(1, matrix_shape)

        matrix = torch.exp(temperature * (self.matrix - torch.matmul(max_row, ones)))
        
        # # NOTE. Aritra's implementation
        # matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
        
        for _ in range(unroll):
            matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
        # ((P x L) x P^T)^T
        output_lower = torch.matmul(torch.matmul(matrix, self.lower), matrix.t()).t()
        ideal_matrix_order = matrix.data.argmax(dim=1, keepdim=True) # gives the ideal order of the constructs
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
                "Ideal Lower Triangular Matrix\n",
                torch.matmul(torch.matmul(new_matrix, self.lower), new_matrix.t()),
            )
            print("Causal Order\n", causal_order)

        return output_lower


class PermutedGru(nn.Module):
    def __init__(
        self,
        init_temp, 
        init_unroll,
        hidden_size,
        bias=False,
        num_layers=1,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.cell = PermutedGruCell(hidden_size=hidden_size, bias=False)
        self.batch_first = batch_first
        self.permuted_matrix = PermutationMatrix(hidden_size, init_temp, init_unroll)

    def forward(self, input_, epoch, lengths=None, hidden=None):
        # input_ is of dimensionalty (T, B, hidden_size, ...)
        # lenghths is B,
        dim = 1 if self.batch_first else 0
        # lower = self.permuted_matrix(verbose=True)
        lower = self.permuted_matrix(epoch, verbose=False) # (PLP')'
        outputs = []
        # print("Forwarding PermutedGru")
        # print("input_: ", input_)
        # NOTE: Pass for every question at a time for all students x -> (num_students, num_constructs)
        for x in torch.unbind(input_, dim=dim):  # x dim is B, I
            # print("x: ", x.shape) #[2, 5]
            # print("lower: ", lower)
            hidden = self.cell(x, lower, hidden)
            outputs.append(hidden.clone())

        hidden_states = torch.stack(outputs)  # T, B, H
        last_states = []
        if lengths is None:
            lengths = [len(input_)] * len(input_[0])
        for idx, l in enumerate(lengths):
            last_states.append(hidden_states[l - 1, idx, :]) # last hidden states for all students 
        last_states = torch.stack(last_states) # [num_students, num_constructs]
        return hidden_states, last_states


class PermutedDKT(nn.Module):
    def __init__(self, init_temp, init_unroll, n_concepts, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_matrix = nn.Parameter(torch.empty(n_concepts, self.embed_dim, device=device))
        nn.init.kaiming_uniform_(self.embed_matrix, a=math.sqrt(5))
        
        self.delta_matrix = nn.Parameter(torch.empty(n_concepts, self.embed_dim, device=device))
        nn.init.kaiming_uniform_(self.delta_matrix, a=math.sqrt(5))

        self.embed_input = nn.Linear(self.embed_dim, n_concepts)

        self.gru = PermutedGru(init_temp, init_unroll, n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        self.output_layer = nn.Linear(self.embed_dim+1, 1) 
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, concept_input, labels, epoch):
        # Input shape is T (timestep - questions), B (batch size - num of students) 
        # Input[i,j]=k at time i, for student j, concept k is attended
        # label is T,B 0/1
        T, B = concept_input.shape
        # print("PermutedDKT")
        # print("Number of questions: ", T)
        # print("Number of students: ", B)
        # print("Number of concepts:", self.n_concepts)
        # print("Concept input: ", concept_input)
        input = torch.zeros(T, B, self.n_concepts, device=device)

        # print("Before input\n: ", input)
        # Unsqueeze concept_input & lables
        # [T,B] -> [T,B,1], Put 1 at index 2
        # Scatter input
        # scatter_(dim, index, src)

        input.scatter_(2, concept_input.unsqueeze(2), labels.unsqueeze(2).float())

        # TODO: Transform input to account for construct embeddings
        rawembed = torch.matmul(abs(input), self.embed_matrix)
        rawdelta = torch.matmul(input, self.delta_matrix)
        preembed = rawembed + rawdelta
        input_embed = self.embed_input(preembed)


        # TODO: Create a mask (0 when the input is 0)
        mask = torch.ones(T, B, device=device)
        zero_index_row, zero_index_col = (labels==0).nonzero(as_tuple=True)
        zero_index_row, zero_index_col = list(zero_index_row.cpu()), list(zero_index_col.cpu())
        for r, c in zip(zero_index_row, zero_index_col):
            r_num, c_num = r.item(), c.item()
            mask[r_num][c_num] = 0


        labels = torch.clamp(labels, min=0)
        hidden_states, _ = self.gru(input_embed, epoch)

        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[:-1, :, :].to(device) 
        # TODO: Add construct embeddings
        relevant_hidden_states = torch.gather(shifted_hidden_states, 2, concept_input.unsqueeze(2)) # [num_questions, num_students, 1]
        preoutput = torch.cat((rawembed, relevant_hidden_states), dim=2)

        output = self.output_layer(preoutput).squeeze()

        pred = (output > 0.0).float()
        acc_raw = torch.mean((pred*mask == labels*mask).float())
        acc_corrrection = len((mask==0).nonzero())/(mask.shape[0]*mask.shape[1])
        acc = acc_raw - acc_corrrection
        raw_loss = self.ce_loss(output, labels.float())
        loss_masked = (raw_loss * mask).mean()
        return loss_masked, acc
