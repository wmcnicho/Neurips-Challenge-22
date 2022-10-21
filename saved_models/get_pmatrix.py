import os
import copy
import math
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# setting the seed
seed_val = 37
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

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
            nn.init.kaiming_uniform_(w, a=math.sqrt(self.hidden_size))

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
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(input_size))
        self.lower = nn.Parameter(torch.tril(torch.ones(input_size, input_size)), requires_grad=False)
        # self.lower = torch.tril(torch.ones(input_size, input_size, device=device))

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
        # self.cell = PermutedGruCell(hidden_size=hidden_size, bias=False)
        self.batch_first = batch_first
        self.permuted_matrix = PermutationMatrix(hidden_size, init_temp, init_unroll)
        self.hidden_size = hidden_size
        self.W_ir = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_iz = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_in = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size, device=device))
        
        nn.init.kaiming_normal_(self.W_ir, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_hr, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_iz, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_hz, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_in, a=math.sqrt(hidden_size), mode='fan_out')
        nn.init.kaiming_normal_(self.W_hn, a=math.sqrt(hidden_size), mode='fan_out')


    def forward(self, input_, epoch, lengths=None, hidden=None):
        # input_ is of dimensionalty (T, B, hidden_size, ...)
        # lenghths is B,
        dim = 1 if self.batch_first else 0
        # lower = self.permuted_matrix(verbose=True)
        lower = self.permuted_matrix(epoch, verbose=False) # (PLP')'
        outputs = []
        W_ir = self.W_ir * lower
        W_hr = self.W_hr * lower
        W_iz = self.W_iz * lower
        W_hz = self.W_hz * lower
        W_in = self.W_in * lower
        W_hn = self.W_hn * lower
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        i = 0
        # NOTE: Pass for every question at a time for all students x -> (num_students, num_constructs)
        for x in torch.unbind(input_, dim=dim):  # x dim is B, I
            if hidden is None:
                hidden = torch.zeros(x.size(0), self.hidden_size).to(device)
            r_t = sigmoid(torch.matmul(x, W_ir) + torch.matmul(hidden, W_hr))
            z_t = sigmoid(torch.matmul(x, W_iz) + torch.matmul(hidden, W_hz))
            n_t = tanh(torch.matmul(x, W_in) + torch.matmul(r_t * hidden, W_hn))
            hidden = hidden * z_t + (1.0 - z_t) * n_t
            i = i + 1
            # if (i % 400 == 0):
            #     print("hidden state:" + str(i))
            #     for deviceid in range(torch.cuda.device_count()):
            #         print("memory :", str(deviceid), torch.cuda.memory_summary(device=deviceid, abbreviated=True))
            outputs.append(hidden.clone())
            # hidden = self.cell(x, lower, hidden)
            # outputs.append(hidden.clone().detach())

        hidden_states = torch.stack(outputs)  # T, B, H
        return hidden_states


# class PermutedGru(nn.Module):
#     def __init__(
#         self,
#         init_temp, 
#         init_unroll,
#         hidden_size,
#         bias=False,
#         num_layers=1,
#         batch_first=False,
#         dropout=0.0,
#     ):
#         super().__init__()
#         self.cell = PermutedGruCell(hidden_size=hidden_size, bias=False)
#         self.batch_first = batch_first
#         self.permuted_matrix = PermutationMatrix(hidden_size, init_temp, init_unroll)

#     def forward(self, input_, epoch, lengths=None, hidden=None):
#         # input_ is of dimensionalty (T, B, hidden_size, ...)
#         # lenghths is B,
#         dim = 1 if self.batch_first else 0
#         # lower = self.permuted_matrix(verbose=True)
#         lower = self.permuted_matrix(epoch, verbose=False) # (PLP')'
#         outputs = []
#         # print("Forwarding PermutedGru")
#         # print("input_: ", input_)
#         # NOTE: Pass for every question at a time for all students x -> (num_students, num_constructs)
#         for x in torch.unbind(input_, dim=dim):  # x dim is B, I
#             # print("x: ", x.shape) #[2, 5]
#             # print("lower: ", lower)
#             hidden = self.cell(x, lower, hidden)
#             outputs.append(hidden.clone())

#         hidden_states = torch.stack(outputs)  # T, B, H
#         last_states = []
#         if lengths is None:
#             lengths = [len(input_)] * len(input_[0])
#         for idx, l in enumerate(lengths):
#             last_states.append(hidden_states[l - 1, idx, :]) # last hidden states for all students 
#         last_states = torch.stack(last_states) # [num_students, num_constructs]
#         return hidden_states, last_states


class PermutedDKT(nn.Module):
    def __init__(self, init_temp, init_unroll, n_concepts, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_matrix = nn.Parameter(torch.empty(n_concepts, self.embed_dim, device=device))
        nn.init.kaiming_uniform_(self.embed_matrix, a=math.sqrt(self.embed_dim))
        
        self.delta_matrix = nn.Parameter(torch.empty(n_concepts, self.embed_dim, device=device))
        nn.init.kaiming_uniform_(self.delta_matrix, a=math.sqrt(self.embed_dim))

        self.embed_input = nn.Linear(self.embed_dim, n_concepts)

        self.gru = PermutedGru(init_temp, init_unroll, n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        # self.output_layer = nn.Linear(self.embed_dim+self.n_concepts, 1) 
        self.output_layer = nn.Linear(self.embed_dim+1, 1) 
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, concept_input_untrans, labels_untrans, epoch):

        print("batch input size:", labels_untrans.size(), 'Device:', labels_untrans.get_device())

        concept_input = torch.transpose(concept_input_untrans, 0, 1).to(device)
        labels = torch.transpose(labels_untrans, 0, 1).to(device)

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
        mask_ones = nn.Parameter(torch.ones(T, B, device=device), requires_grad=False)
        mask = mask_ones - (labels==0).long().to(device)
        # zero_index_row, zero_index_col = (labels==0).nonzero(as_tuple=True)
        # zero_index_row, zero_index_col = list(zero_index_row.cpu()), list(zero_index_col.cpu())
        # for r, c in zip(zero_index_row, zero_index_col):
        #     r_num, c_num = r.item(), c.item()
        #     mask[r_num][c_num] = 0


        labels = torch.clamp(labels, min=0)
        hidden_states = self.gru(input_embed, epoch) # initially hidden_states, _

        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[:-1, :, :].to(device) 
        # TODO: Add construct embeddings
        # relevant_hidden_states = shifted_hidden_states * abs(input)
        relevant_hidden_states = torch.gather(shifted_hidden_states, 2, concept_input.unsqueeze(2)) # [num_questions, num_students, 1]
        preoutput = torch.cat((rawembed, relevant_hidden_states), dim=2)

        output = (self.output_layer(preoutput)).squeeze()

        # pred = (output > 0.0).float()
        # acc_raw = torch.mean((pred*mask == labels*mask).float())
        # acc_corrrection = len((mask==0).nonzero())/(mask.shape[0]*mask.shape[1])
        # acc = acc_raw - acc_corrrection
        acc = 0 
        raw_loss = self.ce_loss(output, labels.squeeze().float()) # output.squeeze()
        loss_masked = (raw_loss * mask).sum()/mask.sum().item()
        return loss_masked

def get_mapped_concept_input(initial_concept_input, tot_construct_list):
    map = {k:i for i, k in enumerate(tot_construct_list)}
    if 0 not in map.keys():
        map[0] = len(map)
    else:
        print('Warning: 0 cannot be used ')
    new_matrix = []
    for row in initial_concept_input:
        row_values = []
        for value in row:
            row_values.append(map[int(value.item())])
        new_matrix.append(row_values)
    return new_matrix

def get_data_loader(batch_size, concept_input, labels):
    print('Using batch size:', batch_size)
    data = TensorDataset(concept_input, labels)
    sampler = SequentialSampler(data) # change to randomsampler later
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def get_optimizer_scheduler(name, model, train_dataloader_len, epochs):
    if name == "Adam":
        optimizer = AdamW(model.parameters(),
                    lr = 5e-3, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        total_steps = train_dataloader_len * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
    return optimizer, scheduler

def get_sinkhorn_output(matrix):
    temperature=40
    unroll=100

    matrix_shape = matrix.shape[0]

    max_row = torch.max(matrix, dim=1).values.reshape(matrix_shape, 1)
    ones = torch.ones(matrix_shape, device=device).reshape(1, matrix_shape)

    matrix = torch.exp(temperature * (matrix - torch.matmul(max_row, ones)))
    
    # # NOTE. Aritra's implementation
    # matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
    
    for _ in range(unroll):
        matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
        matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
    
    return matrix

def search_argmax(matrix):
    taken_indices = {}
    index_lst = []
    for row in matrix:
        val_index_pair = {k:i for i, k in enumerate(row)}
        val_index_sorted = sorted(val_index_pair.items(), reverse=True)
        for val, index in val_index_sorted:
            try:
                res = taken_indices[index]
            except KeyError:
                index_lst.append(index)
                taken_indices[index] = 1
                break
    return index_lst




def main():
    # # dataset = torch.load('serialized_torch/student_data_tensor.pt')
    # dataset_tensor = torch.load('../serialized_torch/student_data_tensor.pt')
    # with open("../serialized_torch/student_data_construct_list.json", 'rb') as fp:
    #     tot_construct_list = json.load(fp)
    
    # num_of_questions, _, num_of_students = dataset_tensor.shape

    # # dkt_model = nn.DataParallel(PermutedDKT(n_concepts=len(tot_construct_list)+1)).to(device) # using dataparallel
    # dkt_model = PermutedDKT(n_concepts=len(tot_construct_list)+1).to(device)
    if device == torch.device('cpu'):
        model_load = torch.load('nips_embed_300_newmask_loss.pt', map_location=torch.device('cpu'))
    else:
        model_load = torch.load('nips_embed_300_newmask_loss.pt')
    try:
        p_matrix = model_load.gru.permuted_matrix.matrix
    except:
        p_matrix = model_load.module.gru.permuted_matrix.matrix
    sinkhorn_output = get_sinkhorn_output(p_matrix)
    np_matrix = sinkhorn_output.cpu().detach().numpy()
    np.save('sinkhorn_matrix_embed_300_newmask_loss.npy', np_matrix)
    argmax_search = search_argmax(np_matrix)
    # argmax_list_row = np.argmax(np_matrix, axis=1)
    # argmax_list_col = np.argmax(np_matrix, axis=0)
    # print('P Matrix by row', len(set(argmax_list_row)))
    # print('P Matrix by col', len(set(argmax_list_col)))
    p_matrix = np.zeros(np_matrix.shape)
    for row, col in enumerate(argmax_search):
        p_matrix[row][col] = 1
    np.save('p_matrix_embed_300_newmask_loss.npy', p_matrix)
    # dkt_model.load('nips.pt')
    # for param in model_load.parameters():
    #     print(param)
    # 
    # concept_input = dataset_tensor[:, 0, :]
    # labels = dataset_tensor[:, 1, :]
    # initial_concept_input = dataset_tensor[:, 0, :]
    # map_concept_input = get_mapped_concept_input(initial_concept_input, tot_construct_list)
    # concept_input = torch.tensor(map_concept_input, dtype=torch.long)
    
    # labels = torch.tensor(dataset_tensor[:, 1, :].clone().detach(), dtype=torch.long)
    # # TODO: Batch student-wise not question-wise (dim-1 must be student)
    # concept_inp_transpose = torch.transpose(concept_input, 0, 1)
    # labels_transpose = torch.transpose(labels, 0, 1)

if __name__ == "__main__":
    main()