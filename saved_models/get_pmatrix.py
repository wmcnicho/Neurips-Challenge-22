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
    def __init__(self, input_size, temperature=100, unroll=1000):
        super().__init__()
        self.unroll, self.temperature = unroll, temperature
        self.matrix = nn.Parameter(torch.empty(input_size, input_size, device=device))
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(5))
        self.lower = torch.tril(torch.ones(input_size, input_size, device=device))

    def forward(self, verbose=False):
        # NOTE: For every element of the matrix subtract with the max value, multiply by the temperature and make it exponential
        print(self.matrix)
        
        matrix_shape = self.matrix.shape[0]

        max_row = torch.max(self.matrix, dim=1).values.reshape(matrix_shape, 1)
        ones = torch.ones(matrix_shape, device=device).reshape(1, matrix_shape)

        matrix = torch.exp(self.temperature * (self.matrix - torch.matmul(max_row, ones)))
        
        # # NOTE. Aritra's implementation
        # matrix = torch.exp(self.temperature * (self.matrix - torch.max(self.matrix)))
        
        for _ in range(self.unroll):
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
        # lower = self.permuted_matrix(verbose=True)
        lower = self.permuted_matrix(verbose=False) # (PLP')'
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
    def __init__(self, n_concepts):
        super().__init__()
        self.gru = PermutedGru(n_concepts, batch_first=False)
        self.n_concepts = n_concepts
        self.output_layer = nn.Linear(1, 1)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, concept_input, labels):
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

        # TODO: Create a mask (0 when the input is 0)
        mask = torch.ones(T, B, device=device)
        zero_index_row, zero_index_col = (labels==0).nonzero(as_tuple=True)
        zero_index_row, zero_index_col = list(zero_index_row.cpu()), list(zero_index_col.cpu())
        for r, c in zip(zero_index_row, zero_index_col):
            r_num, c_num = r.item(), c.item()
            mask[r_num][c_num] = 0


        labels = torch.clamp(labels, min=0)
        hidden_states, _ = self.gru(input)

        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[:-1, :, :].to(device) # NOTE: Why this?
        output = self.output_layer(shifted_hidden_states.unsqueeze(3)).squeeze(3)
        output = torch.gather(output, 2, concept_input.unsqueeze(2)).squeeze(2) # [num_questions, num_students]
        pred = (output > 0.0).float()
        acc = torch.mean((pred == labels).float())
        raw_loss = self.ce_loss(output, labels.float())
        loss_masked = (raw_loss * mask).mean()
        return loss_masked, acc

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

def train(epochs, model, train_dataloader, val_dataloader, optimizer, scheduler):
    if os.path.exists('train_debug_new.txt'):
        os.remove('train_debug_new.txt')
    train_file = open('train_debug_new.txt', 'w')
    epochswise_train_losses, epochwise_val_losses = [], []
    prev_val_loss, early_stop_ctr, early_stop_threshold, early_stop_patience = 0, 0, 5, 0.0001
    least_val_loss = math.inf

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            print('Step:', step)
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            # TODO: Re-transpose to make the dimension (Q, S)
            b_input_ids = torch.transpose(batch[0], 0, 1).to(device)
            b_labels = torch.transpose(batch[1], 0, 1).to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            loss, acc = model(b_input_ids, b_labels)
            print('step loss:', loss)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        epochswise_train_losses.append(avg_train_loss)            

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        ############### Validation ###############
        tot_val_loss, tot_val_acc = 0, 0
        for valstep, valbatch in enumerate(val_dataloader):
            b_input_ids_val = torch.transpose(valbatch[0], 0, 1).to(device)
            b_labels_val = torch.transpose(valbatch[1], 0, 1).to(device)
            
            with torch.no_grad():
                valloss, valacc = model(b_input_ids_val, b_labels_val)
                tot_val_loss += valloss.item()
                tot_val_acc += valacc
        avg_val_loss = tot_val_loss / len(val_dataloader)
        avg_acc = tot_val_acc/ len(val_dataloader)
        print("  Average validation loss: {0:.2f}".format(avg_val_loss))
        print("  Average vakidation accuracy: {0:.2f}".format(avg_acc))
        epochwise_val_losses.append(avg_val_loss)
        
        # # NOTE: Early stopping
        # if abs(avg_val_loss-prev_val_loss) < early_stop_patience:
        #     if early_stop_ctr < early_stop_threshold:
        #         early_stop_ctr += 1
        #     else:
        #         print('Early Stopping. No improvement in validation loss')
        #         break 
        
        if avg_val_loss < least_val_loss:
            model_copy = copy.deepcopy(model)
            least_val_loss = avg_val_loss

        prev_val_loss = avg_val_loss
    
    print('Least Validation loss:', least_val_loss)
    return model_copy, epochswise_train_losses, epochwise_val_losses

def get_sinkhorn_output(matrix):
    temperature=10
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
    model_load = torch.load('saved_models/nips_vanilla.pt', map_location=torch.device('cpu'))
    sinkhorn_output = get_sinkhorn_output(model_load.gru.permuted_matrix.matrix)
    np_matrix = sinkhorn_output.cpu().detach().numpy()
    argmax_search = search_argmax(np_matrix)
    # argmax_list_row = np.argmax(np_matrix, axis=1)
    # argmax_list_col = np.argmax(np_matrix, axis=0)
    # print('P Matrix by row', len(set(argmax_list_row)))
    # print('P Matrix by col', len(set(argmax_list_col)))
    p_matrix = np.zeros(np_matrix.shape)
    for row, col in enumerate(argmax_search):
        p_matrix[row][col] = 1
    np.save('saved_models/p_matrix_vanilla.npy', p_matrix)
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