import wandb
import os
import copy
import math
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,  SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
from datetime import datetime

# setting the seed
seed_val = 37
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

class PermutationMatrix(nn.Module):
    def __init__(self, input_size, temperature, unroll, verbose=False):
        super().__init__()
        self.unroll, self.temperature = unroll, temperature
        self.matrix = nn.Parameter(torch.empty(input_size, input_size, device=device))
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(input_size))

        # NOTE: Trainable L.
        self.lower = nn.Parameter(0.1 * torch.randn(input_size, input_size, device=device))
        if verbose:
            torch.set_printoptions(threshold=100_000)
            print("Initial value of L bar matrix\n", self.lower)
        self.verbose = verbose

    def forward(self, epoch):
        temperature = ((epoch//10)+1)*self.temperature
        unroll = ((epoch//10)+1)*self.unroll

        # NOTE: Sinkhorn Operation
        # For every element of the matrix subtract with the max value, 
        # multiply by the temperature and make it exponential
        if self.verbose:
            print("Initial value of P bar matrix\n", self.matrix)
        
        matrix_shape = self.matrix.shape[0]

        max_row = torch.max(self.matrix, dim=1).values.reshape(matrix_shape, 1)
        ones = torch.ones(matrix_shape, device=device).reshape(1, matrix_shape)

        matrix = torch.exp(temperature * (self.matrix - torch.matmul(max_row, ones)))
        # NOTE: Trainable L.
        lower = torch.empty(matrix_shape, matrix_shape, device = device)
        l_mask = torch.tril(torch.ones(matrix_shape, matrix_shape, device=device))
        lower = torch.sigmoid(self.lower) * l_mask
        if self.verbose:
            torch.set_printoptions(threshold=100_000)
            print("L bar matrix:\n", self.lower)
            print("L matrix\n", lower)
        for _ in range(unroll):
            matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
        # ((P x L) x P^T)^T
        output_lower = torch.matmul(torch.matmul(matrix, lower), matrix.t()).t()
        ideal_matrix_order = matrix.data.argmax(dim=1, keepdim=True) # gives the ideal order of the constructs
        new_matrix = torch.zeros_like(matrix)
        new_matrix.scatter_(
            1, ideal_matrix_order, torch.ones_like(ideal_matrix_order).float()
        )
        causal_order = [(idx, int(d[0])) for idx, d in enumerate(ideal_matrix_order)]
        causal_order.sort(key=lambda x: x[1])
        causal_order = [d[0] for d in causal_order]
        if self.verbose:
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
        verbose=False
    ):
        super().__init__()
        self.batch_first = batch_first
        self.verbose = verbose
        self.permuted_matrix = PermutationMatrix(hidden_size, init_temp, init_unroll, verbose=verbose)
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

        dim = 1 if self.batch_first else 0
        lower = self.permuted_matrix(epoch) # (PLP')'
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
            outputs.append(hidden.clone())
            i = i + 1
            if self.verbose:
                if (i % 400 == 0):
                    print("hidden state:" + str(i))
                    for deviceid in range(torch.cuda.device_count()):
                        print("memory :", str(deviceid), torch.cuda.memory_summary(device=deviceid, abbreviated=True))

        hidden_states = torch.stack(outputs)  # T, B, H
        return hidden_states

class PermutedDKT(nn.Module):
    def __init__(self, init_temp, init_unroll, n_concepts, embed_dim, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.embed_dim = embed_dim
        self.embed_matrix = nn.Parameter(torch.empty(n_concepts, self.embed_dim, device=device))
        nn.init.kaiming_uniform_(self.embed_matrix, a=math.sqrt(self.embed_dim))
        
        self.delta_matrix = nn.Parameter(torch.empty(n_concepts, self.embed_dim, device=device))
        nn.init.kaiming_uniform_(self.delta_matrix, a=math.sqrt(self.embed_dim))

        self.embed_input = nn.Linear(self.embed_dim, n_concepts)

        self.gru = PermutedGru(init_temp, init_unroll, n_concepts, batch_first=False, verbose=self.verbose)
        self.n_concepts = n_concepts
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
        if self.verbose:
            print("PermutedDKT")
            print("Number of questions: ", T)
            print("Number of students: ", B)
            print("Number of concepts:", self.n_concepts)
            print("Concept input: ", concept_input)
        input = torch.zeros(T, B, self.n_concepts, device=device)

        # Unsqueeze concept_input & lables
        # [T,B] -> [T,B,1]
        input.scatter_(2, concept_input.unsqueeze(2), labels.unsqueeze(2).float())

        # Transform input to account for construct embeddings
        rawembed = torch.matmul(abs(input), self.embed_matrix)
        rawdelta = torch.matmul(input, self.delta_matrix)
        preembed = rawembed + rawdelta
        input_embed = self.embed_input(preembed)

        # Create a mask (0 when the input is 0)
        mask_ones = nn.Parameter(torch.ones(T, B, device=device), requires_grad=False)
        mask = mask_ones - (labels==0).long().to(device)

        labels = torch.clamp(labels, min=0)
        hidden_states = self.gru(input_embed, epoch)

        init_state = torch.zeros(1, input.shape[1], input.shape[2]).to(device)
        shifted_hidden_states = torch.cat([init_state, hidden_states], dim=0)[:-1, :, :].to(device) 
        
        # NOTE: Adding construct embeddings
        relevant_hidden_states = torch.gather(shifted_hidden_states, 2, concept_input.unsqueeze(2)) # [num_questions, num_students, 1]
        preoutput = torch.cat((rawembed, relevant_hidden_states), dim=2)

        output = (self.output_layer(preoutput)).squeeze()
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
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def get_optimizer_scheduler(name, model, lr, train_dataloader_len, epochs):
    if name == "Adam":
        optimizer = AdamW(model.parameters(),
                    lr = lr, # args.learning_rate 
                    eps = 1e-8 # args.adam_epsilon
                )
        total_steps = train_dataloader_len * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0,
                                                    num_training_steps = total_steps)
    return optimizer, scheduler

def train(epochs, model, train_dataloader, val_dataloader, optimizer, scheduler, verbose=False):

    epochswise_train_losses, epochwise_val_losses = [], []
    prev_val_loss, early_stop_ctr, early_stop_threshold, early_stop_patience = 0, 0, 5, 0.0001
    least_val_loss, cur_least_epoch = math.inf, 0

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

        # Switch model to the train mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            print('Step:', step)

            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            if verbose:
                print('After loading data'.upper())
                for id in range(torch.cuda.device_count()):
                    print(torch.cuda.memory_summary(device=id))

            # Clear previously accumulated gradients
            model.zero_grad()        

            # NOTE: Forward pass
            loss = model(b_input_ids, b_labels, epoch_i+1)
            print(f'Step {step} loss: {loss}')

            if verbose:
                print('After Forward Pass'.upper())
                for id in range(torch.cuda.device_count()):
                    print(torch.cuda.memory_summary(device=id))

           
            # Perform a backward pass to calculate the gradients.
            if(torch.cuda.device_count() > 1):
                total_loss += loss.mean().item()
                loss.mean().backward() # When using dataparallel
            else:
                total_loss += loss.item()
                loss.backward()

            if verbose:
                print('After Backprop step'.upper())
                for id in range(torch.cuda.device_count()):
                    print(torch.cuda.memory_summary(device=id))
            

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        epochswise_train_losses.append(avg_train_loss)            

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        # ========================================
        #               Validation
        # ========================================
        tot_val_loss, tot_val_acc = 0, 0
        for valstep, valbatch in enumerate(val_dataloader):
            b_input_ids_val = valbatch[0].to(device)
            b_labels_val = valbatch[1].to(device)
            
            with torch.no_grad():
                valloss = model(b_input_ids_val, b_labels_val, epoch_i+1)
                if(torch.cuda.device_count() > 1):
                    tot_val_loss += valloss.mean().item()
                else:
                    tot_val_loss += valloss.item()
        avg_val_loss = tot_val_loss / len(val_dataloader)
        print("  Average validation loss: {0:.2f}".format(avg_val_loss))
        epochwise_val_losses.append(avg_val_loss)
                
        if avg_val_loss < least_val_loss:
            cur_least_epoch = epoch_i
            model_copy = copy.deepcopy(model)
            least_val_loss = avg_val_loss
            date = datetime.now().strftime('%m_%d_%H_%M_%S')
            torch.save(model_copy, os.path.join('saved_models', date + '_' + hyper_params.file_name +'.pt'))

        # Wandb Log Metrics
        if hyper_params.wandb is not None:
            wandb.log({"Epoch": epoch_i,
                    "Average training loss": avg_train_loss,
                    "Average validation loss":avg_val_loss,
                    "cur_least_epoch":cur_least_epoch})
    
    print('Least Validation loss:', least_val_loss)
    return model_copy, epochswise_train_losses, epochwise_val_losses


def main(hyper_params, file_path='serialized_torch/', data_name='student_data', verbose=False):
    if data_name == 'sample_student_data':
        print("Sanity check, you're running on sample data")
    dataset_tensor = torch.load(file_path + data_name + '_tensor.pt')
    with open(file_path + data_name + '_construct_list.json', 'rb') as fp:
        tot_construct_list = json.load(fp)    
    num_of_questions, _, num_of_students = dataset_tensor.shape

    initial_concept_input = dataset_tensor[:, 0, :]
    map_concept_input = get_mapped_concept_input(initial_concept_input, tot_construct_list)
    concept_input = torch.tensor(map_concept_input, dtype=torch.long)
    
    labels = torch.tensor(dataset_tensor[:, 1, :].clone().detach(), dtype=torch.long)
    # Batch student-wise not question-wise (dim-1 must be student)
    concept_inp_transpose = torch.transpose(concept_input, 0, 1)
    labels_transpose = torch.transpose(labels, 0, 1)
    train_input, valid_input, train_label, valid_label = train_test_split(concept_inp_transpose, labels_transpose, 
                                                            train_size=0.8, random_state=seed_val)
    
    if verbose:
        print("Number of questions: ", num_of_questions)
        print("Number of students: ", num_of_students)
        print("Number of concepts:", len(tot_construct_list)+1)

    batch_size = hyper_params.batch_size
    epochs = hyper_params.epochs
    train_dataloader = get_data_loader(batch_size=batch_size, concept_input=train_input, labels=train_label)
    val_dataloader = get_data_loader(batch_size=batch_size, concept_input=valid_input, labels=valid_label)

    # Log Hyperparameters
    if hyper_params.wandb is not None:
        wandb.config = hyper_params

    dkt_model = PermutedDKT(hyper_params.init_temp, hyper_params.init_unroll, len(tot_construct_list)+1, hyper_params.embed_dim, verbose=verbose).to(device)
    dkt_model = nn.DataParallel(dkt_model)
    if verbose:
        print('After loading the model'.upper())
        for id in range(torch.cuda.device_count()):
            print(torch.cuda.memory_summary(device=id))
        print("Successfull in data prepration!")

    optimizer, scheduler = get_optimizer_scheduler("Adam", dkt_model, hyper_params.lr, len(train_dataloader), epochs)
    if verbose:
        print("Successfully loaded the optimizer")

    # Main Traning
    model, epoch_train_loss, epoch_val_loss = train(epochs, dkt_model, train_dataloader, val_dataloader, optimizer, scheduler) # add val_dataloader later
    date = datetime.now().strftime('%m_%d_%H_%M_%S')
    torch.save(model, os.path.join('saved_models', "final_" + date + '_' + hyper_params.file_name + '.pt'))

    if hyper_params.debug:
        with open(hyper_params.file_name + '_train_epochwise_loss.json', 'w') as infile:
            json.dump(epoch_train_loss, infile)

        with open(hyper_params.file_name + '_val_epochwise_loss.json', 'w') as infile:
            json.dump(epoch_val_loss, infile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UMass 2022 casual ordering model training')
    parser.add_argument('-B', '--batch_size', type=int ,default=64, help='batch size')
    parser.add_argument('-E', '--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-D', '--embed_dim', type=int, default=300, help='embedding dimension')
    parser.add_argument('-IT', '--init_temp', type=int, default=2, help='initial temperature')
    parser.add_argument('-IU', '--init_unroll', type=int, default=5, help='initial unroll')
    parser.add_argument('-L', '--lr', type=float, default=5e-4, help='learning_rate')
    parser.add_argument('-V', '--verbose', action=argparse.BooleanOptionalAction, help='Controls amount of printing')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help='Writes additional debug files for debugging')
    parser.add_argument('-WAB', '--wandb', action=argparse.BooleanOptionalAction, help='Write to weights and biases, note you will need to log in and modify config')
    parser.add_argument('-F', '--file_prefix', type=str, default="student_data", help='The prefix for the dataset tensor/construct list')
    parser.add_argument('-P', '--file_path', type=str, default="serialized_torch/", help='The directory containing the the constructs and dataset tensor')

    hyper_params = parser.parse_args()

    hyper_params.file_name = f"final_stretch_batch_{hyper_params.batch_size}_epoch_{hyper_params.epochs}_embed_{hyper_params.embed_dim}"
    
    if hyper_params.wandb is not None:
        # Start Wandb run
        wandb.init(project="predict-graph", entity="ml4ed", name=hyper_params.file_name)

    main(hyper_params, file_path=hyper_params.file_path, data_name=hyper_params.file_prefix, verbose=hyper_params.verbose)