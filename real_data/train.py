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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model import PermutedDKT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

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

def get_optimizer_scheduler(name, model, lr, train_dataloader_len, epochs):
    if name == "Adam":
        optimizer = AdamW(model.parameters(),
                    lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
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
            loss, acc = model(b_input_ids, b_labels, epoch_i+1)
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
                valloss, valacc = model(b_input_ids_val, b_labels_val, epoch_i+1)
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

        # Wandb Log Metrics
        wandb.log({"Epoch": epoch_i,
            "Average training loss": avg_train_loss,
            "Average validation loss":avg_val_loss,
            "Validation Accuracy": avg_acc})
    
    print('Least Validation loss:', least_val_loss)
    return model_copy, epochswise_train_losses, epochwise_val_losses


def main():
    if False:
        dataset_tensor = torch.load('../serialized_torch/student_data_tensor.pt')
        with open("../serialized_torch/student_data_construct_list.json", 'rb') as fp:
            tot_construct_list = json.load(fp)
    else:
        dataset_tensor = torch.load('../serialized_torch/sample_student_data_tensor.pt')
        with open("../serialized_torch/sample_student_data_construct_list.json", 'rb') as fp:
            tot_construct_list = json.load(fp)    

    num_of_questions, _, num_of_students = dataset_tensor.shape

    # dkt_model = nn.DataParallel(PermutedDKT(n_concepts=len(tot_construct_list)+1)).to(device) # using dataparallel
    
    # concept_input = dataset_tensor[:, 0, :]
    # labels = dataset_tensor[:, 1, :]
    initial_concept_input = dataset_tensor[:, 0, :]
    map_concept_input = get_mapped_concept_input(initial_concept_input, tot_construct_list)
    concept_input = torch.tensor(map_concept_input, dtype=torch.long)
    
    labels = torch.tensor(dataset_tensor[:, 1, :].clone().detach(), dtype=torch.long)
    # TODO: Batch student-wise not question-wise (dim-1 must be student)
    concept_inp_transpose = torch.transpose(concept_input, 0, 1)
    labels_transpose = torch.transpose(labels, 0, 1)
    # TODO: Get train-validation set
    train_input, valid_input, train_label, valid_label = train_test_split(concept_inp_transpose, labels_transpose, 
                                                            train_size=0.8, random_state=seed_val)
    
    print("PermutedDKT")
    print("Number of questions: ", num_of_questions)
    print("Number of students: ", num_of_students)
    print("Number of concepts:", len(tot_construct_list)+1)

    # TODO: construct a tensor dataset
    batch_size = 64
    epochs = 200
    train_dataloader = get_data_loader(batch_size=batch_size, concept_input=train_input, labels=train_label)
    val_dataloader = get_data_loader(batch_size=batch_size, concept_input=valid_input, labels=valid_label)

    # TODO: Set init_temp and init_unroll
    init_temp = 2
    init_unroll = 5
    embed_dim = 300

    torch.manual_seed(seed_val-1)
    dkt_model = PermutedDKT(init_temp, init_unroll, len(tot_construct_list)+1, embed_dim).to(device)
    
    print("Successfull in data prepration!")
    # TODO: Getting optimzer and scheduler
    lr = 5e-4
    optimizer, scheduler = get_optimizer_scheduler("Adam", dkt_model, lr, len(train_dataloader), epochs)
    print("Successfully loaded the optimizer")

    # Log Hyperparameters
    wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size,
    "init_temp": init_temp,
    "init_unroll": init_unroll
    }

    # Main Traning
    model, epoch_train_loss, epoch_val_loss = train(epochs, dkt_model, train_dataloader, val_dataloader, optimizer, scheduler) # add val_dataloader later
    # TODO: Save the model
    torch.save(model, 'saved_models/nips_embed.pt')


    with open('train_epochwise_loss.json', 'w') as infile:
        json.dump(epoch_train_loss, infile)

    with open('val_epochwise_loss.json', 'w') as infile:
        json.dump(epoch_val_loss, infile)

    # loss, acc = dkt_model(concept_input, labels)
    # print(loss, acc)

if __name__ == "__main__":

    # Start Wandb run
    wandb.init(project="predict-graph", entity="ml4ed", name='nips embed')

    seed_val = 36
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    main()