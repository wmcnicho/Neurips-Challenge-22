import math
import numpy as np
import pudb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import neptune.new as neptune
import random
import matplotlib.pyplot as plt
from ground_truth_model import GroundTruthPermutedDKT
from generate_data import generate_labels

from model import PermutedDKT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # output_lower = torch.matmul(torch.matmul(p_matrix, lower), p_matrix.t()).t()
    output_lower = torch.matmul(torch.matmul(p_matrix, lower), p_matrix.t())

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
            # output_lower.t().data.numpy().round(1),
            output_lower.data.numpy().round(1),

        )
        print("Ideal Permutation Matrix\n", new_matrix.data)
        print(
            "Ideal Lower Triangular\
                p_matrix\n",
            torch.matmul(torch.matmul(new_matrix, lower), new_matrix.t()),
        )
        print("Causal Order\n", causal_order)

        return causal_order

def print_lower(explicit_p):
    # row, col = torch.tril_indices(params.num_constructs, params.num_constructs)
    row, col = torch.tril_indices(params.num_constructs, params.num_constructs, -1)
    lower = torch.zeros(params.num_constructs, params.num_constructs)
    lower.fill_diagonal_(1)
    mean = torch.mean(explicit_p)
    idx = 0
    for r, c in zip(row, col):
        lower[r.item(), c.item()] = explicit_p[idx]
        # lower[r.item(), c.item()] = explicit_p[idx] / mean
        # lower[r.item(), c.item()] = torch.sigmoid(explicit_p[idx] / mean)
        # lower[r.item(), c.item()] = torch.sigmoid(explicit_p[idx])
        idx += 1
    print("Lower matrix:\n")
    print(lower)
    return lower
    # print((lower >= 0.5).int())


def sink_horn_(matrix, lower, temperature=100, unroll=20, verbose=False):
    p_matrix = torch.exp(temperature * (matrix - torch.max(matrix)))
    # print("Debug lower: \n", lower)
    # lower = torch.tril(torch.ones(matrix.shape[0], matrix.shape[1]))
    for _ in range(unroll):
        p_matrix = p_matrix / torch.sum(p_matrix, dim=1, keepdim=True)
        p_matrix = p_matrix / torch.sum(p_matrix, dim=0, keepdim=True)
    # output_lower = torch.matmul(torch.matmul(p_matrix, lower), p_matrix.t()).t()
    output_lower = torch.matmul(torch.matmul(p_matrix, lower), p_matrix.t())

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
            # output_lower.t().data.numpy().round(1),
            output_lower.data.numpy().round(1),

        )
        print("Ideal Permutation Matrix\n", new_matrix.data)
        print(
            "Ideal Lower Triangular\
                p_matrix\n",
            torch.matmul(torch.matmul(new_matrix, lower), new_matrix.t()),
        )
        print("Causal Order\n", causal_order)

        return causal_order
        
def model_train():

    gt_dkt = GroundTruthPermutedDKT(n_concepts=params.num_constructs)
    features = torch.randint(0, params.num_constructs, (params.num_students, params.num_questions))
    # tmp = [[i for i in range(C)], [i for i in range(C)]]
    # features = torch.tensor(tmp)
    labels, ideal_params = generate_labels(features, params)
    # labels = gt_dkt(features)

    student_info = {}
    for idx, (feature, label) in enumerate(zip(features.tolist(), labels.tolist())):
        student_info[idx] = {}
        student_info[idx]["feature"] = feature
        student_info[idx]["label"] = label

    if params.permutation:
        gt_perm = list(np.random.permutation(params.num_constructs))
        # print("Ground truth permutation: ", gt_perm)
        for i, xx in enumerate(features):
            for j, yy in enumerate(xx):
                features[i][j] = gt_perm[features[i][j]]

    training_set = createDataset(features, labels)
    # Different seed number
    torch.manual_seed(seed_num-1)
    dkt = PermutedDKT(n_concepts=params.num_constructs, temperature=params.temperature, unroll=params.unroll, objective=params.objective).to(device)

    training_loader = DataLoader(training_set, batch_size=params.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(dkt.parameters(), lr=params.learning_rate)
    # optimizer = torch.optim.Adam(ideal_params, lr=params.learning_rate)
    n_epochs = params.num_epochs
    best_loss = 100.0
    best_accuracy = 0.0
    best_epoch = 0.0
    run = neptune.init(
        project="phdprojects/neurips-sanity",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTRjZTIxNi1kMzE5LTRlNzgtOGUyZC1hZmMwMTRiNzkzMWYifQ==",
        capture_hardware_metrics = False
    )
    PARAMS = {
            'objective' : params.objective,
            'mask_lower' : params.mask_lower,
            'num_constructs': params.num_constructs,
            'num_questions': params.num_questions,
            'num_students': params.num_students,
            'batch_size': params.batch_size,
            'temperature': params.temperature,
            'unroll': params.unroll,
            'learning_rate': params.learning_rate,
            'num_epochs': params.num_epochs,
            'permutation': params.permutation,
            }

    run["parameters"] = PARAMS
    run["student_info"] = student_info

    for epoch in range(n_epochs): # loop over the dataset multiple times
        train_loss=[]
        train_accuracy=[]
        for i, data in enumerate(training_loader):
            b_construct = data['Features']
            b_label = data['Labels']
            optimizer.zero_grad()
            loss, acc = dkt(b_construct, b_label)
            train_accuracy.append(acc)
            train_loss.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            if params.objective == "L" or params.objective == "PL":
                print("*****"*10)
                print_lower(dkt.gru.permuted_matrix.explicit_p )
                with torch.no_grad():
                    dkt.gru.permuted_matrix.explicit_p[:] = torch.clamp(dkt.gru.permuted_matrix.explicit_p, min=0, max=1)
                print_lower(dkt.gru.permuted_matrix.explicit_p)
        if (sum(train_loss)/len(train_loss) < best_loss):
            best_loss = sum(train_loss)/len(train_loss)
            best_accuracy = sum(train_accuracy)/len(train_accuracy)
            best_epoch = epoch
            run['best_loss'] = best_loss
            run['best_accuracy'] = best_accuracy
            run['best_epoch'] = best_epoch
            # torch.save(dkt.state_dict(), f'./model/{params.file_name}_best_dkt.pt') 
            torch.save(dkt, f'./model/{params.file_name}_best_dkt.pt')
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {sum(train_accuracy)/len(train_accuracy):.2f}")
        run['loss'].log(sum(train_loss)/len(train_loss))
        run['accuracy'].log(sum(train_accuracy)/len(train_accuracy))

        torch.save(dkt, f'./model/{params.file_name}_final_dkt.pt')

    best_dkt = torch.load(f'./model/{params.file_name}_best_dkt.pt')
    last_dkt = torch.load(f'./model/{params.file_name}_final_dkt.pt')
    causal_order = []
    trained_params = {}
    print("best_dkt parameters")
    print("-----" * 10)
    for name, param in best_dkt.named_parameters():
        print(name, param)
        if (name == "gru.permuted_matrix.matrix"):
            causal_order = sink_horn(param, params.temperature, params.unroll, verbose=False)
            trained_params["P"] = param
        elif (name == "gru.permuted_matrix.explicit_p"):
            trained_params["L"] = print_lower(param)
    print("#####"*10)
    # print(trained_params)
    causal_order = sink_horn_(trained_params["P"], trained_params["L"], params.temperature, params.unroll, verbose=True)
    # print("last_dkt parameters")
    # print("-----" * 10)
    # for name, param in last_dkt.named_parameters():
    #     print(name, param)
    #     if (name == "gru.permuted_matrix.matrix"):
    #         causal_order = sink_horn(param, params.temperature, params.unroll, verbose=True)
    #     elif (name == "gru.permuted_matrix.explicit_p"):
    #         print_lower(param)
    
    if params.permutation:
        print("====="*10)
        print("Trained permutiation:\n", causal_order)
        print("Ground Truth permutiation:\n", gt_perm)
    else:
        print("====="*10)
        print("Trained permutiation:\n", causal_order)        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ML')

    parser.add_argument('-B', '--batch_size', type=int ,default=1, help='batch size')
    parser.add_argument('-C', '--num_constructs', type=int, default=5, help='number of constructs')
    parser.add_argument('-Q', '--num_questions', type=int, default=10, help='number of questions')
    parser.add_argument('-S', '--num_students', type=int, default=20, help='number of students')
    parser.add_argument('-T', '--temperature', type=int ,default=100, help='temperature')
    parser.add_argument('-U', '--unroll', type=int, default=100, help='unroll')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-E', '--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-P', '--permutation', action="store_true", help='permute construct order')
    parser.add_argument('-O', '--objective', default="P", choices=["P", "L", "PL"], help='training P/L/PL')
    parser.add_argument('-M', '--mask_lower', action="store_true", help='mask L matrix')

    params = parser.parse_args()

    if params.objective == "P":
        assert not params.mask_lower, "Mask not implemented for objective P."

    file_name = [params.num_constructs, params.num_questions, params.num_students, params.temperature, params.unroll, params.learning_rate, params.num_epochs, params.objective, params.mask_lower]
    file_name =  [str(d) for d in file_name]
    params.file_name = '_'.join(file_name)
    seed_num = 36
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

    print(f"# of constructs: {params.num_constructs}\n# of questions: {params.num_questions}\n# of students: {params.num_students}")
    model_train()
