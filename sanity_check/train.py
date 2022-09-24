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

def visualize(features, labels):
    global C, Q, S
    # y1 = features.numpy()
    # y2 = labels.numpy()
    # x = np.array([i for i in range(Q)])
    # y = np.array([i for i in range(C)])
    # plt.xticks(x)
    # plt.yticks(y)
    # step = 0
    # for yy1, yy2 in zip(y1, y2):
    #     kcolors = ['red' if value == 0 else 'blue' for value in yy2]
    #     print("kcolors: \n", kcolors)
    #     xx1= [i + 0.1 * step for i in x]
    #     yy3 = [i + 0.1 * step for i in yy1]
    #     plt.plot(xx1, yy1, zorder=-1, alpha=0.150, lw=2)
    #     plt.scatter(xx1, yy1, c=kcolors, s=1)
    #     step +=1 
    # plt.savefig("./graph/dummy.png")

    for idx, (x1, x2) in enumerate(zip(features.tolist(), labels.tolist())):
        print("========" * 10)
        print(f"Student {idx}: ")
        print(x1)
        print(x2)
def test():
    global C, Q, S
    # C, Q, S = 10, 20, 50
    print(f"# of constructs: {C}\n# of questions: {Q}\n# of students: {S}")

    gt_dkt = GroundTruthPermutedDKT(n_concepts=C)
    # features = torch.randint(0, C, (S, Q))
    tmp = [[i for i in range(C)], [i for i in range(C)]]
    features = torch.tensor(tmp)
    labels = gt_dkt(features)
    # visualize(features, labels)
    # print("========="*10)
    # print("Data generation")
    # print("Features: \n", features)
    # print("Labels: \n", labels)
    # print("========="*10)

    # # Training

    # perm_dataset = True
    # gt_perm = list(np.random.permutation(C))
    # if perm_dataset:
    #     print("Ground truth permutation: ", gt_perm)
    #     for i, xx in enumerate(features):
    #         for j, yy in enumerate(xx):
    #             features[i][j] = gt_perm[features[i][j]]
    # training_set = createDataset(features, labels)
    # # Different seed number
    # torch.manual_seed(seed_num-1)
    # dkt = PermutedDKT(n_concepts=C, temperature=params.temperature, unroll=params.unroll).to(device)

    # training_loader = DataLoader(training_set, batch_size=params.batch_size, shuffle=False)
    # optimizer = torch.optim.Adam(dkt.parameters(), lr=0.01)
    # #0.01
   
    # n_epochs = 1
    # best_loss = 100.0
    # best_accuracy = 0.0
    # best_epoch = 0.0
    # # run = neptune.init(
    # #     project="phdprojects/challenge",
    # #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MTYwYjA3Zi01NmNhLTQ4YWMtOWFmMy0zMjdmZDliOGE4YzAifQ==",
    # #     capture_hardware_metrics = False
    # # )
    # for epoch in range(n_epochs): # loop over the dataset multiple times
    #     # print("Epoch ", epoch)
    #     train_loss=[]
    #     train_accuracy=[]
    #     for i, data in enumerate(training_loader, 0):
    #         b_construct = data['Features']
    #         b_label = data['Labels']
    #         optimizer.zero_grad()
    #         loss, acc = dkt(b_construct, b_label)
    #         train_accuracy.append(acc)
    #         train_loss.append(loss.item())
    #         loss.backward()
    #         optimizer.step()
    #     if (sum(train_loss)/len(train_loss) < best_loss):
    #         best_loss = sum(train_loss)/len(train_loss)
    #         best_accuracy = sum(train_accuracy)/len(train_accuracy)
    #         best_epoch = epoch
    #         # run['best_loss'] = best_loss
    #         # run['best_accuracy'] = best_accuracy
    #         # run['best_epoch'] = best_epoch
    #         # torch.save(dkt.state_dict(), './model/best_dkt.pt') 
    #         torch.save(dkt, './model/best_dkt.pt')
    #     # print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {sum(train_accuracy)/len(train_accuracy):.2f}")
    #     # run['loss'].log(sum(train_loss)/len(train_loss))
    #     # run['accuracy'].log(sum(train_accuracy)/len(train_accuracy))

    #     torch.save(dkt, './model/final_dkt.pt')

    # best_dkt = torch.load('./model/best_dkt.pt')
    # causal_order = []
    # for name, param in best_dkt.named_parameters():
    #     if (name == "gru.permuted_matrix.matrix"):
    #         # torch.set_printoptions(threshold=10_000)
    #         causal_order = sink_horn(param, params.temperature, params.unroll, verbose=True)
    # if perm_dataset: 
    #     print("Ground truth permutation:\n", gt_perm)
    # return causal_order == gt_perm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ML')
    # parser.add_argument('--model', type=str, default='ff', help='type')
    # hidden state -> number of constructs
    parser.add_argument('-C', '--num_constructs', type=int, default=5, help='number of constructs')
    parser.add_argument('-Q', '--num_questions', type=int, default=10, help='number of questions')
    parser.add_argument('-S', '--num_students', type=int, default=20, help='number of students')
    parser.add_argument('-T', '--temperature', type=int ,default=100, help='temperature')
    parser.add_argument('-B', '--batch_size', type=int ,default=1, help='batch size')
    parser.add_argument('-U', '--unroll', type=int, default=100, help='unroll')
    params = parser.parse_args()
    file_name = [params.num_constructs, params.num_questions, params.num_students, params.temperature, params.unroll]
    file_name =  [str(d) for d in file_name]
    params.file_name = '_'.join(file_name)
    seed_num = 37
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

    C, Q, S = 4, 4, 2
    # C, Q, S = 4, 50, 50
    # C, Q, S = 4, 50, 100
    test()
    # for i in range(1):
    #     print("========"*10)
    #     SEED = i
    #     is_true_order = test()
    #     if is_true_order:
    #         print("SEED: ", SEED)
    #     print("========"*10)