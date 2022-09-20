import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import pandas as pd
import torch
from torch.utils import data
import os
import argparse
import heapq
# from utils import open_json, dump_json, compute_auc, compute_accuracy
import time

from dataset_task_3 import create_dataset, f_collate
from model_task_3 import PermutedDKT

import neptune.new as neptune

DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_information = []

def train_model():
    global epoch_information
    # dkt.train()
    for idx, batch in enumerate(train_loader):
        # print("BATCH: ", batch, type(batch))
        optimizer.zero_grad()
        dkt(batch)
        # loss.backward()
        # optimizer.step()
if __name__ == "__main__":

    seedNum = 221
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

    data_path = os.path.normpath('data_task_3/Task_3_dataset/checkins_lessons_checkouts_training.csv')
    # train_dataset, valid_dataset = create_dataset(data_path, DEBUG=True)
    task_dataset = create_dataset(data_path, DEBUG=True)
    unique_const_list = task_dataset.unique_const_list

    if DEBUG:
        for idx, dataset_dict in enumerate(task_dataset):
            print("idx: ", idx, "\t", dataset_dict['construct'], dataset_dict['answer'])
        print("length of the dataset is:", len(task_dataset))
        print("number of constructs is:", len(unique_const_list))
    train_size = int(0.8 * len(task_dataset))
    valid_size = len(task_dataset) - train_size

    train_dataset, valid_dataset = data.random_split(task_dataset, [train_size, valid_size])

    collate_fn = f_collate()
    num_workers = 3
    # print(train_dataset)
    # for idx, tmp in enumerate(train_dataset):
    #     print(tmp)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=4, num_workers=0, shuffle=True, drop_last=True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=True)
    dkt = PermutedDKT(n_constructs=len(unique_const_list)).to(device)
    optimizer = torch.optim.Adam(dkt.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in range(1):
        train_model()
    end_time = time.time()
    print("Time Elapsed: {} hours".format((end_time-start_time)/3600.))