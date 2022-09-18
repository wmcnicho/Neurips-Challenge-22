import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import pandas as pd
import torch
import os
import argparse
import heapq
# from utils import open_json, dump_json, compute_auc, compute_accuracy
import time

from dataset_task_3 import create_dataset
# from model_task_3 import PermutedDKT
DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_information = []

if __name__ == "__main__":

    seedNum = 221
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

    data_path = os.path.normpath('data_task_3/Task_3_dataset/checkins_lessons_checkouts_training.csv')
    train_data, valid_data = create_dataset(data_path)

    print(len(train_data))
    print(len(valid_data))