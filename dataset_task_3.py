import numpy as np
import pandas as pd
import torch
from torch.utils import data
import time
import torch
import random
# from utils import open_json, dump_json

def pivot_df(df, values, DEBUG=False):
    """
    Convert dataframe of question and answerrecords to pivoted array, filling in missing columns if some questions are 
    unobserved.
    """    
    data = df.pivot_table(index='QuestionId', columns='UserId', values=values, sort=False)
    # data = df.pivot_table(index='QuestionId', columns='UserId', values=values, fill_value=0, sort=False)

    print("Number of questions: ", data.shape[0])
    print("Number of students: ", data.shape[1])
    # data.replace(np.nan,0)
    if values == 'ConstructId':
        data.fillna(0, inplace=True)
        data = data.astype(int)
        if DEBUG:
            # Check how many students have answered questions.
            stutdent_count_data = data.astype(bool).sum(axis=0)
            p = plt.plot(stutdent_count_data.index, stutdent_count_data.values)
            plt.show()
    elif values == "IsCorrect":
        data.replace(to_replace=0.0, value=-1.0, inplace=True)
        data.fillna(0, inplace=True)
        data = data.astype(int)

    if DEBUG:
        print(values, "Table:")
        print(data)

    return data

def create_dataset(data_path: str, DEBUG=False):
    data_df = pd.read_csv(data_path)
    checkin_df = data_df[data_df['Type'] == 'Checkin'] # Only consider CheckIn.
    construct_data = pivot_df(checkin_df, 'ConstructId') # feature
    answer_data = pivot_df(checkin_df, 'IsCorrect') # label


# class TrainingDataset(Dataset):
#     def __init__(self, constructs, labels, tot_construct_set):
#         self.constructs = constructs
#         self.labels = labels
#         self.n_constructs = len(tot_construct_set)
#         self.unique_construct_list = list(tot_construct_set)
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         construct = self.constructs[idx]
#         label = self.labels[idx]
#         sample = {"Construct": construct, "Label": label}
#         return sample

# def createDataset(filename: str):
#     lessons_df = pd.read_csv(filename)
#     checkin_df = lessons_df[lessons_df['Type'] == 'Checkin']
#     simple_df = checkin_df.iloc[:, [2, 5, 9]] 
#     simple_df.loc[simple_df["IsCorrect"] == 0, "IsCorrect"] = -1 

#     num_of_questions = stats.mode(simple_df["UserId"]).count[0]

#     tot_construct_set = set()
#     tot_construct_list = list()
#     tot_label_list = list()
    
#     for user, user_info in simple_df.groupby('UserId'):

#         constructs = user_info["ConstructId"].values.tolist() # [C]
#         labels = user_info["IsCorrect"].values.tolist() # [C]

#         tot_construct_set.update(constructs)

#         num_of_constructs = len(constructs)
#         pad_needed = num_of_questions - num_of_constructs # [P = Q - C]

#         constructs += [0] * pad_needed # [Q]
#         labels += [0] * pad_needed # [Q]

#         tot_construct_list.append(constructs)
#         tot_label_list.append(labels)
#     tot_construct_set.add(0)
#     tot_serialized_construct_list = list()
#     unique_construct_list = list(tot_construct_set)
#     # print(unique_construct_list)
#     for constructs in tot_construct_list:
#         # print("constructs: ", constructs)
#         serialized_constructs = list(map(lambda x: unique_construct_list.index(x), constructs))
#         tot_serialized_construct_list.append(serialized_constructs)
#     TD = TrainingDataset(tot_serialized_construct_list, tot_label_list, tot_construct_set)

#     return TD