import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats

class TrainingDataset(Dataset):
    def __init__(self, constructs, labels):
        self.constructs = constructs
        self.labels = labels
        # Put number of construct here?
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        construct = self.constructs[idx]
        label = self.labels[idx]
        sample = {"Construct": construct, "Label": label}
        return sample

def createDataset(filename: str):
    lessons_df = pd.read_csv(filename)
    checkin_df = lessons_df[lessons_df['Type'] == 'Checkin']
    simple_df = checkin_df.iloc[:, [2, 5, 9]] 
    simple_df.loc[simple_df["IsCorrect"] == 0, "IsCorrect"] = -1 

    num_of_questions = stats.mode(simple_df["UserId"]).count[0]

    tot_construct_set = set()
    tot_construct_list = list()
    tot_label_list = list()
    for user, user_info in simple_df.groupby('UserId'):

        constructs = user_info["ConstructId"].values.tolist() # [C]
        labels = user_info["IsCorrect"].values.tolist() # [C]

        tot_construct_set.update(constructs)

        num_of_constructs = len(constructs)
        pad_needed = num_of_questions - num_of_constructs # [P = Q - C]

        constructs += [0] * pad_needed # [Q]
        labels += [0] * pad_needed # [Q]

        tot_construct_list.append(constructs)
        tot_label_list.append(labels)

    ######################
    ### How to store TD?
    ######################

    # 1) [Q, S] as in transposed format
    # TD = TrainingDataset(list(map(list, zip(*tot_construct_list))), list(map(list, zip(*tot_label_list))))
    # 2) [S, Q]
    TD = TrainingDataset(tot_construct_list, tot_label_list)

    # Display text and label.
    print('\nFirst iteration of data set: ', next(iter(TD)), '\n')
    # Print how many items are in the data set
    print('Length of data set: ', len(TD), '\n')
    # Print entire data set
    print('Entire data set: ', list(DataLoader(TD)), '\n')

    DL = DataLoader(TD, batch_size=2, shuffle=False)
    for (idx, batch) in enumerate(DL):
        # Print the 'text' data of the batch
        print(idx, 'Construct ', batch['Construct'])
        # Print the 'class' data of batch
        print(idx, 'Label: ', batch['Label'], '\n')

def main():
    createDataset('data/sample_data_lessons_small.csv')

if __name__ == "__main__":
    main()