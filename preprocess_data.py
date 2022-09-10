from unittest import result
import pandas as pd
from scipy import stats
import torch

def createDataset(filename: str):
    lessons_df = pd.read_csv(filename)
    # This is pretty aggresive cleaning, maybe others Types are useful
    checkin_df = lessons_df[lessons_df['Type'] == 'Checkin']
    # Clean out all the columns we don't need
    simple_df = checkin_df.iloc[:, [2, 5, 9]] # [S, 3]
    simple_df.loc[simple_df["IsCorrect"] == 0, "IsCorrect"] = -1 # [S, 3]
    
    # A little dirty importing a module just for mode
    num_questions = stats.mode(simple_df["UserId"]).count[0] # Q

    #C=num_constructs S=num_students Q=num_questions
    result = torch.tensor([]) # [1] -> [2, S, Q]
    for name, group in simple_df.groupby('UserId'):
        # Extract info for each user
        constructs = group["ConstructId"].values.tolist() # [C]
        correct = group["IsCorrect"].values.tolist() # [C]

        # Pad info
        num_constructs = len(constructs)
        pad_needed = num_questions - num_constructs # [P = Q - C]
        constructs += [0] * pad_needed # [Q]
        correct += [0] * pad_needed # [Q]

        # Build tensor
        student = torch.tensor([constructs, correct]) # [2, Q]
        student = torch.unsqueeze(student, 0) # [1, 2, Q]

        # Add to running result
        result = torch.cat([result, student]) # [i, 2, Q]
    return result # [S, 2, Q]
    
def main():
    #student_data = createDataset('data/sample_data_lessons_small.csv') # [S, 2, Q] Faster and better for debugging
    student_data = createDataset('data/Task_3_dataset/checkins_lessons_checkouts_training.csv') # [S, 2, Q]
    features = student_data[:, 0, :] # [S, Q]
    labels = student_data[:, 1, :] # [S, Q]

if __name__ == "__main__":
    # loaded_file = torch.load('serialized_torch/student_data_tensor.pt') # This is also an option
    main()