import torch
import numpy
from typing import List
from itertools import permutations

def build_student_tensor(num_constructs: int):
    """
    We assume an in order ground truth knowledge graph 
    A -> B -> C or 1 -> 2 -> 3 (except it's dense where 3 also is predicated on 1)
    Genearted students get progressively less smart
    Fake student 1 A Correct B Correct C Correct (1, 2, 3  => 1, 1, 1)
    Fake student 2 A Correct B Correct C Incorrect (1, 2, 3 => 1, 1, 0)
    ... and so on
    """

    student_knowledge = torch.ones(num_constructs)
    students = []
    for i in range(0, num_constructs+1):
        s = student_knowledge.tolist()
        students.append(s)
        final_concept = -1 - i
        if i == num_constructs:
            break
        student_knowledge[final_concept] = 0

    print(students)

    features = []
    labels = []
    question_perm  = list(permutations(range(0, num_constructs)))
    for student in students: # loop over students
        for questions in question_perm:
            q_features = []
            q_labels = []
            for q in questions:
                correct = False
                if student[q] == 1:
                    correct = True
                q_features.append(q)
                q_labels.append(int(correct))
            features.append(q_features)
            labels.append(q_labels)

    #print(features)
    #print(labels)
    return torch.tensor(features), torch.tensor(labels)

if __name__ == "__main__":
    #print(list)
    save_output = False
    features, labels = build_student_tensor(10)
    if save_output == True:
        torch.save(features, "./serialized_torch/gen_features_tensor.pt")
        torch.save(labels, "./serialized_torch/gen_labels_tensor.pt")
    print(features)
    print(labels)