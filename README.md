# Causal Knowledge Tracing
NeurIPS 2022 CausalEdu Competition - Task 3

## Setups
The requiring environment is as bellow:  
- Linux 
- Python 3.9+
- PyTorch 1.12.1 
- Numpy 1.23.1
- Wandb 0.13.4

## Repository Overview 

```/data/``` - Repository containing the dataset

```/serialized_torch/``` - <TODO: Hunter, can you help me writing this?> 

```predict_graph.py``` - Training Causal KT

```construct_solution.py``` - Creating adjacency matrix for submission

```/submissions/final``` - Contains the model and the adjacency matrix for submission on the private leaderboard. 

## Running CausalKT
Here is an example how to train Causal KT. 
```
python3 predict_graph.py -WAB -L 1e-3
```

Here is an example how to create a submission file. We submitted L value of 0.45.
```
python3 construct_solution.py -f ./submissions/final/final_10_27_20_25_29_final_stretch_batch_64_epoch_50_embed_300 -L 0.45
```

Contact: ml4ed @ UMass Amherst
- Jaewook Lee (jaewooklee.jake@gmail.com)
- Hunter McNichols (hmcnich@gmail.com)
- Nischal Ashok Kumar (nischal.ashok09@gmail.com)
- Wanyong Feng (wanyongfeng123@gmail.com)