import matplotlib.pyplot as plt 
import json 
import numpy as np

with open('val_epochwise_loss.json', 'r') as infile:
    val_data = json.load(infile)

x, y = [], []
for i, val_loss in enumerate(val_data):
    print()
    if np.isnan(val_loss):
        continue
    x.append(i+1)
    y.append(val_loss)

plt.plot(x, y)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title('Validation Loss Plot')
plt.savefig('validation.png')