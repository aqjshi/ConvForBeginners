import numpy as np 
import pandas as pd
import os
from pathlib import Path

from tqdm import tqdm

import random
import torch

train_df = pd.read_csv(
    "clips-data-2020/train.csv", 
    na_values=['NA', '?'])



train_df['filename']="clips-"+ train_df["id"].astype(str)+".png"
train_df['FilePaths'] = "/kaggle/input/count-the-paperclips/clips-data-2020/clips/clips-"+train_df["id"].astype(str)+".png"

import matplotlib.pyplot as plt
f,a = plt.subplots(nrows=3, ncols=3,figsize=(13, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(a.flat):
    ax.imshow(plt.imread(train_df.FilePaths[i]))
    ax.set_title(train_df.clip_count[i])

ax.imshow(plt.imread(train_df.FilePaths[11]))
ax.set_title(train_df.clip_count[11]) 


plt.tight_layout()
plt.show()
