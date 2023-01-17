#!/usr/bin/env python 

import pandas as pd
import numpy as np
import glob

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
num_files = len(all_filenames)
print("num_files =", num_files)


df_from_each_file = [pd.read_csv(f, header=None) for f in all_filenames]

frame = pd.concat(df_from_each_file, axis=1)
frame.iloc[0] = (np.linspace(0, 4*num_files - 1, 4*num_files) / 4 + 1).astype(int)

print(frame)

frame.to_csv("collapse.csv", header=False, index=False)
