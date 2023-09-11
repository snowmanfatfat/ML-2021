from sklearn.model_selection import train_test_split
import csv
import numpy as np
import random

result1,result2=train_test_split([i for i in range(10)], test_size = 0.2, random_state = 0)
print(f'{result1}\n{result2}')

tr_path = 'covid.train.csv'
with open(tr_path, 'r') as fp:
    data = list(csv.reader(fp))
    data = np.array(data[1:])[:, 1:].astype(float)
print(data[[1,2]])

print(random.randint(0,1000000))