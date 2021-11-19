import numpy as np
import pandas as pd


def LME(file):
    data = pd.read_excel(file).values
    # print(np.shape(data))
    n = np.shape(data)[1]-1
    gt = data[:, 0]
    for i in range(n):
        E = data[:, i+1]
        error = np.mean(np.abs((E-gt)/gt))
        print('{:.2%}'.format(error))


if __name__ == "__main__":
    LME('./input_data/test_lab2021-11-18.xlsx')