import numpy as np

from Lorenz_data import data_of_lorenz as data

with  open('dataset.txt', 'w') as f:
    f.write(f'[10. 28. {8./3.}]\n')
    for i in range(0, 4000, 100):
        f.write(f'{np.append(data.states[i], (0.01 * i))}\n')
    f.close()