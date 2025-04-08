import numpy as np
import random as rd
import matplotlib.pyplot as plt

def generate_matrix(colors, length):
    base_pattern = np.random.choice(colors, rd.choice(range(2, length**2 // 2)))
    base_pattern = np.tile(base_pattern, length**2 // len(base_pattern))

    if len(base_pattern) < length**2:
        base_pattern = np.append(base_pattern, base_pattern[:length**2 - len(base_pattern)])

    return base_pattern.reshape(length, length)


def shift_matrix(matrix, type):
    match type:
        case 'left':
            return np.roll(matrix, shift=-1, axis=1)
        case 'right':
            return np.roll(matrix, shift=1, axis=1)
        case 'up':
            return np.roll(matrix, shift=-1, axis=0)
        case 'down':
            return np.roll(matrix, shift=1, axis=0)

length = 6
colors = 10
colormap = 'tab20'
m = generate_matrix(colors, length)
print(m)
# m = shift_matrix(shift_matrix(m, 'up'), 'right')
# print(m)
index = np.random.choice(length, 2)
m = m + 1
correct = m[index[0], index[1]]
m[index[0], index[1]] = 0
print(m)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].imshow(m, cmap=colormap, vmin=0, vmax=colors)
ax[0].set_title('Puzzle')
ax[0].axis('off')

options = np.append(rd.sample(range(1, colors), 3), correct)
rd.shuffle(options)
ax[1].imshow([options], cmap=colormap, vmin=0, vmax=colors)
ax[1].set_title('Options')
ax[1].axis('off')

plt.show()
