import numpy as np
import random as rd
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def generate_matrix(colors, length):
    base_pattern = np.random.choice(colors, rd.choice(range(2, 6)))
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


def get_colormap(colors, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name, colors)
    return [tuple(int(255 * c) for c in cmap(i)[:3]) for i in range(colors)]


def render_image(m, colors, options=None, image_size=10, cmap_name='viridis'):
    height, width = m.shape
    colormap = get_colormap(colors+1, cmap_name)

    # Define layout: puzzle on top (80%), options on bottom (20%)
    # puzzle_area = 150
    # option_area = image_size - puzzle_area

    # Create empty RGB image
    img = Image.new('RGB', (image_size, image_size), color=(255, 255, 255))
    pixels = img.load()

    # Draw puzzle
    for y in range(height):
        for x in range(width):
            color_idx = m[y, x] - 1
            color = colormap[color_idx]
            # Scale each cell to fit puzzle area
            # cell_w = puzzle_area // width
            # cell_h = puzzle_area // height
            cell_w = image_size // width
            cell_h = image_size // height
            for dy in range(cell_h):
                for dx in range(cell_w):
                    px = x * cell_w + dx
                    py = y * cell_h + dy
                    pixels[px, py] = color

    # # Draw options (assume 4 options)
    # option_w = image_size // 4
    # for i, val in enumerate(options):
    #     color = colormap[val - 1]
    #     for y in range(option_area):
    #         for x in range(option_w):
    #             px = i * option_w + x
    #             py = puzzle_area + y
    #             pixels[px, py] = color

    return img


if __name__ == "__main__":
    length = 6
    colors = 6
    colormap = 'tab20'

    labels = {'img_path': [], 'label': []}

    for i in range(10000):

        m = generate_matrix(colors, length)

        index = np.random.choice(length, 2)
        m += 1
        correct = m[index[0], index[1]]

        # options = rd.sample([i for i in range(1, colors) if i != correct], 3)
        # options.append(correct)
        # rd.shuffle(options)


        m[index[0], index[1]] = 0

        img = render_image(m, colors, image_size=12, cmap_name=colormap)
        img.save(f'train_puzzles_only/{i}.png')
        img.close()

        # labels['label'].append(options.index(correct))
        labels['label'].append(correct - 1)
        labels['img_path'].append(f'train_puzzles_only/{i}.png')

    df = pd.DataFrame(labels)
    df.to_csv('train_puzzles_only.csv', index=False)
    
    test_labels = {'img_path': [], 'label': []}

    for i in range(1000):
        m = generate_matrix(colors, length)

        index = np.random.choice(length, 2)
        m = m + 1
        correct = m[index[0], index[1]]

        # options = rd.sample([i for i in range(1, colors) if i != correct], 3)
        # options.append(correct)
        # rd.shuffle(options)


        m[index[0], index[1]] = 0

        img = render_image(m, colors, image_size=12, cmap_name=colormap)
        # img.save(f'test_logic_images/{i}.png')
        img.save(f'test_puzzles_only/{i}.png')
        img.close()

        # test_labels['label'].append(options.index(correct))
        test_labels['label'].append(correct - 1)
        test_labels['img_path'].append(f'test_puzzles_only/{i}.png')


    df = pd.DataFrame(test_labels)
    df.to_csv('test_puzzles_only.csv', index=False)
    print('done')
