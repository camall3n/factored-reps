import matplotlib.pyplot as plt
import numpy as np

def add_heatmap_labels(a, ax=None):
    if ax is None:
        ax = plt.gca()
    n_rows, n_cols = a.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, np.round(a[i, j], 2), ha="center", va="center", color="w")

def diagonalize(arr):
    """Exchange rows to greedily diagonalize a 2-D array"""
    out = np.copy(arr)
    max_pos = min(out.shape)
    # print(max_pos)
    labels = np.arange(len(out))
    # iterate down the diagonal
    for pos in range(max_pos):
        # grab all values in current column that haven't been placed yet
        column_values = out[pos:, pos]
        # find index of the largest value
        largest_value_idx = np.argmax(column_values) + pos
        # grab the two rows we want to swap
        largest_value_row = np.copy(out[largest_value_idx, :])
        current_row = np.copy(out[pos, :])
        # swap the rows
        out[pos, :] = largest_value_row
        out[largest_value_idx, :] = current_row
        # swap labels
        labels[largest_value_idx], labels[pos] = labels[pos], labels[largest_value_idx]
    return out, labels
