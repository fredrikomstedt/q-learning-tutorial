import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    x_list = [i for i in range(x)]
    ax.scatter(x_list, running_avg, color="C1")
    ax.set_xlabel("Training Steps", color="C1")
    ax.set_ylabel('Score', color="C1")
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', colors="C1")

    plt.savefig(filename)