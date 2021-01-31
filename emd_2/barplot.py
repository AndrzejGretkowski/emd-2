import numpy as np
import matplotlib.pyplot as plt


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def score_to_percent(score):
    return int(score * 100)


def plot_graph(scores):
    ind = np.arange(len(scores))
    width = 0.35  # the width of the bars

    labels, balanced, precision = [], [], []
    for l, (b, p) in sorted(scores.items(), key=lambda x: x[1][0]):
        labels.append(l)
        balanced.append(score_to_percent(b))
        precision.append(score_to_percent(p))
    fig, ax = plt.subplots()
    rect1 = ax.bar(ind - width/2, balanced, width, label='balanced')
    rect2 = ax.bar(ind + width/2, precision, width, label='precision')

    ax.set_ylabel('Scores [%]')
    ax.set_title('Scores by classifier')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rect1, ax)
    autolabel(rect2, ax)

    plt.show()