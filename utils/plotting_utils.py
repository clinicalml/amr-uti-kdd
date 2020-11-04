from matplotlib import pyplot as plt

def plot_frontier(frontier, label=None):
    plt.scatter(frontier['iat'],
                frontier['broad'],
                label=label, s=60)
 