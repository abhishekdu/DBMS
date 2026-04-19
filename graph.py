import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data for M=16
methods = ["PQ", "OPQ", "AQ", "AQ+Adp"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

data_m16 = {
    "SIFT-1M": {
        "recall": [0.5489, 0.5128, 0.6002, 0.9702],
        "latency": [0.636, 0.637, 19.481, 43.192], "marker": "o"
    },
    "MNIST": {
        "recall": [0.6194, 0.6926, 0.7337, 0.9936],
        "latency": [0.022, 0.023, 1.644, 11.414], "marker": "s"
    },
    "GloVe": {
        "recall": [0.3009, 0.3934, 0.4810, 0.7910],
        "latency": [0.119, 0.117, 0.212, 0.357], "marker": "^"
    },
    "Deep-10M": {
        "recall": [0.3264, 0.3960, 0.5346, 0.8823],
        "latency": [3.042, 3.042, 3.111, 7.023], "marker": "D"
    }
}

plt.figure(figsize=(14, 8))

for dataset, info in data_m16.items():
    # Plot line connecting the methods
    plt.plot(info['latency'], info['recall'], color='gray', alpha=0.15, zorder=1)
    
    for i in range(len(methods)):
        # Plot the point
        plt.scatter(info['latency'][i], info['recall'][i],
                    color=colors[i], marker=info['marker'], s=200,
                    edgecolors='black', alpha=0.8, zorder=3)
        
        # Add the Name Label next to the point
        plt.text(info['latency'][i] * 1.05, info['recall'][i] + 0.01, 
                 methods[i], fontsize=9, fontweight='bold', alpha=0.7)

plt.xscale('log')
plt.xlabel('Search Latency (ms) - Log Scale', fontsize=12, fontweight='bold')
plt.ylabel('Recall @ 10 (Accuracy)', fontsize=12, fontweight='bold')
plt.title('Global Comparative Study: Accuracy vs Latency (M=16)', fontsize=15, pad=20)

# Dataset Legend
dataset_leg = [Line2D([0], [0], marker=info['marker'], color='w', label=d, 
                       markerfacecolor='gray', markersize=10, markeredgecolor='black') 
               for d, info in data_m16.items()]
plt.legend(handles=dataset_leg, loc='lower right', title="Datasets", fontsize=10)

# Disable the grid lines
plt.grid(False)

plt.ylim(0, 1.1)
plt.tight_layout()

plt.savefig('labeled_merged_study_no_grid.png', dpi=300)
plt.show()