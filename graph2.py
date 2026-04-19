import matplotlib.pyplot as plt
import numpy as np

# Datasets and their AQ+Adaptive Recall results for M=8 and M=16
labels = ['SIFT-1M', 'MNIST', 'GloVe', 'Deep-10M']
recall_m8 = [0.8274, 0.9536, 0.5648, 0.6427]
recall_m16 = [0.9702, 0.9936, 0.7910, 0.8823]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create the bars
rects1 = ax.bar(x - width/2, recall_m8, width, label='M = 8', color='#A9C9E2', edgecolor='black')
rects2 = ax.bar(x + width/2, recall_m16, width, label='M = 16', color='#1f77b4', edgecolor='black')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Recall @ 10', fontsize=12, fontweight='bold')
ax.set_title('Impact of Codebook Size (M) on Recall', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left')

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(False) # Clean background as requested
fig.tight_layout()

plt.savefig('m8_vs_m16_comparison.png', dpi=300)
plt.show()