import matplotlib.pyplot as plt

# Data for M=16
datasets = ['MNIST', 'GloVe', 'Deep-10M', 'SIFT-1M']
aq_recall = [0.7337, 0.4810, 0.5346, 0.6002]
adaptive_recall = [0.9936, 0.7910, 0.8823, 0.9702]

plt.figure(figsize=(10, 7))

# Plotting the slope chart
for i in range(len(datasets)):
    # Plot the line connecting AQ and Adaptive
    plt.plot([0, 1], [aq_recall[i], adaptive_recall[i]], marker='o', markersize=8, linewidth=2)
    
    # Add text labels for the exact values
    plt.text(-0.05, aq_recall[i], f'{aq_recall[i]:.2f}', ha='right', va='center', fontweight='bold')
    plt.text(1.05, adaptive_recall[i], f'{adaptive_recall[i]:.2f}', ha='left', va='center', fontweight='bold')
    
    # Label the dataset name along the slope
    plt.text(0.5, (aq_recall[i] + adaptive_recall[i])/2 + 0.02, datasets[i], 
             ha='center', fontsize=11, fontweight='bold', rotation=15)

# Formatting the chart
plt.xlim(-0.3, 1.3)
plt.xticks([0, 1], ['Base AQ', 'AQ + Adaptive'], fontsize=13, fontweight='bold')
plt.ylabel('Recall @ 10 Accuracy', fontsize=12, fontweight='bold')
plt.title('Jump in Recall', fontsize=15, pad=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('recall_slope_chart.png', dpi=300)
plt.show()