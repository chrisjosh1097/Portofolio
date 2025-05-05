import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison_boxplots(base_scores, tuned_scores):
    """
    Plots box and strip comparison for base vs tuned model scores across metrics.
    Inputs should be dicts like:
        {
            'MSE': [...],
            'MAE': [...],
            'R²': [...]
        }
    """
    data = []
    for metric in ['MSE', 'MAE', 'R²']:
        for score in base_scores[metric]:
            data.append({'Metric': metric, 'Score': score, 'Model': 'Base'})
        for score in tuned_scores[metric]:
            data.append({'Metric': metric, 'Score': score, 'Model': 'Tuned'})
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for i, metric in enumerate(['MSE', 'MAE', 'R²']):
        ax = axes[i]
        subset = df[df['Metric'] == metric]
        sns.boxplot(data=subset, x='Model', y='Score', hue='Model', palette='Set2', ax=ax, legend=False)
        sns.stripplot(data=subset, x='Model', y='Score', color='black', alpha=0.5, jitter=True, ax=ax)
        ax.set_title(metric)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.suptitle("Cross-Validation Performance: Base vs Tuned", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def plot_comparison_lineplots(base_scores, tuned_scores):
    """
    Plots line comparison for base vs tuned model scores across folds.
    Uses colors consistent with Seaborn's Set2 palette.
    """
    metrics = ['MSE', 'MAE', 'R²']
    n_folds = len(next(iter(base_scores.values())))
    folds = list(range(1, n_folds + 1))

    # Get consistent Set2 colors
    palette = sns.color_palette('Set2')
    base_color = palette[0]
    tuned_color = palette[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(folds, base_scores[metric], marker='o', label='Base', color=base_color)
        ax.plot(folds, tuned_scores[metric], marker='s', label='Tuned', color=tuned_color)
        ax.set_title(metric)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    fig.suptitle("Fold-wise Performance Comparison: Base vs Tuned", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()