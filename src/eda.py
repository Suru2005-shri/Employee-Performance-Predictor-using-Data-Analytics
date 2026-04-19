"""
eda.py
------
Exploratory Data Analysis — generates and saves all key charts
that serve as proof/screenshots for GitHub and interviews.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ─── Style ───────────────────────────────────────────────────────────
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({
    'figure.facecolor' : '#1a1a2e',
    'axes.facecolor'   : '#16213e',
    'axes.labelcolor'  : 'white',
    'xtick.color'      : 'white',
    'ytick.color'      : 'white',
    'text.color'       : 'white',
    'grid.color'       : '#0f3460',
    'axes.titlecolor'  : 'white',
})
COLORS = ['#e94560', '#0f3460', '#533483', '#e8b046', '#07BEB8']
OUT    = '../images'
os.makedirs(OUT, exist_ok=True)


def load(path='data/hr_dataset.csv'):
    return pd.read_csv(path)


def plot_performance_dist(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['performance_label'].value_counts()
    bars   = ax.bar(counts.index, counts.values, color=COLORS[:3], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', color='white', fontsize=11)
    ax.set_title('Performance Label Distribution', fontsize=14, pad=12)
    ax.set_xlabel('Performance Label'); ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{OUT}/01_performance_distribution.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 1: Performance Distribution")


def plot_dept_performance(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df.groupby(['department', 'performance_label']).size().unstack(fill_value=0)
    data.plot(kind='bar', ax=ax, color=COLORS[:3], edgecolor='white', linewidth=0.5)
    ax.set_title('Performance by Department', fontsize=14)
    ax.set_xlabel('Department'); ax.set_ylabel('Count')
    ax.legend(title='Performance', facecolor='#1a1a2e', labelcolor='white')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/02_dept_performance.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 2: Department vs Performance")


def plot_training_vs_perf(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in zip(['High', 'Medium', 'Low'], COLORS):
        subset = df[df['performance_label'] == label]['training_hours']
        ax.hist(subset, bins=20, alpha=0.7, label=label, color=color, edgecolor='white')
    ax.set_title('Training Hours vs Performance', fontsize=14)
    ax.set_xlabel('Training Hours'); ax.set_ylabel('Frequency')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout()
    plt.savefig(f'{OUT}/03_training_vs_performance.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 3: Training vs Performance")


def plot_correlation_heatmap(df):
    num_cols = df.select_dtypes(include=np.number).drop(columns=['employee_id'])
    corr     = num_cols.corr()
    fig, ax  = plt.subplots(figsize=(11, 9))
    mask     = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, linewidths=0.5, linecolor='#1a1a2e',
                annot_kws={'size': 8}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT}/04_correlation_heatmap.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 4: Correlation Heatmap")


def plot_salary_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    order   = ['Low', 'Medium', 'High']
    data    = [df[df['performance_label'] == l]['salary'] for l in order]
    bp = ax.boxplot(data, patch_artist=True, labels=order,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title('Salary Distribution by Performance Level', fontsize=14)
    ax.set_xlabel('Performance Level'); ax.set_ylabel('Salary (USD)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/05_salary_boxplot.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 5: Salary Boxplot")


def plot_satisfaction_violin(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    order   = ['Low', 'Medium', 'High']
    parts = ax.violinplot(
        [df[df['performance_label'] == l]['satisfaction_score'] for l in order],
        positions=[1, 2, 3], showmeans=True
    )
    for pc, color in zip(parts['bodies'], COLORS):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(order)
    ax.set_title('Satisfaction Score by Performance Level', fontsize=14)
    ax.set_ylabel('Satisfaction Score')
    plt.tight_layout()
    plt.savefig(f'{OUT}/06_satisfaction_violin.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 6: Satisfaction Violin Plot")


def plot_feature_importance(path='../outputs/feature_importance.csv'):
    if not os.path.exists(path):
        print("⚠ feature_importance.csv not found — run train_model.py first")
        return
    fi  = pd.read_csv(path).head(12).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi['Feature'], fi['Importance'], color=COLORS[0], edgecolor='white', linewidth=0.3)
    ax.set_title('Top Feature Importances (Random Forest / Gradient Boosting)', fontsize=13)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{OUT}/07_feature_importance.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Chart 7: Feature Importance")


def run_all():
    df = load()
    plot_performance_dist(df)
    plot_dept_performance(df)
    plot_training_vs_perf(df)
    plot_correlation_heatmap(df)
    plot_salary_boxplot(df)
    plot_satisfaction_violin(df)
    plot_feature_importance()
    print(f"\n🖼  All charts saved to {OUT}/")


if __name__ == '__main__':
    run_all()
