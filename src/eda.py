"""
eda.py
------
Generates 7 EDA charts saved as PNG to images/ folder.
Run standalone:  python src/eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel): return os.path.join(ROOT_DIR, rel)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#1a1a2e',
    'axes.facecolor'   : '#16213e',
    'axes.labelcolor'  : 'white',
    'xtick.color'      : 'white',
    'ytick.color'      : 'white',
    'text.color'       : 'white',
    'grid.color'       : '#0f3460',
    'axes.titlecolor'  : 'white',
    'axes.edgecolor'   : '#0f3460',
})
COLORS = ['#10d9a0', '#f5a623', '#f04060', '#4f8ef7', '#7c3aed']
OUT    = _path('images')


def load():
    path = _path('data/hr_dataset.csv')
    df   = pd.read_csv(path)
    print(f"  [eda] loaded {len(df)} rows")
    return df


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [eda] saved {name}")


def plot_performance_dist(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts  = df['performance_label'].value_counts()
    bars    = ax.bar(counts.index, counts.values, color=COLORS[:3],
                     edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', color='white', fontsize=11)
    ax.set_title('Performance Label Distribution', fontsize=14, pad=12)
    ax.set_xlabel('Performance Label')
    ax.set_ylabel('Count')
    _save(fig, '01_performance_distribution.png')


def plot_dept_performance(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = (df.groupby(['department', 'performance_label'])
              .size()
              .unstack(fill_value=0))
    data.plot(kind='bar', ax=ax, color=COLORS[:3], edgecolor='white', linewidth=0.5)
    ax.set_title('Performance by Department', fontsize=14)
    ax.set_xlabel('Department')
    ax.set_ylabel('Count')
    ax.legend(title='Performance', facecolor='#1a1a2e', labelcolor='white')
    plt.xticks(rotation=30, ha='right')
    _save(fig, '02_dept_performance.png')


def plot_training_vs_perf(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in zip(['High', 'Medium', 'Low'], COLORS):
        subset = df[df['performance_label'] == label]['training_hours']
        ax.hist(subset, bins=20, alpha=0.7, label=label, color=color, edgecolor='white')
    ax.set_title('Training Hours vs Performance', fontsize=14)
    ax.set_xlabel('Training Hours')
    ax.set_ylabel('Frequency')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    _save(fig, '03_training_vs_performance.png')


def plot_correlation_heatmap(df):
    num_cols = df.select_dtypes(include=np.number).drop(columns=['employee_id'])
    corr     = num_cols.corr()
    fig, ax  = plt.subplots(figsize=(11, 9))
    mask     = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, linewidths=0.5, linecolor='#1a1a2e',
                annot_kws={'size': 8}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontsize=14)
    _save(fig, '04_correlation_heatmap.png')


def plot_salary_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    order   = ['Low', 'Medium', 'High']
    data    = [df[df['performance_label'] == l]['salary'].values for l in order]
    bp = ax.boxplot(data, patch_artist=True, labels=order,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title('Salary Distribution by Performance Level', fontsize=14)
    ax.set_xlabel('Performance Level')
    ax.set_ylabel('Salary (USD)')
    _save(fig, '05_salary_boxplot.png')


def plot_satisfaction_violin(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    order   = ['Low', 'Medium', 'High']
    data    = [df[df['performance_label'] == l]['satisfaction_score'].values for l in order]
    parts   = ax.violinplot(data, positions=[1, 2, 3], showmeans=True)
    for pc, color in zip(parts['bodies'], COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(order)
    ax.set_title('Satisfaction Score by Performance Level', fontsize=14)
    ax.set_ylabel('Satisfaction Score')
    _save(fig, '06_satisfaction_violin.png')


def plot_feature_importance():
    fi_path = _path('outputs/feature_importance.csv')
    if not os.path.exists(fi_path):
        print(f"  [eda] Skipping chart 7 — run train_model.py first")
        return
    fi  = pd.read_csv(fi_path).head(12).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi['Feature'], fi['Importance'], color=COLORS[0],
            edgecolor='white', linewidth=0.3)
    ax.set_title('Top Feature Importances', fontsize=13)
    ax.set_xlabel('Importance Score')
    _save(fig, '07_feature_importance.png')


def run_all():
    print("\n[EDA] Generating charts...")
    df = load()
    plot_performance_dist(df)
    plot_dept_performance(df)
    plot_training_vs_perf(df)
    plot_correlation_heatmap(df)
    plot_salary_boxplot(df)
    plot_satisfaction_violin(df)
    plot_feature_importance()
    print(f"[EDA] All charts saved to '{OUT}'\n")


if __name__ == '__main__':
    run_all()
