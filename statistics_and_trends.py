"""
Wine Quality Dataset Exploration Script
- Analyzes chemical properties and quality ratings
- Features: Line chart (relational), horizontal bar (categorical), heatmap (statistical)
- Computes statistical moments for 'alcohol'


Student Details:
Name: Karthik malidevaraju 
Roll Number: 24083985
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# New vineyard-inspired earthy palette
COLORS = ['#5C4033', '#7A5C47', '#98755A', '#B68E6D', '#D4A780', '#F2C093']

def plot_relational_plot(df):
    """Plots average alcohol content by quality as a line chart"""

    fig = plt.figure(figsize=(10, 5), facecolor='#FFF5E6')
    ax = fig.add_subplot(111)
    avg_data = df.groupby('quality')['alcohol'].mean()
    
    ax.plot(avg_data.index, avg_data.values, color=COLORS[0], marker='s', 
            linestyle='-', lw=2, alpha=0.9)
    
    # Minimalist styling
    ax.set_title('Alcohol vs Quality', fontsize=20, color=COLORS[1], pad=15, 
                 family='sans-serif', weight='light')
    ax.set_xlabel('Quality', fontsize=14, color=COLORS[2], family='sans-serif')
    ax.set_ylabel('Alcohol (%)', fontsize=14, color=COLORS[2], family='sans-serif')
    ax.tick_params(axis='both', colors=COLORS[3], labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS[4])
    ax.spines['bottom'].set_color(COLORS[4])
    
    # Subtle grid
    ax.grid(True, ls=':', alpha=0.3, color=COLORS[5])
    
    # Annotations below points
    for x, y in zip(avg_data.index, avg_data.values):
        ax.text(x, y - 0.1, f'{y:.1f}', ha='center', va='top', fontsize=10, fontweight='bold',
                color=COLORS[1], alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor())
    plt.close()


def plot_categorical_plot(df):
    """Plots wine quality distribution as a horizontal bar chart"""

    fig = plt.figure(figsize=(8, 6), facecolor='#FFF5E6')
    ax = fig.add_subplot(111)
    counts = df['quality'].value_counts().sort_index()
    
    bars = ax.barh(counts.index, counts.values, color=COLORS[2], 
                   edgecolor=COLORS[0], height=0.5, alpha=0.85)
    
    # Modern, clean styling
    ax.set_title('Quality Distribution', fontsize=20, color=COLORS[1], 
                 pad=15, family='sans-serif', weight='light')
    ax.set_xlabel('Count', fontsize=14, color=COLORS[2], family='sans-serif')
    ax.set_ylabel('Quality', fontsize=14, color=COLORS[2], family='sans-serif')
    ax.tick_params(axis='both', colors=COLORS[3], labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS[4])
    ax.spines['bottom'].set_color(COLORS[4])
    
    # No grid for a cleaner look
    ax.grid(False)
    
    # Annotations inside bars for a sleek design
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 40, bar.get_y() + bar.get_height() / 2, f'{int(width)}', 
                ha='right', va='center', fontsize=12, color='black', 
                fontweight='bold', alpha=0.9)
    
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor())
    plt.close()


def plot_statistical_plot(df):
    """Plots a correlation heatmap of wine features"""

    fig = plt.figure(figsize=(10, 8), facecolor='#FFF5E6')
    ax = fig.add_subplot(111)
    corr = df.corr()
    
    sns.heatmap(corr, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax, 
                linewidths=0.2, linecolor=COLORS[5], 
                cbar_kws={'label': 'Corr'})
    
    # Simplified, elegant styling
    ax.set_title('Feature Correlations', fontsize=20, color=COLORS[1], 
                 pad=15, family='sans-serif', weight='light')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_color(COLORS[2])
        label.set_rotation(45 if label in ax.get_xticklabels() else 0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10, colors=COLORS[3])
    ax.collections[0].colorbar.ax.set_ylabel('Corr', fontsize=12, color=COLORS[2])
    
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor())
    plt.close()


def statistical_analysis(df, col):
    """Computes statistical moments for a specified column"""

    mean_val = np.mean(df[col])
    std_val = np.std(df[col])
    skew_val = stats.skew(df[col])
    kurt_val = stats.kurtosis(df[col])
    return mean_val, std_val, skew_val, kurt_val


def preprocessing(df):
    """Cleans and formats the wine quality dataset"""

    print("--- Initial Data ---".center(50))
    print(df.head(), "\n")
    print("--- Last Rows ---".center(50))
    print(df.tail(), "\n")
    print("--- Summary Stats ---".center(50))
    print(df.describe(), "\n")
    print("--- Correlations ---".center(50))
    print(df.corr(), "\n")
    
    df_clean = df.dropna()
    
    numeric_fields = ['fixed acidity', 'volatile acidity', 'citric acid', 
                      'residual sugar', 'chlorides', 'free sulfur dioxide', 
                      'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                      'alcohol', 'quality']
    for field in numeric_fields:
        df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
    
    df_clean = df_clean.dropna()
    df_clean['quality'] = df_clean['quality'].astype(int)
    
    return df_clean


def writing(moments, col):
    """Displays and interprets statistical moments"""

    print(f"Statistics for '{col}':")
    print(f"  Mean: {moments[0]:.2f}")
    print(f"  Std Dev: {moments[1]:.2f}")
    print(f"  Skewness: {moments[2]:.2f}")
    print(f"  Excess Kurtosis: {moments[3]:.2f}")
    
    skew_desc = 'right' if moments[2] > 0.2 else 'left' if moments[2] < -0.2 else 'not'
    kurt_desc = 'leptokurtic' if moments[3] > 0.2 else 'platykurtic' if moments[3] < -0.2 else 'mesokurtic'
    print(f"  Shape: {skew_desc} skewed, {kurt_desc}")


def main():
    """Runs the wine quality analysis pipeline."""

    data = pd.read_csv('winequality-red.csv')
    
    processed_data = preprocessing(data)

    # Chosen alcohol column
    analysis_col = 'alcohol'

    plot_relational_plot(processed_data)
    plot_statistical_plot(processed_data)
    plot_categorical_plot(processed_data)
    moments = statistical_analysis(processed_data, analysis_col)
    writing(moments, analysis_col)


if __name__ == '__main__':
    main()