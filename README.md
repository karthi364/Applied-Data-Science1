# Statistics and Trends Assignment (Submitted By: Karthik malidevaraju - 24083985)


This project analyzes the **Wine Quality Dataset** from Kaggle (https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) to explore the chemical properties of red wine and their relationship to quality ratings. The script generates three distinct plots and calculates statistical moments for the `alcohol` column.

## Dataset Overview
- **Source**: Red wine quality data with 1,599 instances.
- **Features**: 11 chemical properties (e.g., `fixed acidity`, `alcohol`) and a `quality` rating (3-8).
- **File**: `winequality-red.csv`

## Plots Generated
- **Relational Plot**: 
  - **Type**: Line Chart
  - **Description**: Displays the average alcohol content (% vol) across wine quality ratings (3-8). Points are marked with squares, and values are annotated below each point for clarity.
  - **File**: `relational_plot.png`

- **Categorical Plot**: 
  - **Type**: Horizontal Bar Chart
  - **Description**: Shows the count of wines for each quality rating (3-8). Bars use an earthy palette, with counts annotated inside for a sleek presentation.
  - **File**: `categorical_plot.png`

- **Statistical Plot**: 
  - **Type**: Correlation Heatmap
  - **Description**: Visualizes the correlation between all numerical features (e.g., `alcohol`, `pH`). Uses a warm `YlOrRd` color map with annotated coefficients.
  - **File**: `statistical_plot.png`

## Statistics Calculated
- **Column Analyzed**: `alcohol` (alcohol content in % vol)
- **Four Main Statistical Moments**:
  - **Mean**: 10.42
    - Represents the average alcohol content across all wines, indicating a central tendency around 10.42% vol. This suggests most wines have a moderate alcohol level, typical for red wines.
  - **Standard Deviation (Std Dev)**: 1.07
    - Measures the spread of alcohol content, showing a relatively narrow variation (about ±1.07% vol from the mean). This implies consistency in alcohol levels across the dataset.
  - **Skewness**: 0.86
    - Indicates a right skew (positive value > 0.2), meaning the distribution has a longer tail on the higher alcohol side. Some wines have unusually high alcohol content, but most are near or below the mean.
  - **Excess Kurtosis**: 0.20
    - Suggests a mesokurtic distribution (close to 0, < 0.2 threshold for leptokurtic). The tails are slightly heavier than a normal distribution, but not extreme, reflecting a typical spread without significant outliers.

## Preprocessing Steps
- **Initial Exploration**: 
  - Displayed the first 5 rows (`head`), last 5 rows (`tail`), summary statistics (`describe`), and correlation matrix (`corr`) to understand the data structure and relationships.
- **Missing Values**: 
  - Dropped rows with any NaN values to ensure a complete dataset (dataset is typically clean, but this step ensures robustness).
- **Numeric Conversion**: 
  - Converted all columns (`fixed acidity`, `volatile acidity`, etc.) to numeric types using `pd.to_numeric` with `errors='coerce'` to handle any non-numeric entries, followed by dropping any resulting NaN rows.
- **Type Adjustment**: 
  - Ensured the `quality` column is an integer type (`astype(int)`) for consistency in analysis and plotting.


---
