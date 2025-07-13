# CodeAlpha_Task
dataset  

https://www.kaggle.com/datasets/laotse/credit-risk-dataset
## Overview
This project analyzes a credit risk dataset to predict loan default status using various machine learning and statistical techniques. The analysis includes data cleaning, exploratory data analysis (EDA), Principal Component Analysis (PCA), clustering, and predictive modeling with a Random Forest classifier.

## Dataset
The dataset (`credit_risk_dataset.xlsx`) contains the following features:
- **Quantitative Variables**:  
  `person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `loan_status`  
- **Qualitative Variables**:  
  `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`

## Project Structure
1. **Data Cleaning**:
   - Handling missing values (imputation by mean for quantitative variables and mode for qualitative variables).
   - Standardizing quantitative variables using `StandardScaler`.
   - Special handling for the `loan_int_rate` column (conversion from percentage strings to numeric values).

2. **Exploratory Data Analysis (EDA)**:
   - Histograms and boxplots to visualize distributions and detect outliers.
   - Correlation matrix heatmap to identify relationships between variables.

3. **Principal Component Analysis (PCA)**:
   - Dimensionality reduction to identify key components.
   - Visualization of eigenvalues, explained variance, and correlation circles.
   - Interpretation of principal components.

4. **Clustering**:
   - Hierarchical Agglomerative Clustering (CAH) and K-means clustering.
   - Comparison of clustering methods using silhouette scores and visualizations.
   - Characterization of clusters based on centroids.

5. **Predictive Modeling**:
   - Random Forest classifier to predict `loan_status` (default vs. non-default).
   - Evaluation metrics: precision, recall, F1-score, ROC-AUC, and precision-recall curves.
   - Optimization of prediction thresholds to maximize F1-score.

## Key Results
- **PCA**: The first two principal components explain a significant portion of the variance (exact percentage depends on the dataset).
- **Clustering**: Identified distinct customer segments based on income, age, and loan attributes.
- **Random Forest Model**:
  - Achieved high precision/recall (exact metrics depend on the dataset).
  - ROC-AUC score indicates strong discriminative power.

## Dependencies
- Python 3.x
- Libraries:  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xlrd`, `prince`

## Usage
1. **Data Preparation**:
   - Place the dataset (`credit_risk_dataset.xlsx`) in the `/content/` directory.
   - Run the notebook to preprocess the data (handling missing values, encoding, scaling).

2. **Analysis**:
   - Execute cells sequentially to perform EDA, PCA, clustering, and modeling.
   - Adjust parameters (e.g., number of clusters, PCA components) as needed.

3. **Evaluation**:
   - Review visualizations (correlation matrices, dendrograms, ROC curves).
   - Check classification reports and cluster statistics for insights.

## Example Outputs
- **Visualizations**:
  - Histograms and boxplots for feature distributions.
  - Heatmaps for correlation matrices.
  - Scatter plots for PCA projections and clustering results.
- **Model Metrics**:
  - Confusion matrix, ROC curve, and precision-recall curve.

## Conclusion
This project demonstrates a comprehensive approach to credit risk analysis, combining statistical techniques and machine learning to predict loan defaults. The results can inform risk assessment strategies and customer segmentation for financial institutions.

For detailed code and outputs, refer to the Jupyter notebook (`credit.ipynb`).
