# Ridge Regression on California Housing Dataset

A Jupyter Notebook project that implements and compares Ridge Regression using three approaches: closed-form solution, gradient descent, and scikit-learn's built-in `Ridge` estimator — all applied to the California Housing dataset.

## Project Structure

```
ridge_regression/
├── data/
│   └── california_housing.csv   # Auto-generated dataset
├── Model_training.ipynb         # Main notebook
└── README.md
```

## Notebook Overview

The notebook is organized into the following sections:

### 1. Imports & Setup
Loads all required libraries: NumPy, Pandas, Matplotlib, scikit-learn.

### 2. Data Preparation
Loads the California Housing dataset, saves it as a CSV, and checks for missing values. Performs EDA covering summary statistics, target distribution, correlation heatmap, and scatter plots. Features are standardized using `StandardScaler` and a bias term is added.

### 3. Closed-Form Ridge Regression
Implements Ridge Regression using the analytical solution, solving for optimal weights directly in one step via the closed-form formula.

### 4. Gradient Descent Ridge Regression
Implements Ridge Regression using iterative gradient descent, starting from zero weights and updating over 1000 iterations.

### 5. Evaluation & Comparison
Evaluates all three implementations (Closed-Form, Gradient Descent, Sklearn Ridge) using MSE and R² score, with a results table and bar chart. Baseline R² ≈ 0.576.

### 6. Lambda Experiment
Tests three regularization strengths (λ = 0.1, 1.0, 10.0) to observe how the penalty term affects model performance and coefficient shrinkage.

### 7. Learning Rate Experiment
Tests three learning rates (0.001, 0.01, 0.1) over 500 iterations to observe how step size affects gradient descent convergence.

### 8. Feature Engineering
Two sub-experiments:
- **Polynomial Features (degree 2)** — expands the feature space to 44 features, improving R² from 57.6% to 65.7%.
- **Correlation-Based Feature Selection** — selects the top 5 features by Pearson correlation; R² drops slightly to 51.3%, showing all 8 features contribute useful signal.

### 9. Model Interpretability (SHAP)
Uses SHAP (SHapley Additive exPlanations) to explain individual predictions and visualize feature importance via a summary plot.

### 10. Interactive Interface
Adds an `ipywidgets` dashboard with a styled slider for real-time exploration of how regularization strength λ affects model performance.

### 11. Outlier Handling
Removes outliers using the IQR method and retrains the model. MSE drops by ~43% and R² improves to 63.8%, confirming extreme values were distorting weight estimates.

### 12. Results & Discussion
Final comparison of all approaches. Polynomial features achieved the best R² (65.7%); outlier removal achieved the best MSE (0.3165). Concludes that data quality and feature engineering have greater impact than the choice of optimization method.

## Dataset

**California Housing** (from scikit-learn)
- 20,640 samples, 8 features
- Target: median house value (in $100,000s)

| Feature | Description |
|---|---|
| MedInc | Median income in block group |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

```bash
jupyter notebook Model_training.ipynb
```

Run all cells sequentially. The dataset CSV will be auto-generated on first run.
