# User Purchase Classification Prediction Using SVM

This project demonstrates the application of the **Support Vector Machine (SVM)** algorithm to predict user purchase behavior. By leveraging the `user-data.csv` dataset, we aim to build a robust SVM classifier capable of determining whether a user will purchase a particular product based on their features.

## Project Overview
The core objective of this project is to explore the capabilities of SVM for binary classification tasks. The implementation includes essential steps such as data preprocessing, model training, evaluation, and visualization of results.

---

## Dataset Description
The `user-data.csv` file serves as the primary dataset for this project. It contains 400 entries with the following columns:

- **user_id:** Unique identifier for each user.
- **gender:** Gender of the user (Male/Female).
- **age:** Age of the user.
- **estimated_salary:** Estimated annual salary of the user.
- **purchased:** Target variable indicating purchase behavior (`1` for Yes, `0` for No).

For this project, the focus is on the `age` and `estimated_salary` as features, and `purchased` as the target variable.

---

## Implementation Steps

### 1. Data Preprocessing
To ensure optimal model performance, the following preprocessing steps are performed:

1. **Data Import:**
   - The dataset is imported using the `pandas` library.

2. **Feature and Target Extraction:**
   - Independent variables: `age` and `estimated_salary`.
   - Dependent variable: `purchased`.

3. **Dataset Splitting:**
   - The data is divided into training (75%) and test (25%) sets using the `train_test_split` function from `sklearn.model_selection`.

4. **Feature Scaling:**
   - Standardization is applied to features using the `StandardScaler` class from `sklearn.preprocessing` to ensure all values are on a comparable scale.

---

### 2. SVM Algorithm Implementation

The SVM classifier is built using the `SVC` class from `sklearn.svm`. Key details include:

- **Kernel Selection:**
  - A linear kernel is chosen for its effectiveness with linearly separable data.

- **Training:**
  - The model is trained using the `fit` method on the training set.

---

### 3. Predicting Test Set Results

- The trained classifier is evaluated on the test set using the `predict` method.
- Predicted values are compared with actual target values to gauge performance.

---

### 4. Performance Evaluation: Confusion Matrix

- A confusion matrix is created using the `confusion_matrix` function from `sklearn.metrics`.
- The matrix provides insights into:
  - **True Positives (TP):** Correctly predicted purchases.
  - **True Negatives (TN):** Correctly predicted non-purchases.
  - **False Positives (FP):** Incorrectly predicted purchases.
  - **False Negatives (FN):** Missed predictions for purchases.

Additionally, accuracy scores are calculated to summarize overall performance.

---

### 5. Visualization

#### Training Set Visualization
- The decision boundary of the SVM classifier is visualized using:
  - **`contourf` Function:** Displays the classifier’s decision regions.
  - **`scatter` Function:** Plots the training data points, highlighting class separation.

#### Test Set Visualization
- The test set results are visualized similarly, providing a clear depiction of the classifier’s performance on unseen data.

- Both visualizations are enhanced with:
  - Titles
  - Axis labels
  - Legends for better interpretability.

---

## Results and Conclusion
The SVM classifier demonstrates high accuracy in predicting user purchase classifications based on the given features. The project underscores the utility of SVM for binary classification tasks and highlights the importance of visualization in understanding model behavior.

---

## Dependencies
The following Python libraries are used in this project:

- **Numpy:** Numerical operations and array handling.
- **Pandas:** Data manipulation and analysis.
- **Matplotlib:** Data visualization.
- **Seaborn:** Enhanced visualizations.
- **Scikit-learn:** Machine learning algorithms and utilities.
  - `model_selection`
  - `preprocessing`
  - `svm`
  - `metrics`


---

This project is a practical demonstration of machine learning techniques for classification tasks, specifically showcasing the power of Support Vector Machines.

