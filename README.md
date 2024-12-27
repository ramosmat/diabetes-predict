# Diabetes Prediction

This repository contains a machine learning project aimed at predicting the likelihood of diabetes in patients based on medical and lifestyle features. The project demonstrates preprocessing, analysis, and classification techniques using the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## Dataset

The dataset includes 768 observations with the following features:

- **Pregnancies**: Number of pregnancies.
- **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age in years.
- **Outcome**: Target variable (1 = Diabetes, 0 = No Diabetes).

## Key Libraries

The following Python libraries were used in this project:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For implementing machine learning models and preprocessing techniques.

## Machine Learning Models

The following classification algorithms were applied to predict diabetes:

- **Logistic Regression**
- **Random Forest Classifier**
<!-- - **Support Vector Machine (SVM)** -->

## Project Workflow

1. **Data Preprocessing**:
   - Normalizing data to ensure consistent scaling.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzing distributions of features like Glucose and BMI.
   - Identifying correlations between features and the target variable with Covariance Matrix.

3. **Feature Engineering**:
   - Creating derived features to enhance model performance.
   - Removing irrelevant or redundant features.

4. **Model Training and Evaluation**:
   - Splitting data into training and testing sets.
   - Training various classifiers and comparing their performance using metrics like accuracy, recall, and F1 score.
   - Hyperparameter tuning using GridSearchCV to optimize model performance.

5. **Results**:
   - In the Accuracy test, the Logistic Regression achieved the best performance with an accuracy of **77.06%**.
   - In the Recall test, the Random Forest Classifier achieved the best performance with an accuracy of **53.75%**.
   - In the F1-Score test, the Logistic Regression achieved the best performance with an accuracy of **61.31%**.

## Results Summary

Key findings from the project include:
- High glucose levels and BMI are strongly associated with the likelihood of diabetes.
- Age and diabetes pedigree function also contribute significantly to predictions.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/ramosmat/diabetes-predict.git
   ```

2. Navigate to the project directory:
   ```bash
   cd diabetes-predict
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook script.ipynb
   ```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
