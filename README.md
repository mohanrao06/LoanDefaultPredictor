# LOAN DEFAULT PREDICTION USING MACHINE LEARNING

## An End-to-End Machine Learning Classification Project

## Submitted in partial fulfilment of the requirements for the award of the degree of

## BACHELOR OF TECHNOLOGY

## in

## COMPUTER SCIENCE AND ENGINEERING

## Submitted by

```
Gedala Mohan Rao
```
## Under the guidance of

```
Dr. Mrinalini Rana
Assistant Professor
```
```
Department of Computer Science and Engineering
Lovely Professional University
```
```
Academic Year: 2024 – 2025
```
```
GitHub Repository: LoanDefaultPredictor
Python | Pandas | Scikit-learn | SMOTE | ML Models
```

## ABSTRACT

Loan default prediction is a critical task for financial institutions as inaccurate risk assessment can
lead to significant financial losses. This project focuses on developing an end-to-end machine
learning solution to predict loan default using historical borrower data. The dataset contains
demographic, financial, and credit-related attributes, which were extensively analyzed through
data profiling and exploratory data analysis to understand patterns, distributions, and data quality
issues.

Data preprocessing steps included handling missing values, encoding categorical variables, feature
scaling, and addressing class imbalance using the Synthetic Minority Over-sampling Technique
(SMOTE) applied only to the training dataset to prevent data leakage. Multiple machine learning
models were implemented and benchmarked, including Logistic Regression, Naïve Bayes,
Decision Tree, Random Forest, and Gradient Boosting

The models were evaluated using key performance metrics such as Accuracy, Precision, Recall,
F1-score, and Area Under the ROC Curve (AUC). Logistic Regression emerged as the final model
due to its superior performance, achieving an AUC of 0.995 and high recall, which is critical for
identifying high-risk borrowers. The results demonstrate that a well-preprocessed dataset
combined with robust evaluation techniques can significantly improve loan default prediction and
support better decision-making in financial risk management.


## Contents

- Introduction
- Problem Statement
- Objectives
- Dataset Description
- Data Profiling & Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Building
- Model Evaluation
- Final Model Selection & Justification
- Results & Discussion
- Conclusion
- Future Scope
- References
- Appendix


## Introduction

```
In today’s financial ecosystem, lending institutions face significant challenges in assessing the
creditworthiness of loan applicants. With the rapid increase in digital lending platforms, banks
and financial organizations must process large volumes of loan applications efficiently while
minimizing the risk of default. Inaccurate risk assessment can lead to financial losses, reduced
profitability, and increased non-performing assets.
```
```
Traditional credit evaluation methods often rely on manual analysis and limited rule-based
systems, which may fail to capture complex patterns present in large-scale financial data. This
has created a need for intelligent, data-driven approaches that can assist lenders in making
informed and consistent decisions.
```
```
Machine Learning (ML) techniques provide a powerful solution to this problem by analysing
historical loan data and identifying patterns associated with borrower behaviour. By leveraging
ML models, financial institutions can predict the likelihood of loan default more accurately,
improve decision-making efficiency, and enhance overall risk management.
```
```
This project focuses on building and evaluating multiple machine learning classification models
to predict loan default risk. The objective is to compare different algorithms, handle class
imbalance effectively, and identify the most reliable model based on performance metrics such
as accuracy, recall, F1-score, and AUC. The final outcome of this study aims to demonstrate how
data science techniques can support smarter and more transparent lending decisions.
```

## Problem Statement

```
Financial institutions face significant challenges in accurately assessing the credit risk of loan
applicants. Traditional credit evaluation methods often fail to effectively predict loan defaults
due to large data volumes, class imbalance, and complex relationships between financial and
demographic variables. Incorrect risk assessment can lead to increased non-performing assets
and financial losses.
```
```
Therefore, there is a need for a data-driven and reliable loan default prediction system that can
analyse historical loan data, identify key risk factors, and accurately predict the probability of
default to support informed lending decisions.
```

## Objectives

- To efficiently preprocess and manage a large-scale, real-world financial dataset (~300 MB) by
    removing irrelevant features, handling missing values, and preventing data leakage to ensure
    reliable model training.
- To engineer meaningful and high-impact features that capture borrower risk behaviour, including
    time-based credit history metrics and customer segmentation using unsupervised learning
    techniques.
- To benchmark and optimize multiple machine learning classification models by addressing class
    imbalance using SMOTE and applying cross-validation–based hyperparameter tuning.
- To develop a high-performing loan default prediction model that maximizes the identification of
    high-risk borrowers while maintaining strong overall discriminatory power.
- To enhance model transparency and business usability by interpreting model outputs and
    identifying the most influential risk factors to support informed lending and underwriting
    decisions.


## Dataset Description

```
Data Source
The dataset used in this project is the Lending Club Accepted Loans Dataset , obtained from
Kaggle. It contains historical loan application and repayment information for loans issued
between 2007 and Q4 2018.
```
- **Total Records:** ~2.2 million loans
- **Original Features:** 151 columns
- **Dataset Size:** ~300 MB (compressed)
- **Data Type:** Real-world financial and credit risk data

```
This dataset represents one of the most comprehensive publicly available sources for consumer
credit risk modelling.
```
```
Target Variable
The objective of this project is to predict loan default risk.
The original loan_status variable contained multiple loan outcome categories. To simplify
modeling and ensure business relevance, it was converted into a binary target variable:
```
```
To prevent data leakage, loans with incomplete or ongoing statuses such as Current , Issued , and
In Grace Period were removed, ensuring the model was trained only on loans with known final
outcomes.
```
```
Target
Class
```
```
Loan Status Mapping Description
```
```
1 (Risk) Charged Off, Default, late
(31–120 days)
```
```
Indicates a non-performing
loan resulting in financial loss
0 (Safe) Fully Paid^ Indicates a successfully repaid
loan
```

**Key Predictive Features**
From the original 151 features, only the most relevant and high-quality variables were retained
after extensive cleaning, missing value analysis, and feature selection

These features were grouped into the following categories:
**Loan Characteristics**

- loan_amnt
- int_rate
- installment
- term
These variables define the loan amount, interest rate, and repayment structure.
**Credit Risk Indicators**
- grade, sub_grade
- fico_range_high
- dti
These features directly represent borrower creditworthiness and debt burden.
**Income and Employment Attributes**
- annual_inc
- emp_length
- home_ownership
- verification_status
These variables provide insight into the borrower’s financial stability and repayment capacity.
**Engineered Features**
- Credit_History_Length_Years
- Loan_to_Income_Ratio
- Cluster_ID
These features were created during feature engineering and unsupervised learning phases to
capture complex borrower risk patterns.
**Date-Based Features**
- issue_d
- earliest_cr_line
These were used to derive time-based features and subsequently removed from modeling.


## Data Profiling & Exploratory Data Analysis (EDA)

```
Objective of EDA and Data Profiling
Before applying any preprocessing or modelling techniques, a detailed exploratory data analysis
(EDA) and automated data profiling were performed to gain insights into the structure, quality,
and behaviour of the dataset.
The goals of this phase were:
```
- To understand feature distributions and data types
- To identify missing values and sparsity
- To detect outliers and skewness
- To analyse class imbalance in the target variable
- To identify relationships between features
This phase ensured that all subsequent preprocessing and modelling decisions were data-driven.

```
Tools Used
The following tools were used for EDA and profiling:
```
- **Pandas** – Data inspection, summary statistics, value counts
- **NumPy** – Numerical analysis
- **Matplotlib & Seaborn** – Data visualization
- **Pandas Profiling (ydata-profiling)** – Automated exploratory analysis
**Dataset Overview (Initial Inspection)**
Using pandas functions such as .info(), .describe(), and .head(), the dataset was initially
examined.
Key findings:
- The dataset contained **~2.2 million records** and **151 features**
- Data types included numerical, categorical, and date-time variables
- Several features contained high proportions of missing values


**Missing Value Analysis**
EDA revealed that many features had significant missing values, especially those related to post-
loan performance.
Key observations:

- Over **60% of columns** had excessive missing values
- Post-outcome variables posed a **data leakage risk**
- These columns were removed during preprocessing


**Target Variable Distribution (Class Imbalance)**
The target variable (loan_status_clean) showed a **highly imbalanced distribution** , with non-
default loans significantly outnumbering default loans.
Key insight:

- Class imbalance could bias the model toward majority class predictions
- This justified the use of **SMOTE** during model training


**Numerical Feature Analysis**
Key numerical features such as:

- loan_amnt
- annual_inc
- dti
- int_rate

```
were analysed using histograms and box plots.
Observations:
```
- Most numerical features were **right-skewed**
- Outliers were present in income-related variables
- Scaling was preferred over outlier removal


```
Categorical Feature Analysis
Categorical variables such as:
```
- grade
- sub_grade
- home_ownership
- verification_status
were analyzed using value counts and bar charts.
Observations:
- Loan grade showed strong separation between safe and risky loans
- Home ownership and employment length influenced default behavior
- Some categories were merged during preprocessing to reduce sparsity


**Correlation Analysis**
A correlation matrix was generated for numerical variables to understand inter-feature
relationships.
Key insights:

- Strong correlation between grade and int_rate
- Moderate correlation between credit score features and loan status
- Multicollinearity was handled during feature selection


**Automated Data Profiling Report**
An automated profiling report was generated using Pandas Profiling (ydata-profiling), providing
a comprehensive summary of:

- Feature distributions
- Missing values
- Correlations
- Warnings related to skewness and duplicates

This report complemented the manual EDA and ensured no critical issues were overlooked.


**Summary of EDA & Profiling**
The EDA and data profiling phase enabled:

- Identification of high-null and leakage-prone features
- Detection of severe class imbalance
- Insightful feature selection and engineering
- A strong foundation for preprocessing and modeling
A complete interactive profiling report is provided in the appendix.


## Data Preprocessing

**Data Cleaning**
The raw Lending Club dataset contained a large number of irrelevant, highly sparse, and post-
outcome variables that could introduce data leakage. To ensure model integrity, extensive data
cleaning was performed. Columns with more than 60% missing values were removed, and all loans
without a finalized outcome were excluded from the analysis.

**Handling Missing Values**

- Numerical features were imputed using median values to reduce the impact of outliers.
- Categorical variables were imputed using mode values.
- Features with excessive missingness and low predictive value were dropped entirely.

**Target Variable Encoding**
The multi-class loan_status variable was transformed into a binary target variable to enable
supervised classification. Loans categorized as _Charged Off_ , _Default_ , and _Late (31–120 days)_ were
labelled as Risk (1), while _Fully Paid_ loans were labelled as Safe (0). All non-finalized loan states
were removed to prevent data leakage.

**Categorical Encoding**

- Nominal categorical variables were converted using One-Hot Encoding.
- Ordinal features such as loan grade were encoded to preserve their inherent order.
- Encoding was applied after data splitting to avoid information leakage.
**Feature Scaling**
Numerical features were scaled using StandardScaler to normalize feature distributions. Scaling
was applied only on the training data and then propagated to validation and test sets to maintain
consistency.


**Dimensionality Reduction using PCA**
Principal Component Analysis (PCA) was applied to the standardized numerical features to reduce
dimensionality while retaining 95% of the total variance. This step helped mitigate
multicollinearity, improve computational efficiency, and stabilize downstream models.


**Borrower Segmentation using K-Means Clustering** K-Means clustering was applied

on selected standardized financial and credit attributes to segment borrowers into homogeneous
risk profiles. The resulting Cluster_ID was used as an additional engineered feature for supervised

learning**.**

**Train–Validation–Test Split**

- Dataset split into 70% Training, 15% Validation, and 15% Testing.
- Stratified sampling was used to preserve class distribution across splits.
- The test set was held out completely until final model evaluation.

**Class Imbalance Handling**
The dataset exhibited significant class imbalance, with default cases representing a minority class.
To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied only on the
training set. This improved the model’s ability to learn patterns associated with rare default events
while avoiding data leakage.


## Model Building

To identify the most effective model for predicting loan default risk, multiple supervised machine
learning algorithms were trained and evaluated. A diverse set of linear, probabilistic, instance-
based, tree-based, and ensemble models was selected to ensure comprehensive benchmarking. All
models were trained on the SMOTE-balanced training dataset to address class imbalance.

**Logistic Regression (Baseline Model)**
Logistic Regression was used as the baseline classification model due to its simplicity,
interpretability, and strong performance in linearly separable problems. It estimates the probability
of loan default using a logistic function and provides transparent coefficient-based insights into
feature importance.

**Naïve Bayes (Probabilistic Model)**
The Naïve Bayes classifier was included as a probabilistic model that applies Bayes’ Theorem
under the assumption of feature independence. Despite its simplicity, it serves as a useful
benchmark and performs efficiently on large-scale datasets.

**K-Nearest Neighbors (Lazy Learning)**
K-Nearest Neighbors (K-NN) was implemented as a distance-based, instance-learning algorithm.
The model classifies a borrower based on the majority class of its nearest neighbors in the feature
space. Due to its sensitivity to feature scaling, standardized data was used.

**Decision Tree Classifier**
The Decision Tree classifier was used to model non-linear relationships between borrower
attributes and loan default risk. It recursively splits the data based on feature thresholds, creating
an interpretable rule-based structure.

**Random Forest (Ensemble – Bagging)**
Random Forest is an ensemble learning method that combines multiple decision trees trained on
bootstrapped samples of the data. By aggregating predictions from multiple trees, it reduces
overfitting and improves generalization performance.


**Gradient Boosting Machine (Ensemble – Boosting)**
Gradient Boosting Machine (GBM) was employed as a boosting-based ensemble technique that
builds trees sequentially, with each model correcting the errors of the previous one. GBM is known
for its high predictive accuracy in structured tabular data.


## Model Evaluation

The performance of all trained models was evaluated using multiple classification metrics to
ensure a fair and comprehensive comparison. Given the highly imbalanced nature of loan default
data, emphasis was placed on recall, precision, F1-score, ROC-AUC, and confusion matrix
analysis rather than accuracy alone.

**Evaluation Metrics**
The following metrics were used to evaluate model performance:

- Accuracy – Overall correctness of predictions.
- Precision – Proportion of correctly predicted defaulters among all predicted defaulters.
- Recall (Sensitivity) – Ability of the model to correctly identify actual loan defaulters.
- F1-Score – Harmonic mean of precision and recall.
- ROC-AUC – Measures the model’s ability to distinguish between default and non-
    default classes across different thresholds.

**Model Performance Comparison**



**Confusion Matrix Analysis**
Confusion matrices were used to analyze the classification behavior of each model in terms of true
positives, false positives, true negatives, and false negatives. This analysis is particularly critical
in loan default prediction, where false negatives (missed defaulters) can result in significant
financial losses.


**ROC Curve Analysis**
Receiver Operating Characteristic (ROC) curves were plotted to visualize the trade-off between
true positive rate and false positive rate. The area under the ROC curve (AUC) serves as a robust

indicator of model discrimination capability, independent of classification thresholds**.**

**Final Model Selection**
Based on the comparative evaluation across multiple metrics, the Logistic Regression model
demonstrated superior performance, particularly in recall and ROC-AUC. Given the business
objective of minimizing default risk, the model with the strongest ability to identify actual
defaulters was selected as the final production model.


## Final Model Selection & Justification

After evaluating multiple machine learning models using standardized performance metrics,
Logistic Regression emerged as the best-performing model for this loan default prediction task.
The selection was based not only on overall accuracy but primarily on ROC-AUC and recall,
which are critical metrics in financial risk modeling.

**Why Logistic Regression Was Selected**
The Logistic Regression model achieved the highest ROC-AUC score of 0.995, indicating
exceptional discriminatory power between default and non-default borrowers. Additionally, it
delivered strong recall performance, ensuring that the majority of actual defaulters were correctly
identified.

In comparison, ensemble models such as Random Forest showed slightly lower ROC-AUC values
(~0.97–0.98) and did not provide a meaningful improvement after hyperparameter tuning. Given
the marginal performance gap and increased complexity, Logistic Regression was determined to
be the most efficient and effective solution.

**Business & Practical Justification**
From a business perspective, Logistic Regression offers significant advantages. The model is
computationally efficient, highly stable, and inherently interpretable, which is a critical
requirement in regulated financial environments. Its linear decision boundary allows stakeholders
to clearly understand how individual features influence loan approval decisions.

Moreover, the model demonstrates robustness when combined with feature scaling and class
balancing techniques such as SMOTE, making it suitable for real-world deployment scenarios.


**Why Random Forest Was Not Selected**
Although Random Forest is a powerful ensemble technique, it did not outperform Logistic
Regression in this project. Despite hyperparameter tuning, the improvement in performance was
marginal and did not justify the added model complexity. Furthermore, Random Forest models are
less transparent, making regulatory compliance and business interpretation more challenging in
financial risk applications.


**Final Model Summary**

```
Criterion Final Model
Model Selected Logistic Regression
ROC-AUC 0.995
Recall High (Risk Focused)
Interpretability Excellent
Deployment Readiness High
```
The final model selection prioritizes business impact, model transparency, and risk mitigation over
unnecessary algorithmic complexity.


## Results & Discussion

**Model Performance Results**
The final Logistic Regression model demonstrated outstanding predictive performance on the
completely unseen testing dataset. The model achieved a ROC-AUC score of 0.995, confirming
its excellent ability to distinguish between high-risk and low-risk borrowers.

In addition to strong overall discrimination, the model achieved a high recall score, ensuring that
the majority of actual loan defaulters were correctly identified. This is particularly critical in
financial risk modeling, where failing to detect a defaulter can result in significant financial loss.

**ROC Curve Image**


**Confusion Matrix Image**

**Interpretation of Results**
The confusion matrix indicates that the model successfully minimizes false negatives, which aligns
with the primary business objective of identifying high-risk borrowers. The high recall score
reflects the model’s effectiveness in capturing actual defaults, even at the cost of a slightly higher
false positive rate, which is acceptable in conservative lending strategies.


**Feature Importance & Insights**
Feature influence analysis revealed that variables related to borrower creditworthiness and debt
burden played a dominant role in predicting loan default. Features such as debt-to-income ratio
(DTI), credit history length, interest rate, and loan grade were consistently associated with higher
default risk.

Engineered features, particularly Credit_History_Length_Years and borrower segmentation
derived from K-Means clustering, contributed meaningful additional predictive power beyond raw
input features.

**Business Impact Discussion**
From a business perspective, the final model provides a reliable decision-support system for loan
underwriting. By accurately identifying high-risk applicants, the model enables financial
institutions to reduce default rates, optimize interest pricing, and improve overall portfolio health.

The interpretability of the Logistic Regression model further allows risk managers to justify
lending decisions, ensuring compliance with regulatory and ethical standards.

**Final Closing Line**
Overall, the results demonstrate that a well-engineered, interpretable model can outperform more
complex algorithms when combined with robust preprocessing and feature design.


## Conclusion

This project successfully developed a high-performance loan default prediction system using real-
world financial data from Lending Club. Through comprehensive data preprocessing, feature
engineering, and rigorous model benchmarking, an optimal Logistic Regression model was
identified.

The final model achieved an exceptional ROC-AUC score of 0.995 and demonstrated strong recall,
ensuring accurate identification of high-risk borrowers. The integration of advanced techniques
such as PCA and borrower segmentation through K-Means clustering further enhanced model
robustness and interpretability.

Overall, this project highlights the importance of data quality, thoughtful feature engineering, and
model interpretability in building reliable financial risk assessment systems. The resulting solution
provides actionable insights that can support smarter lending decisions and reduce financial risk.


## Future Scope

While the current system demonstrates strong predictive performance, several enhancements can
be explored in future work. Deep learning models such as Artificial Neural Networks (ANNs) can
be evaluated to capture more complex, non-linear patterns in borrower behavior.

The project can be extended into a real-time loan risk assessment system by integrating live
applicant data through a web-based interface or REST API. Additionally, model performance can
be further improved by incorporating alternative data sources such as transaction-level banking
data or behavioral spending patterns.

Future work may also include advanced explainability techniques and bias detection frameworks
to ensure fair and ethical lending decisions. Deploying the model using cloud-based platforms
would allow scalability and integration into production-grade financial systems.


## References

[1] Lending Club Loan Data, Kaggle Dataset.
https://www.kaggle.com/datasets/wordsforthewise/lending-club

[2] Scikit-learn Documentation – Machine Learning in Python.
https://scikit-learn.org/stable/

[3] Pandas Documentation – Data Analysis and Manipulation Tool.
https://pandas.pydata.org/docs/

[4] imbalanced-learn Documentation – Handling Class Imbalance.
https://imbalanced-learn.org/stable/

[5] Kuhn, M. & Johnson, K. (2013). Applied Predictive Modeling.
Springer.
[6] CampusX YouTube
https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH

## Appendix

Appendix A: Automated Data Profiling Report

An extensive automated data profiling report was generated using the Pandas Profiling library.
The report includes detailed statistics on feature distributions, missing values, correlations,
interactions, and potential data quality issues.

Due to its size and interactive nature, the complete profiling report is provided as a separate HTML
file and is available in the project repository.

The profiling report can be accessed via the GitHub repository.


