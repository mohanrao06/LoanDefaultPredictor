# 📈 Loan Default Predictor - Advanced Risk Analysis

## Loan Default Predictor - Dataset

This project uses the **Lending Club Loan Data** from Kaggle to analyze and predict loan defaults.  
The dataset contains **accepted** and **rejected** loan applications spanning 2007–2018.

---

## 📂 Dataset Details

The dataset is divided into two main files:

1. **Accepted Loans**  
   File: `accepted_2007_to_2018Q4.csv.gz`  
   Description: Contains all loans that were approved, along with borrower information and loan details.

2. **Rejected Loans**  
   File: `rejected_2007_to_2018Q4.csv.gz`  
   Description: Contains loan applications that were rejected, including borrower and application information.

> **Note:** Each file is approximately 300MB in size. They are **not included in this repository** because GitHub does not allow files larger than 100MB.

---

## 🔗 Dataset Source

The data is available on Kaggle (login required):

[Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---


- For faster processing, convert `.csv.gz` files to **Parquet** or use **Polars/Dask** if your dataset is too large to fit in memory.

---

## 📌 Suggested For You

1. Download dataset from Kaggle.  
2. Place files in `data/raw/`.  
3. Convert to Parquet for faster loading (optional).  
4. Begin data exploration and preprocessing.



