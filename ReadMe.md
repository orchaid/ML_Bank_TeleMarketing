
# Bank Marketing Campaign - Term Deposit Prediction

This project aims to predict whether a customer will subscribe to a term deposit product based on a marketing campaign conducted via phone calls by a Portuguese bank.

🔗 **Dataset**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)


## Goal

* **Business Goal**: Help the bank optimize its marketing efforts by identifying potential clients likely to subscribe to a term deposit.
* **ML Objective**: Build a binary classification model to predict if a customer will say “yes” or “no” to a term deposit subscription.



## Process & Methodology

### 1. Exploratory Data Analysis (EDA)

* Dataset contains **45,211** records and **17 features**.
* Target variable (`term_deposit`) is **highly imbalanced** (yes: \~12%, no: \~88%).
* Removed `contact_duration` to **prevent data leakage** since this value isn’t known before making the call.

### 2. Data Preprocessing

* **Mapping (Binary)**: `default`, `housing`, `loan`, and `term_deposit` → mapped to 0/1.
* **Label Encoding**: `education` (ordinal).
* **One-Hot Encoding**: `job`, `marital`, `contact`, `past_camp_outcome` (nominal).
* **Cyclical Encoding**: `month` to capture seasonality (since subscription likelihood varies across months).


### 3. Handling Class Imbalance

* Used **Random OverSampling** to balance the classes for fairer training.

### 4. Feature Scaling

* Scaled skewed numerical features using **StandardScaler**:

  * `balance`, `campaign`, `count_prev_contact`, `day_of_week`
* Skipped log transformation for `balance` due to presence of negative values and to avoid introducing NaNs or infs.


## Models & Evaluation

Trained several classifiers to compare performance:

| Model               | Accuracy | Precision (yes) | Recall (yes) | F1-score (yes) |
| ------------------- | -------- | --------------- | ------------ | -------------- |
| **Random Forest**   | 89%      | 0.54            | 0.26         | 0.35           |
| XGBoost             | 85%      | 0.37            | 0.44         | 0.40           |
| SVM                 | 83%      | 0.30            | 0.35         | 0.32           |
| Decision Tree       | 83%      | 0.28            | 0.27         | 0.27           |
| K-Nearest Neighbors | 78%      | 0.24            | 0.38         | 0.29           |
| Logistic Regression | 70%      | 0.22            | 0.66         | 0.33           |

* **Best Performing Model**:  **Random Forest**

  * Accuracy: 89%
  * Balanced performance across both classes but still limited recall for the minority class.


## Key Highlights

* Addressed **data leakage** and **class imbalance** carefully.
* Used **appropriate encoding techniques** based on feature types and model compatibility.
* Applied **cyclical encoding** for seasonal patterns in `month`.
* Evaluated **six machine learning models**, balancing trade-offs between recall and precision.


## Lessons Learned

* Skewed features (e.g., `balance`) significantly impact non-tree-based models when not transformed.
* Class imbalance requires careful resampling to prevent biased models.
* Simple preprocessing choices (like dropping a column or applying correct encoding) can drastically affect model integrity.


