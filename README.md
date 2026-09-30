# Bank Customer Churn Agent

An end-to-end machine learning project that predicts **bank customer churn risk** and connects the prediction to an **agent-style retention workflow** through an interactive Gradio interface.

The project uses a **Random Forest classifier** trained on the Kaggle Bank Customer Churn dataset. Customer information is preprocessed, scaled, and used to estimate the probability that a customer will leave the bank. Based on the predicted probability, the agent assigns a risk status and recommends a corresponding retention action.

## Project Overview

The project follows this workflow:

```text
Bank Customer Churn Dataset
            ↓
     Data Preparation
            ↓
   Categorical Encoding
            ↓
       Feature Scaling
            ↓
     Train/Test Split
            ↓
   Random Forest Classifier
            ↓
     Churn Probability
            ↓
      Agent Decision
            ↓
      Retention Action
            ↓
      Gradio Web Interface
```

## Objectives

* Predict whether a bank customer is likely to churn.
* Calculate a customer's churn probability.
* Classify customers into high-risk and low-risk groups.
* Demonstrate how a machine learning prediction can be connected to an agent-style decision workflow.
* Provide an interactive interface for testing customer profiles.

## Dataset

The project uses the **Bank Customer Churn** dataset from Kaggle.

The dataset contains customer information such as:

* Credit Score
* Age
* Tenure
* Account Balance
* Number of Products
* Credit Card ownership
* Active membership status
* Estimated Salary
* Geography
* Gender
* Card Type
* Registered complaints
* Satisfaction Score
* Points Earned
* Churn indicator (`Exited`)

The columns `RowNumber`, `CustomerId`, and `Surname` are removed before model training because they are not used as predictive features.

##  Data Preprocessing

The notebook performs the following preprocessing steps:

1. Loads the dataset using `kagglehub`.
2. Removes identification columns.
3. Converts categorical variables into numerical dummy variables using `pandas.get_dummies()`.
4. Separates the target variable (`Exited`) from the input features.
5. Standardizes the feature values using `StandardScaler`.
6. Splits the data into:

   * 80% training data
   * 20% testing data
7. Uses stratification during the train/test split to preserve the target-class distribution.

## Machine Learning Model

A **Random Forest Classifier** is used for churn prediction.

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
```

The model uses class weighting to account for the imbalance between customers who churn and customers who remain.

### Evaluation

The notebook evaluates the model using:

* Classification Report
* ROC-AUC Score

The model also produces a probability of churn using:

```python
model.predict_proba(X_test)[:, 1]
```

This probability is then used by the agent decision logic.

## Saved Model

After training, the project saves both the trained model and preprocessing scaler:

```text
churn_model.pkl
scaler.pkl
```

These files can subsequently be loaded to make predictions without retraining the model.

## Agent Logic

The project demonstrates an agent-style workflow on top of the machine learning model.

The churn probability is compared against a **0.5 threshold**:

```text
Churn probability > 50%
        ↓
    HIGH RISK
        ↓
Retention action

Churn probability ≤ 50%
        ↓
    LOW RISK
        ↓
Standard customer communication
```

For a high-risk customer, the implemented demo action is:

* Draft a personalized retention offer email
* Apply a 20% fee waiver for 6 months
* Escalate the customer to a Senior Relationship Manager

For a low-risk customer:

* Mark the customer as stable
* Enroll the customer in standard monthly updates

These actions are part of the project demonstration and are not connected to an actual banking system.

## Interactive Gradio Application

The notebook includes a Gradio interface titled:

**🏦 Bank Customer Churn Agent**

Users can enter:

| Input              | Description                                 |
| ------------------ | ------------------------------------------- |
| Credit Score       | Customer credit score                       |
| Age                | Customer age                                |
| Tenure             | Years with the bank                         |
| Account Balance    | Current account balance                     |
| Number of Products | Number of banking products                  |
| Credit Card        | Whether the customer has a credit card      |
| Active Member      | Whether the customer is an active member    |
| Estimated Salary   | Estimated customer salary                   |
| Geography          | France, Germany, or Spain                   |
| Gender             | Female or Male                              |
| Card Type          | Diamond, Gold, Silver, or Bronze            |
| Complaint          | Whether the customer registered a complaint |
| Satisfaction Score | Customer satisfaction score                 |
| Points Earned      | Customer reward points                      |

The application returns:

1. **Risk Assessment**
2. **Agent Decision & Action**

##  Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* KaggleHub
* Gradio

## Project Structure

A typical project structure can be:

```text
.
├── Agent.ipynb
├── churn_model.pkl
├── scaler.pkl
└── README.md
```

## Running the Project

The notebook was developed with a Google Colab-style workflow.

Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib kagglehub gradio
```

Then open:

```text
Agent.ipynb
```

Run the notebook cells in order.

The final cell launches the Gradio application:

```python
demo.launch(share=True, debug=True)
```

This generates a temporary public Gradio URL when executed in an environment that supports Gradio sharing.

Or use link https://colab.research.google.com/drive/1_JauqvrBHttdVD1FF0vB0ji2IFVwCPZj?authuser=1

## Notes

This project is a machine learning and agent-workflow demonstration.

The retention actions shown in the interface are simulated actions. The notebook does not actually send emails, modify customer accounts, apply fee waivers, or communicate with relationship managers.

The notebook also contains a demonstration `churn_agent_decision()` function whose current implementation initializes the agent rather than performing a complete raw-input prediction pipeline. The complete prediction flow is implemented in the Gradio `predict_churn()` function.

## Possible Extensions

Potential improvements include:

* Building a single reusable preprocessing pipeline.
* Connecting the agent directly to the trained model for arbitrary customer dictionaries.
* Adding feature importance explanations.
* Adding model performance visualizations.
* Persisting the complete preprocessing pipeline alongside the model.
* Connecting retention actions to real APIs or business systems.
* Adding customer-level explanations for why a profile was classified as high risk.
* Deploying the Gradio application as a permanent web application.
