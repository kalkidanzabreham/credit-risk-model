# Credit Scoring Project: Basel II Compliance

## Business Understanding

### Basel II and the Importance of Risk Measurement and Interpretability

The **Basel II Capital Accord** emphasizes the need for robust risk measurement, validation, and transparency in credit decision-making. Financial institutions are required to ensure that their risk estimates are explainable, reproducible, and auditable. In the context of credit scoring, this means that the models used cannot be opaque "black boxes" — they must be interpretable and well-documented to meet the needs of regulators, internal risk committees, and business stakeholders. 

This project is designed with these regulatory requirements in mind. The goal is to develop a credit scoring model that meets these standards by focusing on the following:

- **Explainability:** The model's outcomes must be explainable to business stakeholders and regulators.
- **Transparency:** The assumptions, data sources, and methodologies must be fully documented.
- **Reproducibility:** The model’s development process must be reproducible to ensure consistency and compliance.

### Need for a Proxy Default Variable

The provided **eCommerce dataset** does not include an explicit label indicating whether a customer has defaulted on a loan. Since supervised learning requires a target variable, a **proxy for credit risk** is created based on customer behavioral signals. 

We approximate **default risk** using **Recency, Frequency, and Monetary (RFM)** metrics derived from transaction history. Customers showing low engagement (infrequent transactions, lower spending, or long periods of inactivity) are considered higher credit risks. This proxy introduces potential risks such as misclassification, bias, and imperfect alignment with true defaults. These risks are mitigated through:

- **Conservative interpretation** of the proxy metric.
- **Validation** of results to check model accuracy.
- **Transparent reporting** of assumptions and limitations.

### Trade-offs Between Simple and Complex Models in a Regulated Context

Credit scoring in regulated environments requires a balance between **model interpretability** and **predictive performance**. Simple models (e.g., **Logistic Regression** with **Weight of Evidence**) offer transparency, stability, and ease of regulatory approval. However, they may lack the ability to capture complex patterns in the data.

On the other hand, complex models (e.g., **Gradient Boosting Machines**) can achieve higher accuracy but may suffer from reduced explainability. In practice, institutions often **benchmark complex models against simpler, interpretable models** and apply explainability techniques where higher performance is justified.

This project follows this philosophy, aiming for compliance-ready outcomes that balance interpretability with performance.

---

## Task 2 — Exploratory Data Analysis (EDA)

The **Exploratory Data Analysis (EDA)** is performed in the **notebook**: `notebooks/eda.ipynb`. This notebook serves as an exploratory analysis tool, focusing on hypothesis generation and feature insights. It **does not** contain production-ready code.

### EDA Scope and Methods

The EDA covers the following key areas:

1. **Dataset Structure and Data Types**
   - Investigating the overall structure of the dataset and identifying data types (numerical, categorical, etc.).

2. **Summary Statistics of Numerical Features**
   - Reviewing basic statistics (mean, median, standard deviation, etc.) of numerical variables.

3. **Distributions of Numerical Variables**
   - Analyzing distributions for variables like **Transaction Amount** and **Transaction Value** to identify skewness and outliers.

4. **Distributions of Categorical Variables**
   - Examining categorical variables such as **Product Category** and **Channel ID** for skewness or imbalances.

5. **Correlation Analysis Among Numerical Features**
   - Investigating the relationships between numerical features using correlation metrics and heatmaps.

6. **Missing Value Identification**
   - Identifying missing values to inform the choice of imputation strategies in later stages.

7. **Outlier Detection Using Box Plots**
   - Visualizing potential outliers, particularly in transaction amounts, to distinguish legitimate high-value transactions from data errors.

### Key EDA Insights

The following insights have been identified during the EDA phase, which will guide feature engineering and modeling decisions:

- **Highly Skewed Transaction Values**: Both **Transaction Amounts** and **Transaction Values** exhibit right-skewed distributions with long tails. This suggests the need for scaling or transformations in feature engineering.

- **Customer Activity Imbalance**: A small subset of customers is responsible for a disproportionately large share of transactions. This indicates highly heterogeneous customer behavior, which may require segmentation or special handling in the modeling phase.

- **Dominant Categories**: Certain **Product Categories** and **Channels** dominate transaction volumes. These features could play a critical role in predicting credit risk and will be carefully considered during feature selection.

- **Presence of Outliers**: High-value transactions are present as outliers in the dataset. These appear to represent legitimate purchases, not errors, and will be handled appropriately during data preprocessing.

- **Limited Missingness**: The dataset contains some missing values, but they are limited. This informs the selection of imputation strategies to fill in missing data without losing too much information.

---
## Project Structure

The repository follows a modular, production-oriented structure:

- `notebooks/`: Exploratory analysis (EDA only)
- `src/`: Reusable, production-ready Python modules for data processing and modeling
- `tests/`: Unit tests for validating core data processing logic
- `data/`: Raw and processed data (excluded from version control)

  Missing values will be handled using median imputation for skewed numerical features such as Amount and Value, as median is robust to outliers. Categorical features such as ProductCategory and ChannelId will be imputed using the mode to preserve the most frequent behavior patterns. These choices balance robustness with simplicity and are appropriate for transactional data.

]

## ⚙️ Task 3 — Feature Engineering & Data Processing

### Objective

Transform raw transaction-level data into a model-ready, reproducible dataset using a fully automated preprocessing pipeline compliant with production and regulatory standards.

### Implementation Overview

Feature engineering is implemented as reusable, production-ready code in:

`src/data_processing.py`

The transformation logic is built using `sklearn.pipeline.Pipeline` and `ColumnTransformer` to ensure:

* **Reproducibility**
* **Clear separation** of numerical and categorical transformations
* **Compatibility** with model training and deployment 

### Feature Engineering Steps

1.  **Temporal Feature Extraction**
    * From `TransactionStartTime`, the following features are derived: `transaction_hour`, `transaction_day`, `transaction_month`, `transaction_year`.
    * These features capture customer behavioral patterns across time.

2.  **Aggregate Customer-Level Features**
    * For each `CustomerId`, the following aggregates are computed: **Total Transaction Amount**, **Average Transaction Amount**, **Transaction Count**, **Standard Deviation of Transaction Amounts**.
    * These features capture spending intensity, frequency, and variability.

3.  **Missing Value Handling**
    * Explicit imputation strategies are applied:
        * *Numerical features* → Median imputation (robust to skewness and outliers)
        * *Categorical features* → Mode imputation (preserves dominant behavioral patterns)

4.  **Encoding & Scaling**
    * Categorical variables are encoded using **One-Hot Encoding**
    * Numerical variables are standardized using **StandardScaler**

### Output

The processing function returns:

* A processed dataframe
* A fitted preprocessing pipeline ready for model training and inference

---

## 🎯 Task 4 — Proxy Target Variable Engineering

### Objective

Create a proxy credit risk target variable (`is_high_risk`) in the absence of an explicit default label.

### Methodology

1.  **RFM Metric Calculation**
    * For each customer:
        * **Recency:** Days since last transaction (based on a snapshot date)
        * **Frequency:** Number of transactions
        * **Monetary:** Total transaction amount
    * 

2.  **Customer Segmentation via Clustering**
    * **K-Means clustering** (k = 3) is applied to scaled RFM features
    * `random_state` is fixed to ensure reproducibility

3.  **High-Risk Label Assignment**
    * The cluster characterized by **low frequency** and **low monetary value** is identified as the least engaged (highest-risk) segment
    * Customers in this cluster are labeled: `is_high_risk = 1`
    * All others = 0

4.  **Integration**
    * The target variable is merged back into the processed dataset for supervised learning.

Implementation is located in:

`src/target_engineering.py`

---

## 🔬 Task 5 — Model Training, Evaluation & Experiment Tracking

### Objective

Develop a structured, auditable model training workflow with experiment tracking and evaluation.

### Setup

The following tools are used:

* **MLflow** for experiment tracking and model registry 
* **pytest** for unit testing
* **scikit-learn** for modeling

### Models Trained

At least two models are trained and compared:

* **Logistic Regression** (baseline, interpretable)
* **Tree-based models** (Decision Tree / Random Forest / Gradient Boosting)

### Hyperparameter Tuning

**Grid Search** or **Random Search** is used to optimize model parameters

* Best models are selected based on validation performance

### Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* **ROC-AUC** 

### Experiment Tracking

For each experiment, MLflow logs:

* Model parameters
* Evaluation metrics
* Model artifacts

The best-performing model is registered in the **MLflow Model Registry**.

---

## 🚀 Task 6 — Model Deployment & CI/CD

### Objective

Deploy the trained credit risk model as a containerized REST API with automated testing and quality checks.

### FastAPI Service

The API is implemented in:

`src/api/main.py`

Key Features

* Loads the best model from MLflow
* Exposes a `/predict` endpoint
* Accepts structured customer transaction data
* Returns a credit risk probability
* Request and response validation is handled using **Pydantic** models.

### Dockerization

#### `Dockerfile`

A `Dockerfile` at the repository root builds a lightweight production image:

* Python 3.10 slim base
* Installs dependencies
* Runs the FastAPI app using Uvicorn

#### `Docker Compose`

`docker-compose.yml` enables local deployment with:

* Port mapping (`8000:8000`)
* MLflow tracking URI configuration

The service can be started locally using:

```bash
docker-compose build
docker-compose up
```

Swagger UI is available at:
```bash

http://localhost:8000/docs
```
CI/CD with GitHub Actions
A CI pipeline is defined in:
```bash

.github/workflows/ci.yml
```
Automated Checks on Every Push
   - Code linting (flake8 / black)
   - Unit tests using pytest

The build fails automatically if:
   - Code style checks fail
   - Unit tests fail

This ensures consistent code quality and prevents regressions.


###  Summary
This project follows an end-to-end, production-grade credit risk modeling pipeline aligned with Basel II principles:

   - Business understanding & regulatory context
   - Transparent EDA
   - Reproducible feature engineering
   - Proxy target creation
   - Tracked model training & evaluation
   - Containerized deployment with CI/CD


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We would like to acknowledge the authors of the original eCommerce dataset, as well as various resources used in the creation of this project, including articles, books, and open-source libraries.

