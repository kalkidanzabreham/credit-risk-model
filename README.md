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


## Next Steps

### Feature Engineering

Based on the insights from EDA, the next steps involve:

- **Transforming skewed features** (e.g., applying log-transformation to transaction values).
- **Handling imbalanced customer behavior** (e.g., using techniques like SMOTE for balancing customer engagement).
- **Creating new features** (e.g., RFM metrics to represent customer behavior).



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We would like to acknowledge the authors of the original eCommerce dataset, as well as various resources used in the creation of this project, including articles, books, and open-source libraries.

