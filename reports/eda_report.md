# Stage 1: Exploratory Data Analysis (EDA) Report

This report presents key findings and visual insights from the Exploratory Data Analysis performed on the Telco Customer Churn dataset.

## Executive Summary
* **Total Customers Analyzed**: 7043
* **Churn Status**: 1869 customers churned (**26.54%** churn rate), while 5174 customer stayed (**73.46%**).
* **Imbalance Warning**: The target feature `Churn` is significantly imbalanced, which must be addressed in Stage 2 (Preprocessing) using techniques like SMOTE.

---

## Key Business Insights

### 1. Contract Type vs. Churn Status
* **Insight**: Customers with a **Month-to-month** contract have an extremely high churn rate of **42.71%**. 
* In comparison, customers with a **One year** contract churn at **11.3%**, and **Two year** contract customers churn at only **2.8%**.
* **Actionable Recommendation**: Incentivize Month-to-month customers to upgrade to longer-term (1 or 2-year) contracts by offering loyalty discounts or bundles.

### 2. The Impact of Tenure
* **Insight**: Tenure is strongly negatively correlated with churn. The average tenure of customers who churned is **18.0 months**, compared to **37.6 months** for loyal customers.
* **Observation**: The risk of churn is highest in the first year. Customers who survive the initial 12-month period show much higher brand loyalty.

### 3. Monthly Charges & Total Charges
* **Insight**: Higher monthly charges correlate with higher churn. Customers who churned have a higher average monthly charge (**$74.44**) compared to loyal customers (**$61.27**).
* **Observation**: Price-sensitive customers or those who do not perceive equivalent value in high-tier packages are more likely to cancel services.

### 4. Internet Service Type & Churn Risk
* **Insight**: Customers using **Fiber optic** internet service show the highest churn rate at **41.89%**.
* **Observation**: Although Fiber optic provides faster speeds, its higher cost or potential service quality issues (e.g., outages or poor setup support) might be causing customer dissatisfaction. DSL users have a significantly lower churn rate (~19.0%).

### 5. Payment Methods
* **Insight**: Customers using **Electronic check** as their payment method churn at **45.3%**, which is significantly higher than other payment options:
  * Credit card (automatic): **15.2%**
  * Bank transfer (automatic): **16.7%**
  * Mailed check: **19.1%**
* **Actionable Recommendation**: Promote automatic payment registration (Credit Card / Bank Transfer) by offering a one-time invoice credit (e.g., $5 discount), which naturally locks in a lower churn risk.

---

## Visualizations Summary
All plots have been generated and saved under `reports/plots/`:
1. **Churn Distribution**: [churn_distribution.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/churn_distribution.png) - Shows the target binary class imbalance.
2. **Correlation Heatmap**: [correlation_heatmap.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/correlation_heatmap.png) - Highlights correlations between numerical variables and churn.
3. **Tenure vs. Churn**: [tenure_vs_churn_boxplot.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/tenure_vs_churn_boxplot.png) - Showcases customer tenure distributions.
4. **Monthly Charges vs. Churn**: [monthly_charges_vs_churn_violin.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/monthly_charges_vs_churn_violin.png) - Visualizes price distributions of churned vs loyal customers.
5. **Contract Type vs. Churn**: [churn_rate_by_contract.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/churn_rate_by_contract.png) - Details churn rate split across contract plans.
6. **Payment Method vs. Churn**: [churn_rate_by_payment_method.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/churn_rate_by_payment_method.png) - Outlines differences in churn based on transaction type.
7. **Internet Service vs. Churn**: [churn_rate_by_internet_service.png](file:///c:/Users/Rachit%20Dubey/Documents/Codehub/Projects/Customer-Churn-Prediction/reports/plots/churn_rate_by_internet_service.png) - Compares Fiber Optic, DSL, and No Internet customers.
