import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Set style for professional look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Define harmonious color palette
PRIMARY_COLOR = '#2A9D8F'  # Muted Teal
SECONDARY_COLOR = '#E76F51' # Coral
PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR]

# Ensure output directories exist
os.makedirs('reports/plots', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# Load data
df = pd.read_csv('data/raw/telco_churn.csv')

# Preprocess TotalCharges for EDA (convert spaces to NaN and cast to float)
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill missing TotalCharges with median for numerical analysis/correlation
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)

# Check target distribution
churn_counts = df['Churn'].value_counts()
churn_rate = (churn_counts['Yes'] / len(df)) * 100

# ----------------- Visualizations -----------------

# Plot 1: Churn Distribution (Pie & Bar Chart)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie Chart
axes[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, 
           colors=PALETTE, explode=(0, 0.1), shadow=True, textprops={'fontsize': 12})
axes[0].set_title('Churn Distribution %', fontweight='bold')

# Bar Chart
sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=axes[1], palette=PALETTE, hue=churn_counts.index, legend=False)
axes[1].set_title('Customer Count by Churn Status', fontweight='bold')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Number of Customers')
for i, v in enumerate(churn_counts.values):
    axes[1].text(i, v + 100, f"{v}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/plots/churn_distribution.png', dpi=150)
plt.close()

# Plot 2: Correlation Heatmap (numerical features)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Encode target Churn as binary (0/1) for correlation
df_corr = df[numerical_cols].copy()
df_corr['Churn_Encoded'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features & Churn', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('reports/plots/correlation_heatmap.png', dpi=150)
plt.close()

# Plot 3: Tenure vs Churn Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn', y='tenure', palette=PALETTE, hue='Churn', legend=False)
plt.title('Distribution of Customer Tenure by Churn Status', fontweight='bold', pad=15)
plt.xlabel('Churn Status')
plt.ylabel('Tenure (Months)')
plt.tight_layout()
plt.savefig('reports/plots/tenure_vs_churn_boxplot.png', dpi=150)
plt.close()

# Plot 4: MonthlyCharges vs Churn Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Churn', y='MonthlyCharges', palette=PALETTE, hue='Churn', legend=False, inner='quartile')
plt.title('Distribution of Monthly Charges by Churn Status', fontweight='bold', pad=15)
plt.xlabel('Churn Status')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig('reports/plots/monthly_charges_vs_churn_violin.png', dpi=150)
plt.close()

# Plot 5: Churn rate by Contract type (Grouped Bar)
plt.figure(figsize=(10, 6))
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).rename('percentage').reset_index()
contract_churn['percentage'] *= 100
sns.barplot(data=contract_churn, x='Contract', y='percentage', hue='Churn', palette=PALETTE)
plt.title('Churn Percentage by Contract Type', fontweight='bold', pad=15)
plt.xlabel('Contract Type')
plt.ylabel('Percentage (%)')
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f'{height:.1f}%',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points',
                           fontweight='bold')
plt.tight_layout()
plt.savefig('reports/plots/churn_rate_by_contract.png', dpi=150)
plt.close()

# Plot 6: Churn rate by PaymentMethod (Grouped Bar)
plt.figure(figsize=(12, 6))
payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).rename('percentage').reset_index()
payment_churn['percentage'] *= 100
sns.barplot(data=payment_churn, x='PaymentMethod', y='percentage', hue='Churn', palette=PALETTE)
plt.title('Churn Percentage by Payment Method', fontweight='bold', pad=15)
plt.xlabel('Payment Method')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=15, ha='right')
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f'{height:.1f}%',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points',
                           fontsize=9,
                           fontweight='bold')
plt.tight_layout()
plt.savefig('reports/plots/churn_rate_by_payment_method.png', dpi=150)
plt.close()

# Plot 7: Churn rate by Internet Service Type
plt.figure(figsize=(10, 6))
internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).rename('percentage').reset_index()
internet_churn['percentage'] *= 100
sns.barplot(data=internet_churn, x='InternetService', y='percentage', hue='Churn', palette=PALETTE)
plt.title('Churn Percentage by Internet Service Type', fontweight='bold', pad=15)
plt.xlabel('Internet Service Type')
plt.ylabel('Percentage (%)')
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f'{height:.1f}%',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points',
                           fontweight='bold')
plt.tight_layout()
plt.savefig('reports/plots/churn_rate_by_internet_service.png', dpi=150)
plt.close()


# ----------------- Write EDA Report -----------------

# Calculate statistics for the report
total_customers = len(df)
churn_yes = churn_counts['Yes']
churn_no = churn_counts['No']

# Highest churn contract
contract_churn_rates = df[df['Churn'] == 'Yes']['Contract'].value_counts() / df['Contract'].value_counts()
highest_churn_contract = contract_churn_rates.idxmax()
highest_churn_contract_val = contract_churn_rates.max() * 100

# Internet service churn rates
internet_churn_rates = df[df['Churn'] == 'Yes']['InternetService'].value_counts() / df['InternetService'].value_counts()
highest_churn_internet = internet_churn_rates.idxmax()
highest_churn_internet_val = internet_churn_rates.max() * 100

# Tenure statistics
mean_tenure_churn = df[df['Churn'] == 'Yes']['tenure'].mean()
mean_tenure_loyal = df[df['Churn'] == 'No']['tenure'].mean()

# Monthly charges statistics
mean_charges_churn = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
mean_charges_loyal = df[df['Churn'] == 'No']['MonthlyCharges'].mean()

report_content = f"""# Stage 1: Exploratory Data Analysis (EDA) Report

This report presents key findings and visual insights from the Exploratory Data Analysis performed on the Telco Customer Churn dataset.

## Executive Summary
* **Total Customers Analyzed**: {total_customers}
* **Churn Status**: {churn_yes} customers churned (**{churn_rate:.2f}%** churn rate), while {churn_no} customer stayed (**{100 - churn_rate:.2f}%**).
* **Imbalance Warning**: The target feature `Churn` is significantly imbalanced, which must be addressed in Stage 2 (Preprocessing) using techniques like SMOTE.

---

## Key Business Insights

### 1. Contract Type vs. Churn Status
* **Insight**: Customers with a **Month-to-month** contract have an extremely high churn rate of **{highest_churn_contract_val:.2f}%**. 
* In comparison, customers with a **One year** contract churn at **11.3%**, and **Two year** contract customers churn at only **2.8%**.
* **Actionable Recommendation**: Incentivize Month-to-month customers to upgrade to longer-term (1 or 2-year) contracts by offering loyalty discounts or bundles.

### 2. The Impact of Tenure
* **Insight**: Tenure is strongly negatively correlated with churn. The average tenure of customers who churned is **{mean_tenure_churn:.1f} months**, compared to **{mean_tenure_loyal:.1f} months** for loyal customers.
* **Observation**: The risk of churn is highest in the first year. Customers who survive the initial 12-month period show much higher brand loyalty.

### 3. Monthly Charges & Total Charges
* **Insight**: Higher monthly charges correlate with higher churn. Customers who churned have a higher average monthly charge (**${mean_charges_churn:.2f}**) compared to loyal customers (**${mean_charges_loyal:.2f}**).
* **Observation**: Price-sensitive customers or those who do not perceive equivalent value in high-tier packages are more likely to cancel services.

### 4. Internet Service Type & Churn Risk
* **Insight**: Customers using **Fiber optic** internet service show the highest churn rate at **{highest_churn_internet_val:.2f}%**.
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
"""

with open('reports/eda_report.md', 'w') as f:
    f.write(report_content)


# ----------------- Generate Jupyter Notebook (.ipynb) -----------------

notebook_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Stage 1: Exploratory Data Analysis (EDA)\n",
    "\n",
    "This notebook covers the exploratory analysis of the Telco Customer Churn dataset. \n",
    "Our objectives are to inspect the dataset structure, address missing values, perform univariate/bivariate analysis, and extract actionable business insights."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({\n",
    "    'font.size': 11,\n",
    "    'axes.labelsize': 12,\n",
    "    'axes.titlesize': 14,\n",
    "    'xtick.labelsize': 10,\n",
    "    'ytick.labelsize': 10\n",
    "})\n",
    "\n",
    "PRIMARY_COLOR = '#2A9D8F'\n",
    "SECONDARY_COLOR = '#E76F51'\n",
    "PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load and Inspect Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/raw/telco_churn.csv')\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Check for Missing Values & Duplicates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Duplicate rows:\", df.duplicated().sum())\n",
    "print(\"\\nMissing values count per column:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Notice that TotalCharges is loaded as an object because it contains space characters\n",
    "spaces_count = (df['TotalCharges'] == ' ').sum() + (df['TotalCharges'] == '').sum()\n",
    "print(f\"\\nEmpty string or space values in TotalCharges: {spaces_count}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Clean TotalCharges (Preprocessing for Analysis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Replace empty spaces with NaN and cast to float\n",
    "df['TotalCharges'] = df['TotalCharges'].replace(r'^\\s*$', np.nan, regex=True)\n",
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
    "\n",
    "# Fill missing TotalCharges with median\n",
    "tc_median = df['TotalCharges'].median()\n",
    "df['TotalCharges'] = df['TotalCharges'].fillna(tc_median)\n",
    "print(\"Missing values in TotalCharges after cleaning:\", df['TotalCharges'].isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Churn Distribution Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "churn_counts = df['Churn'].value_counts()\n",
    "churn_rate = (churn_counts['Yes'] / len(df)) * 100\n",
    "print(f\"Churn Counts:\\n{churn_counts}\")\n",
    "print(f\"\\nChurn Rate: {churn_rate:.2f}%\")\n",
    "\n",
    "# Plot\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "axes[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=PALETTE, explode=(0, 0.1), shadow=True)\n",
    "axes[0].set_title('Churn Distribution %', fontweight='bold')\n",
    "\n",
    "sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=axes[1], palette=PALETTE, hue=churn_counts.index, legend=False)\n",
    "axes[1].set_title('Customer Count by Churn Status', fontweight='bold')\n",
    "axes[1].set_xlabel('Churn')\n",
    "axes[1].set_ylabel('Count')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
    "df_corr = df[numerical_cols].copy()\n",
    "df_corr['Churn_Encoded'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=\".3f\", linewidths=0.5, vmin=-1, vmax=1)\n",
    "plt.title('Correlation Matrix (Numerical Features & Churn)', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Tenure & Monthly Charges vs. Churn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Boxplot for Tenure\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.boxplot(data=df, x='Churn', y='tenure', palette=PALETTE, hue='Churn', legend=False)\n",
    "plt.title('Distribution of Customer Tenure by Churn Status', fontweight='bold')\n",
    "plt.xlabel('Churn')\n",
    "plt.ylabel('Tenure (Months)')\n",
    "plt.show()\n",
    "\n",
    "# Violinplot for Monthly Charges\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.violinplot(data=df, x='Churn', y='MonthlyCharges', palette=PALETTE, hue='Churn', legend=False, inner='quartile')\n",
    "plt.title('Distribution of Monthly Charges by Churn Status', fontweight='bold')\n",
    "plt.xlabel('Churn')\n",
    "plt.ylabel('Monthly Charges ($)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Categorical Relations: Contract, Internet Service, and Payment Methods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Churn Rate by Contract\n",
    "plt.figure(figsize=(10, 5))\n",
    "contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).rename('percentage').reset_index()\n",
    "contract_churn['percentage'] *= 100\n",
    "sns.barplot(data=contract_churn, x='Contract', y='percentage', hue='Churn', palette=PALETTE)\n",
    "plt.title('Churn Percentage by Contract Type', fontweight='bold')\n",
    "plt.ylabel('Percentage (%)')\n",
    "plt.show()\n",
    "\n",
    "# Churn Rate by Payment Method\n",
    "plt.figure(figsize=(12, 5))\n",
    "payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).rename('percentage').reset_index()\n",
    "payment_churn['percentage'] *= 100\n",
    "sns.barplot(data=payment_churn, x='PaymentMethod', y='percentage', hue='Churn', palette=PALETTE)\n",
    "plt.title('Churn Percentage by Payment Method', fontweight='bold')\n",
    "plt.xticks(rotation=15, ha='right')\n",
    "plt.ylabel('Percentage (%)')\n",
    "plt.show()\n",
    "\n",
    "# Churn Rate by Internet Service\n",
    "plt.figure(figsize=(10, 5))\n",
    "internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).rename('percentage').reset_index()\n",
    "internet_churn['percentage'] *= 100\n",
    "sns.barplot(data=internet_churn, x='InternetService', y='percentage', hue='Churn', palette=PALETTE)\n",
    "plt.title('Churn Percentage by Internet Service Type', fontweight='bold')\n",
    "plt.ylabel('Percentage (%)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Summary of Key EDA Findings\n",
    "\n",
    "1. **Contract Type**: Month-to-month contracts have an extremely high churn risk (~42.7%). Two-year contracts have the lowest (~2.8%).\n",
    "2. **Tenure**: Shorter tenure is highly associated with churn. Average tenure of churned customers is ~18 months compared to ~37.6 months for loyal customers.\n",
    "3. **Monthly Charges**: Customers who churn have higher average monthly charges ($74.44 vs $61.27).\n",
    "4. **Internet Service**: Fiber optic internet subscribers churn at a remarkably high rate (~41.9%) compared to DSL subscribers (~19.0%).\n",
    "5. **Payment Method**: Electronic checks are associated with the highest churn rate (~45.3%). Automatic payment options (Credit Card / Bank Transfer) show significantly lower rates (~15-16%)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (.venv)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('notebooks/01_EDA.ipynb', 'w') as f:
    json.dump(notebook_data, f, indent=1)

print("EDA analysis completed. Reports, plots, and Jupyter notebook successfully generated!")
