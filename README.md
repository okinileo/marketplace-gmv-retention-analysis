# Brazilian E-Commerce Marketplace Analysis

End-to-end marketplace analytics project using the Olist Brazilian e-commerce dataset.

This project combines:
- exploratory data analysis,
- customer behavior analytics,
- retention analysis,
- experimentation,
- marketplace strategy,
- and predictive modeling

to investigate the main drivers of marketplace GMV growth, retention, seller concentration, and operational performance.

---

# Business Problem

Marketplace growth alone does not guarantee long-term platform sustainability.

This project investigates questions such as:

- What drives marketplace GMV growth?
- How concentrated is marketplace GMV across customers and sellers?
- How weak is customer retention?
- Which categories deserve investment versus operational improvement?
- Can promotional incentives improve customer conversion?
- Which operational variables best predict order-level GMV?

---

# Executive Summary

The analysis reveals a marketplace with strong historical GMV growth but structurally weak customer retention.

Key findings include:

- Only ~3% of customers place more than one order.
- The top customer segment contributes nearly 60% of total marketplace GMV.
- Seller concentration follows a strong Pareto dynamic.
- Delivery quality and logistics reliability directly affect customer reviews.
- Simulated discount campaigns improve conversion modestly, but generate limited short-term GMV gains.
- Product characteristics and basket composition are stronger GMV predictors than geography alone.

> **GMV note:** Throughout this project, GMV is used as a proxy for marketplace transaction volume, calculated as item price plus freight value. The dataset does not provide Olist's actual platform revenue, commissions, contribution margins, or operational costs.

---

# Dataset

Dataset: Olist Brazilian E-Commerce Public Dataset (Kaggle)

The project integrates:
- orders
- customers
- sellers
- products
- payments
- reviews
- geolocation
- freight-related variables

---

# Project Structure

```plaintext
1. Business Understanding
2. Executive Summary
3. Marketplace Growth Analysis
4. Customer Retention Analysis
5. Customer Lifetime Value (LTV)
6. Experimentation (Simulated A/B Test)
7. Marketplace Strategy Analysis
8. Predictive Modeling (GMV Prediction)
9. Strategic Recommendations
```

---

# Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

---

# Key Visualizations

## Marketplace Growth

Tracks the evolution of marketplace GMV, order volume, and average order value over time.

![Marketplace Growth](images/monthly_gmv.png)

---

## Cohort Retention Analysis

Customer retention drops sharply after the first purchase, revealing strong dependence on continuous customer acquisition.

![Cohort Retention](images/cohort_retention.png)

---

## Customer Lifetime Value Distribution

Marketplace GMV is heavily concentrated among high-value customers.

![LTV Distribution](images/ltv_distribution.png)

---

## Seller Concentration and Review Quality

A relatively small group of sellers drives most marketplace GMV.

![Seller Analysis](images/pareto_curves.png)

---

## Simulated A/B Test

Evaluates whether a discount incentive can improve customer conversion and downstream GMV performance.

![AB Test](images/ab_test_results.png)

---

## Predictive Modeling

Random Forest model predicting order-level GMV using operational and transactional variables.

![Predictive Modeling](images/predictive_modeling.png)

---

# Main Business Insights

- Marketplace GMV growth is strong, but retention remains the platform’s main structural weakness.
- High-value customers disproportionately drive marketplace GMV.
- Large operational categories require quality improvements rather than pure scaling.
- Seller concentration creates potential platform dependency risk.
- Logistics performance strongly affects customer satisfaction.
- Promotional incentives improve conversion more than short-term GMV efficiency.

---

# Predictive Modeling Results

Model:
- Random Forest Regressor

Performance:
- MAE: ~63.7 BRL
- Median Absolute Error: ~25.9 BRL
- R²: ~0.37

Most important predictors:
- product weight
- payment installments
- category
- product volume
- item count

---

# Strategic Recommendations

- Improve second-purchase activation and retention loops.
- Expand investment in high-performing categories.
- Improve operational quality in large but lower-rated categories.
- Reduce logistics friction and delivery delays.
- Monitor seller concentration risk.
- Shift optimization focus from pure GMV growth toward long-term customer value.

---

# How to Run

Clone the repository:

```bash
git clone https://github.com/okinileo/ecommerce-olist.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook
```

---

# Author

Leonardo Bernardo

Aspiring Data Scientist focused on:
- marketplace analytics
- experimentation
- predictive modeling
- business intelligence
- customer analytics
