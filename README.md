# Customer Churn Analysis (Banking Dataset)

## Overview
This project explores customer churn in a banking dataset to identify the main churn drivers, discover high-risk customer segments, and propose practical retention actions.

The analysis is intentionally positioned as a **Data Analyst Project**, with a focus on:
- exploratory data analysis,
- churn segmentation,
- multivariate insight discovery,
- and business recommendations.

This notebook is centered on **analysis and interpretation**, not on predictive modeling.

---

## Business question
Which customer segments are more likely to churn, and what patterns can help explain churn behavior?

**Target variable**
- `Exited = 1` → customer churned
- `Exited = 0` → customer retained

---

## Dataset
The dataset contains **10,000 customers** and includes variables related to:
- demographics,
- geography,
- account status,
- engagement,
- and product usage.

The dataset is relatively clean:
- no missing values,
- no duplicate rows,
- no duplicate customer IDs.

---

## What the notebook covers
The analysis follows this structure:

1. dataset review and quality checks  
2. feature engineering for segmentation  
3. baseline churn analysis  
4. exploratory and bivariate analysis  
5. multivariate segment discovery  
6. final priority segments  
7. business recommendations  

---

## Main findings
The strongest churn signals were:

- **Age** → churn increases sharply from **41+**
- **Geography** → **Germany** is the highest-risk market
- **IsActiveMember** → inactive customers churn substantially more
- **NumProductsGroup** → customers with **1 product** are more fragile than customers with 2
- **HasBalance** → customers with positive balance show higher churn than customers with zero balance

---

## Final priority segments
The main high-risk segments identified in the analysis are:

- **Inactive customers with 1 product**
- **Inactive customers in Germany**
- **Inactive customers aged 41–50**

Secondary red flags:
- **Inactive customers aged 51–60**
- **Customers with 3+ products**

These secondary groups show very high churn, but they are smaller and should be interpreted more cautiously.

---

## Business recommendations
Based on the analysis, the main recommended actions are:

- re-engagement campaigns for inactive single-product customers,
- targeted retention actions for inactive customers aged 41–50,
- localized retention analysis for inactive customers in Germany.

---

## Tools used
- Python
- pandas
- numpy
- matplotlib
- seaborn

---

## Notes
This project focuses on **EDA, segmentation, and business interpretation**.  
A simple interpretable churn model can be added later as a separate follow-up step.

---

## Author
**Lorenzo Santarelli**  
Aspiring Data Analyst with a technical background in software and SQL.