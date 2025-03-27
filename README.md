# Food Insecurity Analysis and Causal Inference Study

This document summarizes the steps taken in our analysis of food insecurity, low access, and related socioeconomic factors. The work includes data preprocessing, detailed data visualizations, and a comprehensive causal inference analysis using methods such as Propensity Score Matching (PSM), Inverse Probability of Treatment Weighting (IPTW), and Doubly Robust Estimation.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Causal Inference Analysis](#causal-inference-analysis)
  - [Propensity Score Matching (PSM)](#propensity-score-matching-psm)
  - [Inverse Probability of Treatment Weighting (IPTW)](#inverse-probability-of-treatment-weighting-iptw)
  - [Doubly Robust Estimation](#doubly-robust-estimation)
- [Tableau Dashboard](#tableau-dashboard)
- [Conclusion](#conclusion)

## Introduction
Food insecurity is a complex issue influenced by multiple socioeconomic and demographic factors. This study uses a rich dataset—from Excel and CSV sources—to explore the relationships between food insecurity and variables such as SNAP participation, limited food access, and various demographic measures.

## Data Preprocessing
- **Data Loading:** Data was imported from multiple Excel sheets and a CSV file.
- **Cleaning & Mapping:** The state names were cleaned and mapped using a global dictionary to ensure consistency.
- **Merging Datasets:** Different sheets (e.g., supplemental county data, food access, and socioeconomic variables) were merged on common keys (State, County) to create a comprehensive dataset.
- **Output:** The final merged dataset was saved for further processing and analysis.

## Data Visualization
A variety of visualizations were generated to better understand the data:
- **Histograms & Density Plots:** To show the distributions of population, food insecurity rates, and other key indicators.
- **Scatter Plots:** To explore the relationship between population metrics and SNAP participation.
- **Correlation Heatmaps:** To examine the interrelationships among multiple demographic and food access variables.
- **Box & Bar Charts:** For comparing urban versus rural indicators and binary variables related to food access.

## Causal Inference Analysis
The analysis was extended to estimate the causal effects of SNAP participation (or related interventions) on food insecurity using several methods:

### Propensity Score Matching (PSM)
- **Estimation:** Propensity scores were computed using logistic regression.
- **Matching Process:** Matching was performed to balance the treatment (high SNAP) and control groups.
- **Evaluation:** Pre- and post-matching balance was assessed through summary statistics and t-tests.

### Inverse Probability of Treatment Weighting (IPTW)
- **Weight Calculation:** Stabilized IPTW weights were derived from the propensity scores.
- **Weighted Regression:** A weighted least squares regression was used to estimate the average treatment effect.
- **Robustness:** Variance checks and adjustments (e.g., clipping extreme propensity scores) ensured stability in the estimates.

### Doubly Robust Estimation
- **Method:** Combined outcome modeling (using Random Forests) and propensity score weighting.
- **Implementation:** The LinearDRLearner from the EconML package was used to estimate the average treatment effect.
- **Results:** This approach provided robust estimates even when either the outcome or the treatment model might be misspecified.

## Tableau Dashboard
For interactive visualization of these findings, please visit our Tableau dashboard:

[View Tableau Dashboard]((https://prod-useast-b.online.tableau.com/#/site/gwudats6401sg/workbooks/2571930?:origin=card_share_link))



## Conclusion
This study integrated robust data preprocessing, insightful visualizations, and advanced causal inference methods to shed light on the factors influencing food insecurity. The insights gained can help guide policy decisions aimed at mitigating food insecurity and improving community well‑being.

---

*This markdown file documents the analysis workflow and serves as a companion to the interactive Tableau dashboard for further exploration of the results.*
