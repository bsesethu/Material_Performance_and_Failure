# 📊Material Performance and Failure Analysis
Project Overview

This project investigates material fatigue behaviour across Low-Cycle Fatigue (LCF) and High-Cycle Fatigue (HCF) regimes using a combination of classical materials engineering theory and data-driven modelling techniques.

The objective is to analyse experimental fatigue data, extract meaningful engineering insights, and evaluate the applicability and limitations of regression and machine learning models in predicting fatigue life.

Key Objectives

Analyse fatigue behaviour in LCF and HCF regimes

Explore strain–life and stress–life relationships

Apply classical fatigue models (Coffin–Manson, Basquin)

Evaluate regression and machine learning approaches

Compare predictive performance across fatigue regimes

Assess the limitations of data-driven models in physical systems

Dataset Description

The project uses cleaned experimental fatigue datasets, including:

LCF dataset: Strain-controlled fatigue data with elastic, plastic, and total strain components

HCF dataset: Stress-controlled fatigue data with stress amplitude, frequency, and derived features

All datasets were preprocessed to remove inconsistencies, handle missing values, and ensure modelling readiness.

Project Structure
├── lcf_clean.csv                  # Cleaned LCF dataset
├── hcf_clean.csv                  # Cleaned HCF dataset
├── all_in_one_Clean_nEDA.py        # Data cleaning and exploratory data analysis
├── all_in_one_modelling.py         # Model building and evaluation
├── Report_final.docx               # Final project report (Word)
├── Report_final.pdf                # Final project report (PDF)
├── .venv/                          # Virtual environment (excluded from version control)
└── README.md                       # Project documentation

Methodology
1. Data Preparation & Exploration

Data cleaning and feature validation

Univariate, bivariate, and correlation analysis

Regime-specific exploration for LCF and HCF

2. Modelling

LCF:

Linear regression

Coffin–Manson strain–life analysis

HCF:

Linear regression with polynomial features

Basquin-type stress–life modelling

Comparison with ensemble methods

3. Model Evaluation

Train–test split (80/20)

Performance metrics:

R² (primary)

RMSE (secondary)

Assessment of model saturation and diminishing returns

Key Findings

LCF behaviour is well-structured and strain-dominated, allowing classical fatigue theory to be captured effectively using regression models.

HCF behaviour exhibits significant scatter, limiting predictive accuracy despite statistical significance.

Increasing model complexity does not necessarily improve performance in physically noisy systems.

Data-driven models are most effective when guided by strong domain knowledge.

Tools & Technologies

Python

NumPy, Pandas

Matplotlib, Seaborn

scikit-learn

Limitations

Experimental scatter, particularly in HCF data

Limited control over external variables (e.g. temperature, microstructure)

Predictive models should be interpreted as decision-support tools, not absolute predictors

Future Work

More controlled experimental datasets

Expanded frequency and temperature studies

Incorporation of microstructural descriptors

Physics-informed or hybrid modelling approaches

Author

Sesethu (Matt) Bango
Mechanical Engineering & Data Science
