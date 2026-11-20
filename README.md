
# Online Machine Learning with River and CapyMOA

An experimental study of online machine learning methods using the Python libraries River and CapyMOA.

**Academic project:** M2 Data Science, Institut Polytechnique de Paris  
**Author:** Doha JOUAY

## Overview

This project explores how incremental learning algorithms perform on streaming data, with a focus on predictive performance, computational cost, and concept drift.

## Experiments

### 1. Online Regression (River)
- Hoeffding Tree Regressor
- Adaptive Random Forest Regressor
- Linear Regression with Adam
- K-Nearest Neighbors

Datasets: TrumpApproval, ChickWeights, Bikes, and synthetic Friedman drift.

Metrics: MAE, RMSE, R², and execution time.

### 2. Online Classification (River)
- Hoeffding Tree Classifier
- Adaptive Random Forest Classifier
- Logistic Regression with Adam
- K-Nearest Neighbors

Datasets: Elec2, Phishing, Bananas, and synthetic SEA drift.

Metrics: accuracy, macro F1, Cohen's Kappa, and execution time.

### 3. Concept Drift Detection
Comparison of ADWIN, Page-Hinkley, KSWIN, DDM, and EDDM, including detection delays and the effect of resetting a model after drift alarms.

### 4. River vs. CapyMOA
Comparison of related classification algorithms across the two libraries, focusing on accuracy and processing time.

### 5. Stream Clustering
Exploration of CluStream + KMeans and DenStream + DBSCAN on evolving synthetic data streams.

### 6. Online AutoML
Comparison of a fixed Hoeffding Tree with BanditClassifier and SuccessiveHalvingClassifier.

## Key Findings

- No single algorithm performed best on every dataset.
- Linear models were generally computationally efficient.
- Adaptive Random Forest performed well on synthetic nonlinear regression but produced extreme prediction errors on some real datasets.
- Drift detection did not always improve downstream predictive performance.
- AutoML delivered modest accuracy gains at a substantially higher computational cost.


## Limitations

The experiments use limited stream lengths and single-run runtime measurements. Some cross-library preprocessing and hyperparameters are not strictly equivalent.

Clustering comparisons are exploratory rather than definitive quantitative benchmarks.

## References

- [River documentation](https://riverml.xyz/)
- [CapyMOA documentation](https://capymoa.org/)
