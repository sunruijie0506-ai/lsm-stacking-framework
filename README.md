Integrating InSAR Time-Series and Ensemble Learning for Corri-dor-Scale Landslide Susceptibility Assessment in the Three Gorges Reservoir Area, China
Overview

This repository provides the source code and de-identified processed data required to reproduce the methodological framework presented in the associated manuscript.

The study proposes a point–line–area integrated framework for landslide susceptibility mapping in a mountainous highway corridor. The workflow integrates:

Point scale: Multi-temporal InSAR deformation time-series and trend classification

Line scale: Spatial autocorrelation diagnostics (Moran’s I, LISA, and Consistency–Significance Index, CSI)

Area scale: Bayesian-optimized stacking ensemble learning at the slope-unit scale

These components are combined through a rule-based refinement strategy to generate the final susceptibility classification.

Methodological Workflow
Environmental Factors + Landslide Inventory
                │
                ▼
   Interaction Feature Construction
                │
                ▼
Bayesian-Optimized Stacking Ensemble
 (XGBoost + Random Forest + MLP → Logistic Regression)
                │
                ▼
Initial Susceptibility Probability (Slope Unit Scale)
                │
                ├───────── Spatial Autocorrelation (Moran’s I / LISA → CSI)
                │
                └───────── InSAR Trend Classification
                │
                ▼
        Rule-Based Refinement
                │
                ▼
        Final Susceptibility Zoning
Repository Structure
.
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── train_stacking_optuna.py
│   ├── lisa_and_weights.py
│   ├── compute_csi.py
│   ├── insar_trend_classification.py
│   └── refine_rules.py
│
├── data/
│   ├── raw/                (restricted; not publicly released)
│   └── processed/          (de-identified reproducible data)
│
├── configs/
│   └── config.yaml
│
├── outputs/                (generated during execution)
│
└── docs/
    └── data_dictionary.md
Environment Requirements

Tested under:

Python ≥ 3.9

Windows / Linux

Main dependencies:

numpy

pandas

scikit-learn

xgboost

optuna

geopandas

libpysal

esda

matplotlib

joblib

Install dependencies:

pip install -r requirements.txt
Quick Reproduction Guide
Step 1 — Install Dependencies
pip install -r requirements.txt
Step 2 — Configure Parameters

Edit:

configs/config.yaml

Specify:

Data paths

Hyperparameter ranges

Spatial weight parameters

Rule thresholds

Output directory

Step 3 — Train the Stacking Ensemble (Methods Section 3.2)
python src/train_stacking_optuna.py

Outputs:

best_params.json

cv_auc.txt

test_metrics.json

stacking_model.pkl

Feature importance files

Step 4 — Spatial Autocorrelation Analysis
python src/lisa_and_weights.py
python src/compute_csi.py

Outputs:

Moran’s I statistics

LISA cluster classification

CSI index

Step 5 — InSAR Trend Classification
python src/insar_trend_classification.py

Outputs:

Slope-unit trend categories (accelerating / seasonal / stable)

Step 6 — Rule-Based Refinement
python src/refine_rules.py

Outputs:

Refined susceptibility classification

Final slope-unit attribute table

Output Files Description

Key outputs include:

best_params.json
Optimal hyperparameters obtained via Bayesian optimization.

cv_auc.txt
Mean cross-validation ROC-AUC score.

test_metrics.json
Independent test-set performance metrics.

xgb_gain.csv
XGBoost feature importance (gain).

rf_mdi.csv
Random Forest feature importance (mean decrease in impurity).

mlp_permutation_auc.csv
Permutation importance for MLP.

meta_lr_stdcoef.csv
Standardized regression coefficients of the logistic regression meta-learner.

final_slopeunit_attributes.csv
Final susceptibility classification for each slope unit (anonymous ID only).

Data Availability
Geospatial Data Restriction

Due to national data security regulations, original geospatial datasets containing explicit coordinates or geometries (e.g., raster layers, vector files, spatial boundaries, and coordinate fields) cannot be publicly released.

This includes:

Raster grids (DEM, rainfall rasters, NDVI rasters, etc.)

Vector layers (slope units, landslide polygons, faults, rivers)

Any coordinate or geometry information

Reproducible Data Provided

To ensure methodological transparency and reproducibility, the following de-identified processed datasets are provided:

training_table_no_geo.csv
Feature matrix and labels used for model training/testing (no coordinates).

slopeunit_attributes_no_geo.csv
Slope-unit level derived attributes including susceptibility probabilities, CSI values, InSAR summary metrics, and final classification.

weights_adjacency.csv
Anonymous spatial weight representation (i, j, w_ij) enabling full reproduction of Moran’s I, LISA, and CSI calculations without geographic coordinates.

insar_unit_features.csv
Aggregated InSAR-derived deformation metrics at slope-unit level.

These files are sufficient to reproduce:

Model training and validation results

Cross-validation AUC and test metrics

Feature importance analysis

Spatial autocorrelation statistics

CSI computation

Rule-based refinement outcomes

Map rendering and geographic visualization cannot be reproduced without restricted spatial layers, but all numerical results used to generate the maps are fully reproducible.

Reproducibility Notes

All random seeds are fixed.

Train/test split is stratified.

Hyperparameter optimization uses 5-fold stratified cross-validation.

Objective function maximizes mean ROC-AUC.

Meta-learner uses probability outputs only (no raw features).

Limitations

The repository reproduces all numerical results and classification outputs.
Geographic map visualization is not included due to regulatory restrictions on spatial data sharing.

License

This project is licensed under the MIT License.
See the LICENSE file for details.  
