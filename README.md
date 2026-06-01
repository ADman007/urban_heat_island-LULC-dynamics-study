# Multi-Temporal Urban Heat Island (UHI) Dynamics and Deep Learning LULC Classification (Bokaro: 2015–2024)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript-green)](https://earthengine.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 📌 Project Overview
This repository contains a multi-model machine learning framework designed to detect decadal Land Use/Land Cover (LULC) changes and quantify Urban Heat Island (UHI) intensification in the Bokaro district of Jharkhand, India, between 2015 and 2024. 

By combining cloud-scale geospatial processing in Google Earth Engine (GEE) with local deep learning benchmarking, this project addresses the critical challenge of **temporal domain shift** in satellite imagery. The methodology deliberately decouples absolute Land Surface Temperature (LST) from the classification feature space, forcing models to rely on structurally stable spectral indices to eliminate cross-epoch misclassification artifacts.

<p align="center">
  <img src="https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Bokaro_District_map.png" alt="Map of Bokaro District" width="500"><br>
  <sub><b>Figure 1:</b> Map of Bokaro District</sub>
</p>


## 🔬 Key Methodologies
* **Cloud-Native Data Engineering:** Constructed pre-monsoon Landsat 8 surface reflectance composites, utilizing a strict <15% scene-level cloud filter and QA_PIXEL masking to bypass atmospheric contamination.
* **Feature Decoupling:** Engineered a 9-feature spectral stack including physical bands (SR_B2 to SR_B7) and indices (NDVI, NDBI, MNDWI).
* **Spatial Independence:** Implemented a robust 80/20 train-test split at the *polygon level* (rather than pixel level) using a seeded random column to strictly eliminate spatial autocorrelation and data leakage.
* **Meteorological Detrending:** Integrated ERA5-Land reanalysis data to isolate LULC-driven thermal penalties from broader synoptic climate heatwaves.

## 📊 Algorithmic Benchmarking
The decoupled 9-band spectral features were exported as a Float32 GeoTIFF and benchmarked locally to evaluate performance consistency. The near-identical performance across architectures demonstrates that rigorous feature engineering renders the spatial dataset highly separable.

| Model Architecture | Configuration Details | Macro F1 Score |
| :--- | :--- | :--- |
| **Random Forest** | 100 estimators, Gini impurity | **0.93** |
| **XGBoost** | 200 estimators, max depth 6, learning rate 0.1 | **0.92** |
| **PyTorch MLP** | 4-layer fully connected (128->128->64->32), BatchNorm, Dropout | **0.92** |

![ 2024 LULC Classification Benchmarking Maps](https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Screenshot%202026-05-17%20081638.png)

## 🧠 Mechanistic Interpretability (SHAP)
Post-hoc SHAP (SHapley Additive exPlanations) attribution was applied to a stratified test sample to validate the physical grounding of the models:
* **Tree-based models (RF/XGBoost)** utilized `MNDWI` as a primary, rule-based "veto" threshold to differentiate built-up concrete from water.
* **Deep Learning architectures (MLP)** natively discovered complex nonlinear representations, relying heavily on raw Shortwave Infrared (SWIR) reflectance (`SR_B6`, `SR_B7`) while suppressing pre-engineered indices.

![RandomForest_Beeswarm_plot](https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Screenshot%202026-05-16%20130403.png)
![XGBoost_Beeswarm_plot](https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Screenshot%202026-05-16%20125747.png)
![Neural_Network_Beeswarm_plot](https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Screenshot%202026-05-16%20130558.png)

## 📈 Key Findings
* **Urban Expansion:** The built-up extent in the Bokaro district expanded by **7.9%** (from 359.2 sq km to 387.6 sq km) over the decade.
* **Classification Accuracy:** Achieved a spatially independent overall accuracy of **90.1%** (Kappa: 0.75) for the 2015 baseline and **98.5%** (Kappa: 0.93) for the 2024 target epoch. 
* **UHI Intensification:** While a regional heatwave drove absolute temperatures up across all land covers, the isolated Urban Heat Island Intensity (UHII) gap between urban and rural baselines widened by 50% (from **+0.28°C** to **+0.42°C**).
* **Detrended Thermal Penalty:** After accounting for a 1.19°C background regional climate warming via ERA5, the localized LULC-attributable surface warming was estimated at **5.26°C**.

<p align="center">
  <img src="https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Screenshot%202026-05-19%20210803.png" alt="Air_Temperature_over_the_years" width="500"><br>
  <sub><b>Figure 6:</b> Air_Temperature_over_the_years</sub>
</p>


<p align="center">
  <img src="https://github.com/ADman007/urban_heat_island-LULC-dynamics-study/blob/main/figures/Screenshot%202026-05-20%20201452.png" alt="Areas where builtUp area increased during the study period" width="500"><br>
  <sub><b>Figure 7:</b> Areas where builtUp area increased during the study period.</sub>
</p>

## 🚀 Usage & Reproduction

### 1. Google Earth Engine (Cloud Preprocessing)
The JavaScript file in `code/GEE_script.js` contain the cloud-masking, feature extraction, and sampling logic. Paste these into the GEE Code Editor to generate the datasets.

### 2. Local Inference (Model Training)
Clone the repository and install the required dependencies to run the benchmarking models:
```bash
git clone [https://github.com/ADman007/urban_heat_island-LULC-dynamics.git](https://github.com/ADman007/urban_heat_island-LULC-dynamics.git)
cd urban_heat_island-LULC-dynamics
pip install -r requirements.txt
