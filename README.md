# Armillaria Pathogen Risk Assessment Pipeline

[![CI Pipeline](https://github.com/DanilKras/ArmillariaGeoAIPipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/DanilKras/ArmillariaGeoAIPipeline/actions/workflows/ci.yml/badge.svg)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is a geospatial machine learning pipeline designed to model and map potential infection risks of *Armillaria ostoyae* in forest ecosystems, with a target validation area in the Sopron / Lake Fertő region.

The goal is to demonstrate a production-ready, cloud-native GeoAI workflow utilizing STAC catalogs, Zarr cloud data stores, and spatial cross-validation to account for spatial autocorrelation.

---

## Overview

The pipeline automates raw biodiversity data ingestion, cloud-based environmental feature extraction, spatial machine learning validation, regional raster inference, and interactive serving.

### Core Stages

1. **Occurrence Ingestion:** Fetches verified *Armillaria ostoyae* presence records from the GBIF API, filtering for coordinate uncertainty $\le 250\text{ m}$.
2. **Spatial Rarefaction:** Thins occurrences to a 1×1 km metric grid (EPSG:3035) to mitigate opportunistic citizen-science sampling clustering.
3. **Pseudo-Absence Sampling:** Generates a balanced 1:1 synthetic absence dataset ($>5\text{ km}$ buffer from observed presences) using spatial indexing via `scipy.spatial.KDTree`.
4. **Cloud Feature Store Extraction:**
    - **Elevation:** Copernicus GLO-30 DEM (30 m) via Microsoft Planetary Computer STAC.
    - **Land Cover Mask:** ESA WorldCover (10 m) mapped to identify forest canopy areas (Class 10).
    - **Bioclimatic Predictors:** Long-term bioclimatic indices (`BIO1`, `BIO4`, `BIO12`, `BIO15`) streamed on demand from TerraClimate Zarr arrays.
5. **Spatial Cross-Validation:** Evaluates an XGBoost classifier with 5-fold `GroupKFold` over 150 km geographic blocks (derived from elevation semivariogram sill analysis) to prevent spatial data leakage.
6. **Inference & Serving:** Exports a high-resolution regional risk GeoTIFF raster (30 m) and serves point inferences and SHAP explainability via a FastAPI backend and a Streamlit dashboard.

---

## Project Structure

```text
├── .cadence/                       # Cadence configurations (if applicable)
├── .dvc/                           # Data Version Control internal files
├── .github/workflows/ci.yml        # GitHub Actions CI (Ruff, Pytest, Docker build)
├── api/                            # FastAPI backend service
│   ├── Dockerfile
│   ├── main.py                     # Inference and field feedback endpoints
│   ├── requirements.txt
│   └── schemas.py                  # Pydantic data schemas for validation
├── config/
│   └── params.yaml                 # Centralized pipeline configuration
├── dags/
│   └── armillaria_pipeline_dag.py  # Apache Airflow orchestration DAG
├── data/                           # Data directory managed by DVC
│   ├── 01_raw/                     # Raw fetched occurrences
│   ├── 02_processed/               # Thinned features and Parquet tables
│   ├── 03_results/                 # Outputs
│   │   ├── figures/                # SHAP plots and visual metrics
│   │   └── maps/                   # Generated GeoTIFF risk rasters
│   ├── 01_raw.dvc                  # DVC tracking files
│   ├── 02_processed.dvc            
│   └── 03_results.dvc              
├── frontend/                       # Streamlit web application
│   ├── app.py                      # Interactive Leaflet map & SHAP diagnostic plots
│   ├── Dockerfile
│   └── requirements.txt
├── logs/                           # Airflow and execution logs
├── models/
│   └── armillaria_spatial_xgb.json # Exported production XGBoost model weights
├── notebooks/
│   └── 01_eda_armillaria.ipynb     # Spatial EDA, variogram, and sampling analysis
├── src/                            # Core pipeline modules
│   ├── macro/                      # Macro-scale regional modeling
│   │   ├── __init__.py
│   │   ├── data_extraction.py      # GBIF API occurrence harvester
│   │   ├── data_processing.py      # 1 km spatial rarefaction
│   │   ├── feature_extraction.py   # Planetary Computer DEM & Zarr bioclim extraction
│   │   ├── inference.py            # Regional 30m GeoTIFF risk raster generation
│   │   ├── model_training.py       # Spatial Block CV & XGBoost training
│   │   └── negative_sampling.py    # KDTree ecological pseudo-absence generator
│   └── micro/                      # Micro-scale local modeling (Thesis Expansion)
│       ├── __init__.py
│       └── lidar_processor.py      # Micro-Scale LiDAR & 3D Point Cloud Processing
├── tests/                          # Automated Pytest test suite
│   ├── test_config.py
│   └── test_negative_sampling.py
├── utils/                          # Shared utilities
│   ├── config_loader.py            # Dynamic YAML configuration loader
│   └── logger.py                   # Centralized logging setup
├── .dockerignore                   # Docker exclusion rules
├── .dvcignore                      # DVC exclusion rules
├── .gitignore                      # Git exclusion rules
├── airflow-requirements.txt        # Specific dependencies for Airflow
├── docker-compose.yaml             # Multi-container orchestration (API, Frontend, Airflow)
├── Dockerfile                      # Root Dockerfile for pipeline execution
├── pyproject.toml                  # Python project metadata and tooling config (Ruff, Pytest)
├── README.md                       # Project documentation
├── requirements.txt                # Global project dependencies
├── run_pipeline.py                 # Standalone CLI pipeline runner
└── uv.lock                         # uv lockfile for deterministic dependency resolution
```

## Installation & Setup

### Requirements

- Python 3.12+
- `uv` or `pip`

### Step 1: Set Up Virtual Environment

**Using `uv`:**

```bash
uv venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv sync
```

**Or using standard `pip`:**

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

You can run each stage of the pipeline sequentially:

1. Harvest GBIF occurrences:

```bash
   python -m src.macro.data_extraction
   ```

2. Thin presences to 1x1 km grid:

```bash
  python -m src.macro.data_processing
```

3. Sample balanced ecological pseudo-absences:

```bash
  python -m src.macro.negative_sampling
```

4. Stream DEM and TerraClimate Zarr variables:

```bash
  python -m src.macro.feature_extraction
```

5. Train XGBoost model with 150 km Spatial Block CV:

```bash
  python -m src.macro.model_training
```

6. Generate 30m regional risk GeoTIFF raster:

```bash
  python -m src.macro.inference
```

7. Running the Web Services (Docker)
   To run the API and dashboard locally using Docker Compose:

```bash
  docker-compose up --build
```

| Service                 | URL                                                      | Description                                    |
|:------------------------|:---------------------------------------------------------|:-----------------------------------------------|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501)           | Interactive map risk viewer & SHAP diagnostics |
| **FastAPI Swagger UI**  | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI documentation & inference  |
| **Airflow Orchestrator**| [http://localhost:8080](http://localhost:8080)           | Workflow orchestration & scheduling            |

---

### Automated Execution

Run the complete pipeline end-to-end via CLI:
```bash
uv run python run_pipeline.py
```
---

## Link to demo

[Project Demonstration](https://drive.google.com/file/d/1qoOQKh6-J5Bo6bT5EiugL-U74ZHSM_zP/view?usp=sharing)

## Key Notes on Methodology

* **Spatial Block Cross-Validation**: Standard random K-Fold cross-validation produces overoptimistic metric estimates
  when applied to spatial observations due to Tobler's First Law of Geography. Partitioning the study area into 150 km
  blocks (GroupKFold) ensures out-of-region generalization without spatial data leakage.
* **Ecological Limitations**: Presence records derived from GBIF citizen science exhibit observation density bias toward
  Northern and Western Europe (e.g., high reporting rates in Germany and Denmark). While 1 km spatial rarefaction
  suppresses dense urban reporting clusters, model projections reflect realized ecological reporting niches.

---

## Future Roadmap (MSc Thesis Integration)

The current pipeline establishes a robust Macro-Scale foundation for automated spatial feature extraction, environmental
modeling, and rigorous block cross-validation. As part of my upcoming MSc thesis research, this architecture is being
actively expanded into a hybrid **Multi-Scale GeoAI Framework**.

Upcoming developments focus on the **Micro-Scale** level, specifically targeting:

* **LiDAR & 3D Canopy Integration:** Ingesting UAV-based LiDAR point clouds and Canopy Height Models (CHM) to extract
  high-resolution structural features and perform precise single-tree segmentation.
* **Deep Learning Spatial Encoders:** Implementing PyTorch-based architectures to process co-registered Sentinel-1 (SAR)
  and Sentinel-2 patches for localized canopy health assessment.
* **Hybrid Validation:** Fusing micro-scale neural embeddings with the existing macro-scale Gradient Boosting
  (XGBoost/CatBoost) pipeline to enhance localized vulnerability predictions and 3D crown validation.

## License

This project is licensed under the [MIT License](LICENSE).
