# Armillaria Pathogen Risk Assessment Pipeline (GeoAI SDM)

[![CI Pipeline](https://github.com/DanilKras/ArmillariaGeoAIPipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/DanilKras/ArmillariaGeoAIPipeline/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/data-versioned%20by%20DVC-orange.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, configuration-driven Geospatial Machine Learning pipeline designed to model, explain, and project the spatial infection risk of *Armillaria ostoyae* across forest ecosystems, centered on the Sopron / Lake Fertő region (Hungary).

The architecture unites macro-scale Species Distribution Modeling (SDM) with micro-scale UAV/LiDAR canopy monitoring, utilizing cloud-native STAC catalogs, multi-dimensional Zarr stores, and spatial block cross-validation to account for spatial autocorrelation.

---

## Interactive Dashboard & Model Attribution

| Real-Time Spatial Risk & SHAP Attribution | Global Factor Attribution (SHAP Beeswarm) |
| :---: | :---: |
| ![Interactive UI Preview](data/03_results/figures/dashboard_preview.png) | ![SHAP Beeswarm Summary](data/03_results/figures/spatial_shap_summary.png) |

> 🔗 **Video Walkthrough:** [Project Demonstration Video](https://drive.google.com/file/d/1qoOQKh6-J5Bo6bT5EiugL-U74ZHSM_zP/view?usp=sharing)

---

## Key Achievements & Benchmarks

- **Mean Spatial ROC-AUC: 0.955 ± 0.017** on 5-fold Spatial GroupKFold Cross-Validation (150 km block buffers based on elevation semivariogram range), eliminating spatial data leakage.
- **Dual-Engine Inference Backend:** 
  - **Local Edge Engine:** Sub-50ms inference utilizing local precomputed 11-band GeoTIFF feature stacks.
  - **Cloud-Native Streaming Engine:** Serverless on-demand sampling via STAC APIs (Copernicus DEM, ESA WorldCover) and asynchronous Zarr temporal slicing (TerraClimate).
- **Single Source of Truth (SSOT):** Fully configuration-driven execution via `config/params.yaml`.
- **CMIP6 Projections:** Climate vulnerability scenarios evaluated under SSP2-4.5 and SSP5-8.5 horizons up to 2050.

---

## 11 Ecological Predictors

| Domain | Predictor Variable | Source / Resolution | Ecological Relevance |
| :--- | :--- | :--- | :--- |
| **Topography** | Elevation (m) | Copernicus GLO-30 (30 m) | Bioclimatic lapse rates & vertical zonation |
| | Slope (°) | Computed from DEM 3x3 kernel | Surface water runoff & erosion regimes |
| | Northness (Aspect) | $\cos(\text{aspect}_\text{rad})$ | Solar radiation, shade, and soil moisture |
| | Topographic Wetness Index (TWI) | $\ln(\alpha / \tan \beta)$ | Moisture accumulation in micro-depressions |
| **Land Cover** | Land Cover Class | ESA WorldCover (10 m) | Forest canopy host presence constraint (Class 10) |
| **Soil Properties**| Soil pH ($H_2O$, 0–5 cm) | ISRIC SoilGrids 2.0 (250 m) | Soil acidity; lower pH accelerates rhizomorph growth |
| | Soil Clay Content (%) | ISRIC SoilGrids 2.0 (250 m) | Soil aeration, drainage, and root asphyxiation risk |
| **Macroclimate** | Mean Annual Temp (`BIO1`) | TerraClimate Zarr (1995–2025) | Thermal niche envelope |
| | Temp Seasonality (`BIO4`) | TerraClimate Zarr (1995–2025) | Continental vs oceanic winter stress |
| | Annual Precipitation (`BIO12`)| TerraClimate Zarr (1995–2025) | Cumulative water availability |
| | Precip Seasonality (`BIO15`) | TerraClimate Zarr (1995–2025) | Drought vulnerability and rainfall distribution |

---

## Repository Structure

```text
├── .dvc/                           # Data Version Control internal configurations
│   ├── .gitignore
│   └── config
├── .github/workflows/              # CI/CD automated validation pipelines
│   └── ci.yml                      # GitHub Actions: Ruff linter, Pytest & Docker build
├── api/                            # Dual-Engine FastAPI backend service
│   ├── Dockerfile
│   ├── main.py                     # Local raster sampling & STAC/Zarr streaming endpoints
│   ├── requirements.txt
│   └── schemas.py                  # Pydantic spatial validation models
├── config/                         # Centralized Single Source of Truth (SSOT)
│   └── params.yaml                 # Ecological predictors, spatial bbox, and hyperparams
├── dags/                           # Workflow orchestration
│   └── armillaria_pipeline_dag.py  # Apache Airflow automated DAG
├── data/                           # Data directory managed via DVC
│   ├── 03_results/figures/         # SHAP plots, spatial maps & visual metrics
│   ├── .gitignore
│   ├── 01_raw.dvc                  # DVC tracker for raw GBIF occurrences
│   ├── 02_processed.dvc            # DVC tracker for Parquet datasets & spatial blocks
│   ├── 03_results.dvc              # DVC tracker for generated GeoTIFF rasters
│   └── micro_soe.dvc               # DVC tracker for UAV LiDAR & crown polygons
├── frontend/                       # Interactive Streamlit dashboard
│   ├── Dockerfile
│   ├── app.py                      # Leaflet risk viewer with live TreeSHAP breakdown
│   └── requirements.txt
├── models/                         # Model registry
│   └── .gitkeep                    # Serialized production XGBoost weights
├── notebooks/                      # Exploratory Data Analysis & spatial modeling
│   └── eda_armillaria.ipynb        # Spatial autocorrelation, semivariogram & sampling EDA
├── src/                            # Core research source code
│   ├── __init__.py
│   ├── macro/                      # Macro-Scale Regional SDM Pipeline (Operational)
│   │   ├── __init__.py
│   │   ├── data_extraction.py      # GBIF API occurrence harvester
│   │   ├── data_processing.py      # 1 km spatial rarefaction
│   │   ├── feature_extraction.py   # Planetary Computer STAC DEM & Zarr extractor
│   │   ├── inference.py            # 11-band regional GeoTIFF risk raster builder
│   │   ├── model_training.py       # 150 km Spatial Block CV & XGBoost training
│   │   └── negative_sampling.py    # KDTree balanced pseudo-absence generator
│   └── micro/                      # Micro-Scale UAV LiDAR Module (Active Research)
│       ├── __init__.py
│       ├── data_preprocessing.py   # Silvicultural inventory & crown parsing
│       └── lidar_processor.py      # Point cloud, CHM & orthophoto feature extraction
├── tests/                          # Automated Pytest suite
│   ├── test_config.py
│   └── test_negative_sampling.py   # Validation for spatial buffers & synthetic sampling
├── utils/                          # Shared system utilities
│   ├── config_loader.py            # Dynamic YAML configuration parser
│   ├── generate_future_risk_maps.py# CMIP6 2050 climate horizon projection engine
│   └── logger.py                   # Centralized structured logging setup
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile                      # Root container specification
├── README.md                       # Project technical documentation
├── airflow-requirements.txt        # Standalone dependencies for Airflow
├── docker-compose.yaml             # Multi-container orchestration (API, UI, Airflow)
├── pyproject.toml                  # Python package specifications and tool settings
├── requirements.txt                # Global Python dependencies
├── run_pipeline.py                 # CLI end-to-end execution script
└── uv.lock                         # Deterministic dependency lockfile
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- `uv` (recommended) or `pip`
- Docker & Docker Compose

### 1. Environment Setup

```bash
# Clone the repository
git clone [https://github.com/DanilKras/ArmillariaGeoAIPipeline.git](https://github.com/DanilKras/ArmillariaGeoAIPipeline.git)
cd ArmillariaGeoAIPipeline

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 2. Pulling Datasets via DVC

Heavy rasters, LiDAR point cloud products, and intermediate Parquet stores are versioned using DVC:

```bash
dvc pull
```

### 3. Running Services via Docker Compose

Launch the inference API, dashboard, and orchestration cluster:

```bash
docker-compose up --build -d
```

| Service | Endpoint | Description |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Interactive Leaflet inspection & live SHAP breakdown |
| **FastAPI Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | OpenAPI interactive documentation |
| **Airflow Webserver** | [http://localhost:8080](http://localhost:8080) | Pipeline execution DAG and task monitoring |

---

## Standalone Pipeline Execution

Execute the macro-scale workflow sequentially via CLI:

```bash
# 1. Occurrence ingestion & 1 km rarefaction
python -m src.macro.data_extraction
python -m src.macro.data_processing

# 2. Balanced pseudo-absence sampling
python -m src.macro.negative_sampling

# 3. Environmental extraction (Copernicus DEM, WorldCover, TerraClimate, SoilGrids)
python -m src.macro.feature_extraction

# 4. Spatial Block CV & Model Training
python -m src.macro.model_training

# 5. Risk map & 11-band GeoTIFF stack export
python -m src.macro.inference

# 6. CMIP6 Climate Horizon Projections (2050)
python -m utils.generate_future_risk_maps
```

---

## Scientific Methodology

1. **Combating Spatial Autocorrelation:** Standard random $K$-fold cross-validation yields overoptimistic metrics due to spatial dependence. We partition the study domain into 150 km geographic blocks (`GroupKFold`) informed by elevation semivariogram ranges, enforcing complete spatial independence between training and validation folds.
2. **Local & Global Interpretability:** TreeSHAP calculates exact additive feature contributions in log-odds space for each queried coordinate, linking statistical predictions to physical drivers (soil pH, moisture index, aspect).
3. **Multi-Scale GeoAI Vision:** The macro-scale model provides an ecological prior ($P_\text{macro}$ at $30\text{ m}$), which serves as a Bayesian boundary for micro-scale deep learning models processing individual crowns from UAV LiDAR ($1\text{ m}$ CHM) and ultra-high-resolution orthophotos ($20\text{ cm}$).

---

## License

Distributed under the [MIT License](LICENSE).
