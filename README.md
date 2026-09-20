# Armillaria Pathogen Risk Assessment Pipeline

This project is a geospatial machine learning pipeline designed to model and map potential infection risks of *Armillaria ostoyae* in forest areas, with an initial test focus on the Sopron / Lake Fertő region.

The goal of the project is to explore cloud-native geospatial tools (STAC, Zarr) and practice spatial machine learning workflows using open environmental data.

---

## Overview

The workflow automates data collection, environmental feature extraction, spatial validation, and map generation.

### Core Stages

1. **Occurrence Data**  
   Fetches *Armillaria ostoyae* observations from the GBIF API, filtering records with coordinate uncertainty ≤ 250 m.

2. **Spatial Rarefaction**  
   Thins presence points to a 1×1 km grid to reduce sampling clustering.

3. **Pseudo‑Absence Sampling**  
   Generates balanced synthetic absence points at least 5 km away from known presences using vectorized NumPy distance calculations.

4. **Feature Extraction**  
   - **Elevation and slope**: derived from the Copernicus 30 m DEM via Microsoft Planetary Computer STAC.  
   - **Forest mask**: applied using ESA WorldCover (10 m).  
   - **Bioclimatic variables**: BIO1, BIO4, BIO12, BIO15 computed from TerraClimate Zarr arrays.

5. **Spatial Cross‑Validation**  
   Evaluates an XGBoost model using `GroupKFold` on 150 km blocks to prevent spatial autocorrelation leakage. The block size was guided by an empirical semivariogram of elevation.

6. **Inference & Serving**  
   Produces a 30 m regional risk raster (GeoTIFF) and provides a local FastAPI backend with a Streamlit interface for point predictions.

---

## Project Structure

```text
├── api/                            # FastAPI application
│   ├── main.py                     # Prediction and feedback endpoints
│   ├── schemas.py                  # Pydantic schemas
│   └── Dockerfile
├── config/
│   └── params.yaml                 # Configuration parameters
├── dags/
│   └── armillaria_pipeline_dag.py  # Airflow pipeline DAG
├── data/
│   ├── 01_raw/                     # Raw downloaded points
│   ├── 02_processed/               # Processed feature tables (Parquet)
│   └── 03_results/                 # Output GeoTIFF and plots
├── frontend/                       # Streamlit dashboard
│   ├── app.py                      # Map UI and SHAP visualization
│   └── Dockerfile
├── models/
│   └── armillaria_spatial_xgb.json # Trained XGBoost model
├── notebooks/
│   └── 01_eda_armillaria.ipynb     # EDA and variogram analysis
├── src/                            # Pipeline source code
│   ├── data_extraction.py
│   ├── data_processing.py
│   ├── negative_sampling.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   └── inference.py
├── docker-compose.yaml
└── pyproject.toml
```

## Installation & Setup

### Requirements

- Python 3.11+
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
### Step 2: Code Quality Check

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Running the Pipeline

You can run each stage of the pipeline sequentially:

1. Fetch GBIF occurrences:
   ```bash
   python -m src.data_extraction
2. Thin occurrences to 1×1 km grid:

  ```bash
  python -m src.data_processing
```
3. Generate pseudo-absences:

  ```bash
  python -m src.negative_sampling
```
4. Extract DEM and TerraClimate features:

  ```bash
  python -m src.feature_extraction
```
5. Train XGBoost model with Spatial CV:

  ```bash
  python -m src.model_training
```
6. Generate regional 30m risk GeoTIFF:

  ```bash
  python -m src.inference
```
7. Running the Web Services (Docker)
To run the API and dashboard locally using Docker Compose:

  ```bash
  docker-compose up --build
```
| Service | URL | Description |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Interactive map risk viewer & SHAP diagnostics |
| **FastAPI Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI documentation & inference |
---

## Key Notes on Methodology

* **Spatial Block Cross-Validation**: Standard random splits often lead to overly optimistic performance due to spatial autocorrelation. Using 150 km blocks ensures that validation folds are geographically separated from training folds.
* **Ecological Limitations**: This project relies on presence-only data from citizen science (GBIF), which contains noticeable geographic observation bias (e.g., higher reporting density in Germany and Denmark). Pseudo-absences are generated purely based on spatial distance and should be refined with actual host-tree species distribution and soil characteristics in future work.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
