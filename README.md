# **Armillaria Pathogen Risk Assessment Pipeline**

This project is a geospatial machine learning pipeline designed to model and map potential infection risks of *Armillaria ostoyae* in forest areas, with an initial test focus on the Sopron / Lake Fertő region.

&nbsp;

I built this project to explore cloud-native geospatial tools (STAC, Zarr) and practice spatial machine learning workflows using open environmental data.

# **Overview**

The workflow automates data collection, environmental feature extraction, spatial validation, and map generation:

&nbsp;

1. **Occurrence Data**: Fetches *Armillaria ostoyae* observations from the GBIF API, filtering records with coordinate uncertainty \<= 250 m.  
2. **Spatial Rarefaction**: Thins presence points to a 1x1 km grid to reduce sampling clustering.  
3. **Pseudo-Absence Sampling**: Generates balanced synthetic absence points at least 5 km away from known presences using vectorized NumPy distance calculations.  
4. **Feature Extraction**:  
   * Elevation and slope derived from the Copernicus 30m DEM via Microsoft Planetary Computer STAC.  
   * Forest mask applied using ESA WorldCover (10m).  
   * Bioclimatic variables (BIO1, BIO4, BIO12, BIO15) computed from TerraClimate Zarr arrays.  
5. **Spatial Cross-Validation**: Evaluates an XGBoost model using GroupKFold on 150 km blocks to prevent spatial autocorrelation leakage (the block size was guided by an empirical semivariogram of elevation).  
6. **Inference & Serving**: Produces a 30m regional risk raster (GeoTIFF) and provides a local FastAPI backend with a Streamlit interface for point predictions.

# **Project Structure**


├── api/                             \# FastAPI application  
│   ├── main.py                      \# Prediction and feedback endpoints  
│   ├── schemas.py                   \# Pydantic schemas  
│   └── Dockerfile  
├── config/  
│   └── params.yaml                  \# Configuration parameters  
├── dags/  
│   └── armillaria\_pipeline\_dag.py   \# Airflow pipeline DAG  
├── data/  
│   ├── 01\_raw/                      \# Raw downloaded points  
│   ├── 02\_processed/                \# Processed feature tables (Parquet)  
│   └── 03\_results/                  \# Output GeoTIFF and plots  
├── frontend/                        \# Streamlit dashboard  
│   ├── app.py                       \# Map UI and SHAP visualization  
│   └── Dockerfile  
├── models/  
│   └── armillaria\_spatial\_xgb.json  \# Trained XGBoost model  
├── notebooks/  
│   └── 01\_eda\_armillaria.ipynb      \# EDA and variogram analysis  
├── src/                             \# Pipeline source code  
│   ├── data\_extraction.py  
│   ├── data\_processing.py  
│   ├── negative\_sampling.py  
│   ├── feature\_extraction.py  
│   ├── model\_training.py  
│   └── inference.py  
├── docker-compose.yaml  
└── pyproject.toml

&nbsp;

\#\# Installation & Setup

&nbsp;

\#\#\# Requirements

&nbsp;

\*   Python 3.11+

&nbsp;

\*   uv or pip

&nbsp;

\#\#\# Step 1: Set up virtual environment

&nbsp;

Using \*\*uv\*\*:

&nbsp;

\`\`\`bash

&nbsp;

uv venv .venv

&nbsp;

source .venv/bin/activate  \# On Windows: .venv\\Scripts\\activate

&nbsp;

uv sync

&nbsp;

Or using standard **pip**:python \-m venv .venv

&nbsp;

source .venv/bin/activate  \# On Windows: .venv\\Scripts\\activate

&nbsp;

pip install \-r requirements.txt

## **Step 2: Code quality check**

uv run ruff check . \--fix

&nbsp;

uv run ruff format .

# **Running the Pipeline**

You can run each stage of the pipeline sequentially:\# 1\. Fetch GBIF occurrences

&nbsp;

python \-m src.data\_extraction

&nbsp;

\# 2\. Thin occurrences to 1x1 km grid

&nbsp;

python \-m src.data\_processing

&nbsp;

\# 3\. Generate pseudo-absences

&nbsp;

python \-m src.negative\_sampling

&nbsp;

\# 4\. Extract DEM and TerraClimate features

&nbsp;

python \-m src.feature\_extraction

&nbsp;

\# 5\. Train XGBoost model with Spatial CV

&nbsp;

python \-m src.model\_training

&nbsp;

\# 6\. Generate regional 30m risk GeoTIFF

&nbsp;

python \-m src.inference

# **Running the Web Services (Docker)**

To run the API and dashboard locally using Docker Compose:docker-compose up \--build

&nbsp;

* **Streamlit Interface**: http://localhost:8501  
* **FastAPI Documentation**: http://localhost:8000/docs

# **Key Notes on Methodology**

* **Spatial Block Cross-Validation**: Standard random splits often lead to overly optimistic performance due to spatial autocorrelation. Using 150 km blocks ensures that validation folds are geographically separated from training folds.  
* **Ecological Limitations**: This project relies on presence-only data from citizen science (GBIF), which contains noticeable geographic observation bias (e.g., higher reporting density in Germany and Denmark). Pseudo-absences are generated purely based on spatial distance and should be refined with actual host-tree species distribution and soil characteristics in future work.

# **License**

This project is licensed under the MIT License.