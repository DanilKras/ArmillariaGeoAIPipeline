from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    lat: float = Field(..., description="Latitude in WGS84", ge=-90.0, le=90.0, examples=[47.68])
    lon: float = Field(..., description="Longitude in WGS84", ge=-180.0, le=180.0, examples=[16.58])


class ExtractedFeatures(BaseModel):
    elevation_m: float
    landcover_class: int
    bio1_temp_c: float
    bio4_seasonality: float
    bio12_precip_mm: float
    bio15_precip_seas: float


class InferenceResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probability (0.0 - 1.0)")
    risk_level: str = Field(..., description="Low, Medium, or High")
    extracted_features: ExtractedFeatures
