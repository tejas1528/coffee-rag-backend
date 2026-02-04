"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from typing import Optional, Literal, Union

# Request schemas
class RecipeRequest(BaseModel):
    preference: str = Field(
        ..., 
        description="User's coffee preference",
        example="I want bitter coffee with sweetness"
    )

    filter_type: Optional[Literal["min_rating", "cluster"]] = Field(
        None,
        description="Type of filter to apply (only one allowed)"
    )

    filter_value: Optional[Union[int, float]] = Field(
        None,
        description="Value for the selected filter"
    )

    n_results: int = Field(
        50, 
        ge=10, 
        le=200, 
        description="Number of results to retrieve"
    )

class SimilarCoffeesRequest(BaseModel):
    query: str = Field(..., description="Search query", example="sweet coffee with chocolate")
    n_results: Optional[int] = Field(10, ge=1, le=50, description="Number of results")

class CompareRequest(BaseModel):
    preference_1: str = Field(..., description="First preference", example="sweet coffee")
    preference_2: str = Field(..., description="Second preference", example="bold coffee")

class PredictionRequest(BaseModel):
    sensory_features: List[float] = Field(..., description="20 sensory features in order")
    user_preference: Optional[str] = Field(None, description="Optional preference for context")

class ParameterDetails(BaseModel):
    optimal: Optional[float] = None
    range: Optional[List[float]] = None
    sweet_spot: Optional[List[float]] = None

class TDSDetails(BaseModel):
    optimal: Optional[float] = None
    range: Optional[List[float]] = None

class BrewingParameters(BaseModel):
    temperature: ParameterDetails
    tds: TDSDetails
    extraction: TDSDetails
    dose: TDSDetails

class ExpectedResults(BaseModel):
    average_rating: float
    rating_range: List[float]
    most_common_rating: float

class TopExample(BaseModel):
    judge: int
    rating: float
    temp: float
    tds: float
    extraction: float

class RecipeResponse(BaseModel):
    query: str
    sample_size: int
    brewing_parameters: BrewingParameters
    expected_results: ExpectedResults
    sensory_profile: Dict[str, float]
    top_examples: List[TopExample]
    brewing_tips: List[str]

class CoffeeDetail(BaseModel):
    description: str
    rating: float
    judge: int
    cluster: int
    brewing: Dict[str, float]
    flavors: Dict[str, bool]

class SimilarCoffeesResponse(BaseModel):
    query: str
    count: int
    coffees: List[CoffeeDetail]

class PredictionResponse(BaseModel):
    rf_prediction: float
    rag_context: Optional[Dict] = None
    combined_prediction: Optional[float] = None
    explanation: str
    feature_importance: Dict[str, float]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None