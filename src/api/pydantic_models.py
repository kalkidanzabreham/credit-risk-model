from pydantic import BaseModel

class PredictionRequest(BaseModel):
    CustomerId: int
    ProductCategory: str
    ChannelId: str
    CountryCode: int
    PricingStrategy: int
    Amount: float
    TransactionStartTime: str


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool
