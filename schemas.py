"""
Database Schemas for Trading Assistant

Each Pydantic model here maps to a MongoDB collection (lowercased class name).
"""
from pydantic import BaseModel, Field
from typing import Optional

class WatchItem(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    note: Optional[str] = Field(None, description="Optional note about this ticker")
