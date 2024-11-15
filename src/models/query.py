from typing import Optional
from pydantic import BaseModel, Field, field_validator
from enum import IntEnum
from models.product import Product
from qdrant_client.client_base import models


class RetrievalType(IntEnum):
    hybrid = 0
    semantic = 1
    keyword = 2


class Query(BaseModel):
    """Query model with validation"""
    query: str = Field(..., min_length=1, max_length=255)
    retrieval_type: Optional[RetrievalType] = Field(default=RetrievalType.hybrid)
    size: Optional[int] = Field(5, ge=1, le=255)
    filters: Optional[dict] = Field({})

    @field_validator('filters')
    def validate_filters(cls, field_value):
        for key in field_value:
            if key not in Product.model_fields:
                raise ValueError(f'filter {key} is not valid!')
        if field_value:
            filters = models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))
                                          for key, value in field_value.items()])
        else:
            filters = None
        return filters
