import re
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, computed_field
from datetime import datetime


class Product(BaseModel):
    """Product data model with validation"""
    id: int = Field(...)
    name: str = Field(..., min_length=1, max_length=200)

    description: Optional[str] = Field(None, max_length=100000)
    material: Optional[str] = Field(None, min_length=1, max_length=200)

    current_price: float = Field(..., ge=0)
    off_percent: Optional[float] = Field(None, ge=0)
    currency: str = Field(min_length=1, max_length=200)

    images: List[str]

    brand_id: int | None = Field(None, ge=0)
    brand_name: str | None = Field(None, min_length=1, max_length=200)

    code: str = Field(..., min_length=1, max_length=200)

    category_id: int | None = Field(None, ge=0)
    category_name: str | None = Field(None, min_length=1, max_length=200)

    gender_id: int | None = Field(None, ge=0)
    gender_name: str | None = Field(None, min_length=1, max_length=200)

    shop_id: int | None = Field(None, ge=0)
    shop_name: str | None = Field(None, min_length=1, max_length=200)

    link: str = Field(..., min_length=1, max_length=200)

    status: str | None = Field(None, min_length=1, max_length=200)

    colors: Optional[List[str]] = Field([], min_length=0, max_length=200)
    sizes: Optional[List[str]] = Field([], min_length=0, max_length=200)

    region: Optional[str] = Field(None, min_length=1, max_length=200)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field(return_type=str)
    @property
    def uuid(self):
        """Generate a deterministic UUID4 based on the product ID"""
        generated_uuid = str(uuid.UUID(int=self.id, version=4))

        return generated_uuid

    @field_validator('colors')
    def validate_colors(cls, v):
        if v is None:
            return []
        for color in v:
            if not re.match(r'#[0-9A-F]{5}', color.upper()):
                raise ValueError(f'Invalid color: {color}')
        return [color.upper() for color in v]

    def to_response_obj(self):
        return {
            'id': self.uuid,
            'name': self.name,
            'price': self.current_price,
            'currency': self.currency,
            'link': self.link,
            'images': self.images,
        }

    def to_vector_record(self, embedding: List[float]) -> Dict[str, Any]:
        """Convert product to vector record format"""
        payload = self.dict().copy()
        payload.pop('uuid')

        payload = {k: v for k, v in payload.items()
                   if v is not None}

        return {
            'id': self.uuid,
            'vector': embedding,
            'payload': payload
        }
