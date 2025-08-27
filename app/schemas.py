from bson import ObjectId
from pydantic import BaseModel, Field
from typing import List, Optional

class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, info):
        if isinstance(v, ObjectId):
            return str(v)
        
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        
        return str(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        # Generate JSON Schema for OpenAPI
        return {"type": "string", "example": "60d5ec49f9a7b3c3f8d0dfe2"}

class View(BaseModel):
    Id: int
    Url: str
    Status: str
    VideoUrl: str


class Camera(BaseModel):
    Id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    Source: str
    Roadway: Optional[str] # Can be optionally a null
    Direction: str
    Latitude: float
    Longitude: float
    Location: str
    Views: List[View]
    
    class Config:
        validate_by_name = True
        json_encoders = {ObjectId: str}
