from fastapi import APIRouter, HTTPException, status

from typing import List

from ..schemas import Camera
from ..database import db

router = APIRouter(
    prefix="/cameras",
    tags=["Cameras"]
)

@router.get("/", status_code=status.HTTP_200_OK, response_model=List[Camera])
def get_cameras():
    
    response = db["GA511"].find()
    if not response:
        raise HTTPException(status_code=status.HTTP_404_BAD_REQUEST,
                             detail="No cameras found")
        
    return list(response)

@router.get("/{id}", status_code=status.HTTP_200_OK, response_model=Camera)
def get_camera_by_id(id: int):
    response = db["GA511"].find_one({"Id": id})
    
    if not response:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                             detail=f"camera id: {id} was not found")
        
        
    return response
    
@router.get("/locations/{location_id}", status_code=status.HTTP_200_OK, response_model=Camera)
def get_camera_by_location_id(location_id: str):
    response = db["GA511"].find_one({"Location": {"$regex": location_id}})
    
    if not response:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                             detail=f"camera id: {location_id} was not found")
        
    
    return response
