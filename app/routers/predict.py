from fastapi import APIRouter, HTTPException, status

router = APIRouter(
    prefix="inference",
    tags=["Inference"]
)

@router.post("/", status_code=status.HTTP_202_ACCEPTED)
def predict():
    pass