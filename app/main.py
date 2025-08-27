from fastapi import FastAPI
from .routers import camera

app = FastAPI()
app.include_router(camera.router)

@app.get("/")
def root():
    return {"Hello": "World"}