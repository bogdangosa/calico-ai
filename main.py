from fastapi import FastAPI
from src.api.routes import router as game_router

app = FastAPI()
app.include_router(game_router, prefix="/game", tags=["game"])
