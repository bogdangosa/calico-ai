from fastapi import FastAPI
from src.api.routes import router as game_router
from src.api.sockets import router as socket_router
from src.repository.database.connection import dispose_engine

app = FastAPI()
app.include_router(game_router, prefix="/game", tags=["game"])
app.include_router(socket_router, tags=["sockets"])

@app.on_event("shutdown")
async def shutdown_event():
    await dispose_engine()
