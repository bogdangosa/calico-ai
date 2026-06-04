from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, game_code: str, websocket: WebSocket):
        await websocket.accept()
        if game_code not in self.active_connections:
            self.active_connections[game_code] = []
        self.active_connections[game_code].append(websocket)

    def disconnect(self, game_code: str, websocket: WebSocket):
        if game_code in self.active_connections:
            self.active_connections[game_code].remove(websocket)
            if not self.active_connections[game_code]:
                del self.active_connections[game_code]

    async def broadcast(self, game_code: str, message: dict):
        if game_code in self.active_connections:
            for connection in self.active_connections[game_code]:
                await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/websocket/game/{game_code}")
async def websocket_endpoint(websocket: WebSocket, game_code: str):
    await manager.connect(game_code, websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(game_code, websocket)
