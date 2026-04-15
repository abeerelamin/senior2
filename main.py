# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.core_routes import router as core_router
from routes.map_routes import router as map_router
from routes.change_routes import router as change_router
from routes.chat_routes import router as chat_router
from services.ee_runtime import init_ee


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_ee()


app.include_router(core_router)
app.include_router(map_router)
app.include_router(change_router)
app.include_router(chat_router)
