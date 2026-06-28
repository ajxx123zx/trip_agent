"""FastAPI 主应用 — LangGraph 版本"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import get_settings, validate_config, print_config
from .routes import trip, poi

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于 LangGraph StateGraph 的多智能体旅行规划系统",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trip.router)
app.include_router(poi.router)


@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print(f"[START] {settings.app_name} v{settings.app_version}")
    print("=" * 60)
    print_config()
    try:
        validate_config()
        print("[OK] Config validation passed")
    except ValueError as e:
        print(f"[FAIL] Config validation failed:\n{e}")
        raise
    print(f"Docs: http://{settings.host}:{settings.port}/docs")
    print("=" * 60)


@app.get("/")
async def root():
    return {"name": settings.app_name, "version": settings.app_version, "status": "running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": settings.app_name, "version": settings.app_version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host=settings.host, port=settings.port, reload=True)
