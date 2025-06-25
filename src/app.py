import logging

import uvicorn
from fastapi import FastAPI

from src.api import profile_endpoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="用户画像系统 API",
    description="基于LangChain和LangGraph的智能用户画像生成和分析系统",
    version="1.0.0",
    docs_url="/docs",
)

# 包含路由
app.include_router(profile_endpoints.router, prefix="/api/v1/profiles", tags=["用户画像"])



if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
