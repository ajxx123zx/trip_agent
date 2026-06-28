"""配置管理 — LangGraph 版本"""

import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    app_name: str = "LangGraph 智能旅行助手"
    app_version: str = "3.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    amap_api_key: str = ""
    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(',')]


settings = Settings()


def get_settings() -> Settings:
    return settings


def validate_config():
    errors = []
    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY 未配置")
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not llm_key:
        errors.append("LLM_API_KEY 或 OPENAI_API_KEY 未配置")
    if errors:
        raise ValueError("配置错误:\n" + "\n".join(f"  - {e}" for e in errors))
    return True


def print_config():
    """打印当前配置(隐藏敏感信息)"""
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    llm_base = os.getenv("LLM_BASE_URL") or settings.openai_base_url
    llm_model = os.getenv("LLM_MODEL_ID") or settings.openai_model
    print(f"LLM API Key: {'已配置' if llm_key else '未配置'}")
    print(f"LLM Base URL: {llm_base}")
    print(f"LLM Model: {llm_model}")
