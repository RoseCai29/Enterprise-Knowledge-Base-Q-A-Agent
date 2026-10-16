"""API 鉴权：配置 API_KEY 后要求请求携带 X-API-Key 头，否则拒绝。

若未配置 API_KEY（config.api_key 为空），则放行 —— 便于本地调试/仅内网暴露。
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from config import config


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not config.api_key:
        return  # 未开启鉴权
    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key",
        )