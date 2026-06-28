"""POI + 景点图片 API 路由"""

from fastapi import APIRouter, HTTPException
from ...services.amap_service import get_amap_service
from ...services.unsplash_service import get_unsplash_service

router = APIRouter(prefix="/api/poi", tags=["POI & 图片"])


@router.get("/photo", summary="获取景点图片",
            description="根据景点名称从 Unsplash 获取高质量图片")
async def get_attraction_photo(name: str):
    try:
        unsplash_service = get_unsplash_service()
        photo_url = unsplash_service.get_photo_url(f"{name} China landmark")
        if not photo_url:
            photo_url = unsplash_service.get_photo_url(name)
        return {
            "success": True,
            "message": "获取图片成功",
            "data": {"name": name, "photo_url": photo_url}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")
