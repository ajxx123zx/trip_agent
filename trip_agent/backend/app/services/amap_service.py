"""高德地图服务 - 直接HTTP API调用 + LangChain工具"""

import json
import re
from typing import List, Dict, Any, Optional
import httpx
from langchain_core.tools import tool
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo

# ═══════════════════════════════════════════════════════════════
# 底层 HTTP API 调用
# ═══════════════════════════════════════════════════════════════

_http_client: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    """获取HTTP客户端(连接池复用)"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=30.0)
    return _http_client


def _get_api_key() -> str:
    """获取高德API Key"""
    return get_settings().amap_api_key


def _call_amap_api(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """调用高德地图API"""
    params["key"] = _get_api_key()
    url = f"https://restapi.amap.com/v3{endpoint}"
    resp = _get_client().get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if int(data.get("status", 0)) != 1:
        raise RuntimeError(f"高德API错误: {data.get('info', '未知错误')}")
    return data


# ═══════════════════════════════════════════════════════════════
# LangChain 工具函数 (供Agent使用)
# ═══════════════════════════════════════════════════════════════

@tool
def amap_text_search(keywords: str, city: str, citylimit: bool = True) -> str:
    """
    高德地图POI关键词搜索。根据关键词和城市搜索兴趣点(景点、酒店、餐厅等)。
    参数:
    - keywords: 搜索关键词，如"历史文化"、"公园"、"酒店"
    - city: 城市名称，如"北京"、"上海"
    - citylimit: 是否限制在城市范围内，默认True
    返回JSON格式的搜索结果，包含名称、地址、经纬度坐标等信息。
    """
    try:
        data = _call_amap_api("/place/text", {
            "keywords": keywords,
            "city": city,
            "citylimit": str(citylimit).lower(),
            "offset": 10,
            "extensions": "all"
        })
        pois = data.get("pois", [])
        if not pois:
            return f"在{city}未找到与'{keywords}'相关的POI"

        result_parts = []
        for poi in pois[:10]:
            location = poi.get("location", "")
            lon, lat = "", ""
            if location and "," in location:
                lon, lat = location.split(",")

            parts = [
                f"名称: {poi.get('name', '未知')}",
                f"地址: {poi.get('address', '未知')}",
                f"类别: {poi.get('type', '未知')}",
                f"经纬度: ({lon}, {lat})",
            ]
            if poi.get("tel"):
                parts.append(f"电话: {poi['tel']}")
            if poi.get("biz_ext", {}).get("rating"):
                parts.append(f"评分: {poi['biz_ext']['rating']}")
            if poi.get("biz_ext", {}).get("cost"):
                parts.append(f"人均: {poi['biz_ext']['cost']}元")

            result_parts.append("\n".join(parts))

        return "\n\n---\n\n".join(result_parts)
    except Exception as e:
        return f"POI搜索失败: {str(e)}"


@tool
def amap_weather(city: str) -> str:
    """
    查询指定城市的天气预报。
    参数:
    - city: 城市名称，如"北京"、"上海"
    返回JSON格式的天气信息，包含日期、天气状况、温度、风力等。
    """
    try:
        data = _call_amap_api("/weather/weatherInfo", {
            "city": city,
            "extensions": "all"
        })
        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未找到{city}的天气信息"

        result_parts = []
        for forecast in forecasts:
            result_parts.append(f"城市: {forecast.get('city', city)}")
            for cast in forecast.get("casts", [])[:7]:
                parts = [
                    f"日期: {cast.get('date', '未知')}",
                    f"白天: {cast.get('dayweather', '未知')} {cast.get('daytemp', '?')}°C",
                    f"夜间: {cast.get('nightweather', '未知')} {cast.get('nighttemp', '?')}°C",
                    f"风力: {cast.get('daywind', '未知')} {cast.get('daypower', '?')}级",
                ]
                result_parts.append(" | ".join(parts))

        return "\n".join(result_parts)
    except Exception as e:
        return f"天气查询失败: {str(e)}"


@tool
def amap_geo(address: str, city: str = "") -> str:
    """
    地理编码: 将地址转换为经纬度坐标。
    参数:
    - address: 详细地址
    - city: 所在城市(可选)
    返回JSON格式的经纬度坐标。
    """
    try:
        params = {"address": address}
        if city:
            params["city"] = city
        data = _call_amap_api("/geocode/geo", params)
        geocodes = data.get("geocodes", [])
        if not geocodes:
            return f"未找到地址'{address}'的坐标"
        geo = geocodes[0]
        return f"地址: {geo.get('formatted_address', address)}, 坐标: ({geo.get('location', '未知')})"
    except Exception as e:
        return f"地理编码失败: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# 导出工具列表 (供Agent使用)
# ═══════════════════════════════════════════════════════════════

AMAP_TOOLS = [amap_text_search, amap_weather, amap_geo]


def get_amap_tools() -> List:
    """获取高德地图LangChain工具列表"""
    return AMAP_TOOLS


# ═══════════════════════════════════════════════════════════════
# AmapService类 (供REST API端点使用)
# ═══════════════════════════════════════════════════════════════

class AmapService:
    """高德地图服务封装类(同步,供API路由使用)"""

    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """搜索POI"""
        try:
            data = _call_amap_api("/place/text", {
                "keywords": keywords,
                "city": city,
                "citylimit": str(citylimit).lower(),
                "offset": 10,
                "extensions": "all"
            })
            pois = data.get("pois", [])
            results = []
            for poi in pois:
                location = poi.get("location", "")
                lon, lat = 0.0, 0.0
                if location and "," in location:
                    parts = location.split(",")
                    lon, lat = float(parts[0]), float(parts[1])
                results.append(POIInfo(
                    id=poi.get("id", ""),
                    name=poi.get("name", ""),
                    type=poi.get("type", ""),
                    address=poi.get("address", ""),
                    location=Location(longitude=lon, latitude=lat),
                    tel=poi.get("tel")
                ))
            return results
        except Exception as e:
            print(f"[ERROR] POI搜索失败: {e}")
            return []

    def get_weather(self, city: str) -> List[WeatherInfo]:
        """查询天气"""
        try:
            data = _call_amap_api("/weather/weatherInfo", {
                "city": city,
                "extensions": "all"
            })
            forecasts = data.get("forecasts", [])
            results = []
            for forecast in forecasts:
                for cast in forecast.get("casts", []):
                    results.append(WeatherInfo(
                        date=cast.get("date", ""),
                        day_weather=cast.get("dayweather", ""),
                        night_weather=cast.get("nightweather", ""),
                        day_temp=cast.get("daytemp", 0),
                        night_temp=cast.get("nighttemp", 0),
                        wind_direction=cast.get("daywind", ""),
                        wind_power=cast.get("daypower", "")
                    ))
            return results
        except Exception as e:
            print(f"[ERROR] 天气查询失败: {e}")
            return []

    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Dict[str, Any]:
        """规划路线"""
        try:
            endpoint_map = {
                "walking": "/direction/walking",
                "driving": "/direction/driving",
                "transit": "/direction/transit/integrated"
            }
            endpoint = endpoint_map.get(route_type, "/direction/walking")
            params = {
                "origin": origin_address,
                "destination": destination_address
            }
            if origin_city:
                params["city"] = origin_city
            if destination_city:
                params["destination_city"] = destination_city

            data = _call_amap_api(endpoint, params)

            route_info = {"distance": 0.0, "duration": 0, "route_type": route_type, "description": ""}
            route = data.get("route", {})
            if route_type == "walking" and route.get("paths"):
                path = route["paths"][0]
                route_info["distance"] = float(path.get("distance", 0))
                route_info["duration"] = int(path.get("duration", 0))
                route_info["description"] = f"步行约{int(path.get('distance', 0))}米,预计{int(path.get('duration', 0)) // 60}分钟"
            elif route_type == "driving" and route.get("paths"):
                path = route["paths"][0]
                route_info["distance"] = float(path.get("distance", 0))
                route_info["duration"] = int(path.get("duration", 0))
                route_info["description"] = f"驾车约{int(path.get('distance', 0))}米,预计{int(path.get('duration', 0)) // 60}分钟"
            elif route.get("transits"):
                transit = route["transits"][0]
                route_info["distance"] = float(transit.get("distance", 0))
                route_info["duration"] = int(transit.get("duration", 0))
                route_info["description"] = f"公交/地铁,预计{int(transit.get('duration', 0)) // 60}分钟"

            return route_info
        except Exception as e:
            print(f"[ERROR] 路线规划失败: {e}")
            return {}

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """地理编码(地址转坐标)"""
        try:
            params = {"address": address}
            if city:
                params["city"] = city
            data = _call_amap_api("/geocode/geo", params)
            geocodes = data.get("geocodes", [])
            if geocodes:
                location = geocodes[0].get("location", "0,0")
                parts = location.split(",")
                return Location(longitude=float(parts[0]), latitude=float(parts[1]))
            return None
        except Exception as e:
            print(f"[ERROR] 地理编码失败: {e}")
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """获取POI详情"""
        try:
            data = _call_amap_api("/place/detail", {"id": poi_id})
            return data
        except Exception as e:
            print(f"[ERROR] 获取POI详情失败: {e}")
            return {}


# 全局服务实例
_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapService()
    return _amap_service
