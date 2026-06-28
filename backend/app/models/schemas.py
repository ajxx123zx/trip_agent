"""数据模型定义 — LangGraph 版本"""

from typing import List, Optional, Union, Any, TypedDict, Annotated
from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
import enum
from operator import add


# ============ 请求模型 ============

class TripRequest(BaseModel):
    """旅行规划请求"""
    city: str = Field(..., description="目的地城市", example="北京")
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", example="2025-06-01")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", example="2025-06-03")
    travel_days: int = Field(..., description="旅行天数", ge=1, le=30, example=3)
    transportation: str = Field(..., description="交通方式", example="公共交通")
    accommodation: str = Field(..., description="住宿偏好", example="经济型酒店")
    preferences: List[str] = Field(default=[], description="旅行偏好标签", example=["历史文化", "美食"])
    free_text_input: Optional[str] = Field(default="", description="额外要求")


# ============ 响应模型 ============

class Location(BaseModel):
    longitude: float
    latitude: float


class Attraction(BaseModel):
    name: str
    address: str
    location: Location
    visit_duration: int = Field(default=120, description="建议游览时间(分钟)")
    description: str = ""
    category: Optional[str] = "景点"
    rating: Optional[float] = None
    photos: Optional[List[str]] = Field(default_factory=list)
    poi_id: Optional[str] = ""
    image_url: Optional[str] = None
    ticket_price: int = 0


class Meal(BaseModel):
    type: str
    name: str
    address: Optional[str] = None
    location: Optional[Location] = None
    description: Optional[str] = None
    estimated_cost: int = 0


class Hotel(BaseModel):
    name: str
    address: str = ""
    location: Optional[Location] = None
    price_range: str = ""
    rating: str = ""
    distance: str = ""
    type: str = ""
    estimated_cost: int = 0


class DayPlan(BaseModel):
    date: str
    day_index: int
    description: str
    transportation: str
    accommodation: str
    hotel: Optional[Hotel] = None
    attractions: List[Attraction] = Field(default=[])
    meals: List[Meal] = Field(default=[])


class WeatherInfo(BaseModel):
    date: str
    day_weather: str = ""
    night_weather: str = ""
    day_temp: Union[int, str] = 0
    night_temp: Union[int, str] = 0
    wind_direction: str = ""
    wind_power: str = ""

    @field_validator('day_temp', 'night_temp', mode='before')
    @classmethod
    def parse_temperature(cls, v):
        if isinstance(v, str):
            v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
            try: return int(v)
            except ValueError: return 0
        return v


class Budget(BaseModel):
    total_attractions: int = 0
    total_hotels: int = 0
    total_meals: int = 0
    total_transportation: int = 0
    total: int = 0


class TripPlan(BaseModel):
    city: str
    start_date: str
    end_date: str
    days: List[DayPlan]
    weather_info: List[WeatherInfo] = Field(default=[])
    overall_suggestions: str = ""
    budget: Optional[Budget] = None


class TripPlanResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[TripPlan] = None


class POIInfo(BaseModel):
    """POI信息"""
    id: str = ""
    name: str = ""
    type: str = ""
    address: str = ""
    location: Location = Field(default_factory=lambda: Location(longitude=0, latitude=0))
    tel: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    message: str = ""
    error_code: Optional[str] = None


# ============ LangGraph 状态定义 ============

class AgentStage(str, enum.Enum):
    INIT = "init"
    GATHERING = "gathering"
    MERGING = "merging"
    PLANNING = "planning"
    REVIEW = "review"
    DONE = "done"
    ERROR = "error"


class AgentErrorInfo(BaseModel):
    agent_name: str
    error_type: str = "api_error"
    message: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TripAgentState(TypedDict, total=False):
    request: TripRequest
    attraction_output: str
    weather_output: str
    hotel_output: str
    plan_json: Optional[str]
    trip_plan: Optional[TripPlan]
    stage: str
    errors: Annotated[List[AgentErrorInfo], add]
    retry_count: int
    human_approved: bool
    human_feedback: str
