"""
LangGraph StateGraph 多智能体旅行规划器
========================================
架构: 3 Agent 并行搜索 → 汇聚 → 规划 → 兜底
容错: 超时/API异常 → 条件边降级 → 语义完整兜底计划

图结构:
  START ──→ attraction_agent ──→ merge ──→ planner ──→ END
         ├─→ weather_agent    ──→     │        │
         └─→ hotel_agent      ──→     │   ┌────┘
                                      │   │ (条件边)
                                      │   ▼
                                      │ fallback ──→ END
"""

import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ..services.llm_service import get_llm
from ..services.amap_service import get_amap_tools
from ..models.schemas import (
    TripRequest, TripPlan, DayPlan, Attraction, Meal,
    WeatherInfo, Location, Hotel, Budget,
    TripAgentState, AgentErrorInfo
)

# ============ 超时配置 ============
TIMEOUT_ATTRACTION = 60.0   # 景点搜索
TIMEOUT_WEATHER = 45.0      # 天气查询
TIMEOUT_HOTEL = 60.0        # 酒店搜索
TIMEOUT_PLANNER = 90.0      # 行程规划
TOTAL_TIMEOUT = 180.0       # 整体超时
MAX_RETRIES = 1             # 单个Agent最大重试次数


# ============ 辅助函数 ============

def _parse_json_safe(text: str) -> dict:
    """三级降级 JSON 解析: ```json → ``` → 裸{}"""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return json.loads(text[start:end].strip())
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return json.loads(text[start:end].strip())
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    raise ValueError("未找到 JSON 数据")


def _normalize_trip_json(data: dict) -> dict:
    """字段映射兜底：将LLM输出的非标准字段名映射到Pydantic模型期望字段名"""

    # 外层字段映射: destination→city, dates→拆分, daily_plans/itinerary→days
    field_map = {
        "destination": "city",
        "trip_destination": "city",
        "daily_plans": "days",
        "itinerary": "days",
        "plan": "days",
        "weather": "weather_info",
        "forecast": "weather_info",
        "suggestions": "overall_suggestions",
        "tips": "overall_suggestions",
    }

    for old_key, new_key in field_map.items():
        if old_key in data and new_key not in data:
            data[new_key] = data.pop(old_key)

    # 处理 dates → start_date / end_date
    if "dates" in data and ("start_date" not in data or "end_date" not in data):
        dates_val = str(data.pop("dates"))
        if "~" in dates_val:
            parts = dates_val.split("~")
            if "start_date" not in data:
                data["start_date"] = parts[0].strip()
            if "end_date" not in data:
                data["end_date"] = parts[1].strip() if len(parts) > 1 else parts[0].strip()

    # 处理每个 day 内部字段映射
    if "days" in data and isinstance(data["days"], list):
        day_field_map = {
            "attraction": "attractions",
            "sights": "attractions",
            "spots": "attractions",
            "meal": "meals",
            "dining": "meals",
            "restaurants": "meals",
            "stay": "hotel",
            "lodging": "hotel",
            "accommodation_detail": "hotel",
        }
        for day in data["days"]:
            if not isinstance(day, dict):
                continue
            for old_key, new_key in day_field_map.items():
                if old_key in day and new_key not in day:
                    day[new_key] = day.pop(old_key)

            # meal → meals 数组包装
            if "meal" in day:
                day["meals"] = [day.pop("meal")] if "meals" not in day else day.get("meals", []) + [day.pop("meal")]

    # 确保 weather_info 是数组
    if "weather_info" in data and isinstance(data["weather_info"], dict):
        data["weather_info"] = [data["weather_info"]]

    return data


def _truncate_text(text: str, max_chars: int) -> str:
    """智能截断(首尾保留)防 Token 溢出"""
    if len(text) <= max_chars:
        return text
    head_len = max_chars * 2 // 3
    tail_len = max_chars // 3
    return (text[:head_len] +
            f"\n\n...(省略 {len(text) - max_chars} 字符)...\n\n" +
            text[-tail_len:])


def _create_fallback_plan(request: TripRequest) -> TripPlan:
    """兜底计划 — 任何异常路径都返回语义完整计划，绝不500"""
    start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    days = []
    for i in range(request.travel_days):
        current_date = start_date + timedelta(days=i)
        day_plan = DayPlan(
            date=current_date.strftime("%Y-%m-%d"),
            day_index=i,
            description=f"第{i+1}天行程（降级方案）",
            transportation=request.transportation,
            accommodation=request.accommodation,
            attractions=[
                Attraction(
                    name=f"{request.city}推荐景点{j+1}",
                    address=f"{request.city}市",
                    location=Location(longitude=116.4+i*0.01+j*0.005, latitude=39.9+i*0.01+j*0.005),
                    visit_duration=120,
                    description=f"{request.city}的热门景点",
                    category="景点"
                ) for j in range(2)
            ],
            meals=[
                Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
            ]
        )
        days.append(day_plan)

    return TripPlan(
        city=request.city, start_date=request.start_date, end_date=request.end_date,
        days=days, weather_info=[],
        overall_suggestions=f"部分服务不可用，已生成{request.city}{request.travel_days}日游降级计划。刷新可获取完整AI规划。"
    )


# ============ LangGraph 节点函数 ============

async def _attraction_node(state: TripAgentState) -> dict:
    """景点搜索节点 — LLM + amap_text_search 工具"""
    request = state["request"]
    llm = get_llm()
    tools = get_amap_tools()
    llm_with_tools = llm.bind_tools(tools)

    keywords = request.preferences[0] if request.preferences else "热门景点"
    query = f"请用 amap_text_search 搜索 {request.city} 的 {keywords}，返回10个结果，列出名称、地址、坐标、类别、评分、门票。"

    try:
        messages = [HumanMessage(content=query)]
        for _ in range(3):
            response = await asyncio.wait_for(
                asyncio.to_thread(llm_with_tools.invoke, messages),
                timeout=TIMEOUT_ATTRACTION
            )
            messages.append(response)  # 先追加AI响应，确保后续ToolMessage合法
            if not response.tool_calls:
                return {"attraction_output": response.content}
            for tc in response.tool_calls:
                tool_fn = next((t for t in tools if t.name == tc.get("name", "")), None)
                if tool_fn:
                    messages.append(ToolMessage(content=str(tool_fn.invoke(tc["args"])), tool_call_id=tc.get("id", "")))
        return {"attraction_output": response.content}
    except asyncio.TimeoutError:
        return {
            "attraction_output": f"{request.city} 热门景点（搜索超时）",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="attraction", error_type="timeout",
                               message=f"超时({TIMEOUT_ATTRACTION}s)")
            ]
        }
    except Exception as e:
        return {
            "attraction_output": f"{request.city} 热门景点（服务异常）",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="attraction", error_type="api_error", message=str(e))
            ]
        }


async def _weather_node(state: TripAgentState) -> dict:
    """天气查询节点 — LLM + amap_weather 工具"""
    request = state["request"]
    llm = get_llm()
    tools = get_amap_tools()
    weather_tools = [t for t in tools if t.name == "amap_weather"]
    llm_with_tools = llm.bind_tools(weather_tools)

    query = f"请用 amap_weather 查询 {request.city} 未来{request.travel_days}天天气。"

    try:
        messages = [HumanMessage(content=query)]
        response = await asyncio.wait_for(
            asyncio.to_thread(llm_with_tools.invoke, messages),
            timeout=TIMEOUT_WEATHER
        )
        messages.append(response)  # 追加AI响应
        if response.tool_calls and weather_tools:
            for tc in response.tool_calls:
                messages.append(ToolMessage(content=str(weather_tools[0].invoke(tc["args"])),
                                            tool_call_id=tc.get("id", "")))
            messages.append(HumanMessage(content="将天气整理为可读中文格式。"))
            response = await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, messages),
                timeout=TIMEOUT_WEATHER
            )
        return {"weather_output": response.content}
    except asyncio.TimeoutError:
        return {
            "weather_output": f"{request.city} 天气（查询超时）",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="weather", error_type="timeout",
                               message=f"超时({TIMEOUT_WEATHER}s)")
            ]
        }
    except Exception as e:
        return {
            "weather_output": f"{request.city} 天气（服务异常）",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="weather", error_type="api_error", message=str(e))
            ]
        }


async def _hotel_node(state: TripAgentState) -> dict:
    """酒店推荐节点 — LLM + amap_text_search 工具"""
    request = state["request"]
    llm = get_llm()
    tools = get_amap_tools()
    llm_with_tools = llm.bind_tools(tools)

    query = f"请用 amap_text_search 搜索 {request.city} 的 {request.accommodation}，返回10个结果，列出名称、地址、坐标、类型、评分、价格。"

    try:
        messages = [HumanMessage(content=query)]
        for _ in range(3):
            response = await asyncio.wait_for(
                asyncio.to_thread(llm_with_tools.invoke, messages),
                timeout=TIMEOUT_HOTEL
            )
            messages.append(response)  # 先追加AI响应，确保后续ToolMessage合法
            if not response.tool_calls:
                return {"hotel_output": response.content}
            for tc in response.tool_calls:
                tool_fn = next((t for t in tools if t.name == tc.get("name", "")), None)
                if tool_fn:
                    messages.append(ToolMessage(content=str(tool_fn.invoke(tc["args"])),
                                                tool_call_id=tc.get("id", "")))
        return {"hotel_output": response.content}
    except asyncio.TimeoutError:
        return {
            "hotel_output": f"{request.city} 酒店推荐（搜索超时）",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="hotel", error_type="timeout",
                               message=f"超时({TIMEOUT_HOTEL}s)")
            ]
        }
    except Exception as e:
        return {
            "hotel_output": f"{request.city} 酒店推荐（服务异常）",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="hotel", error_type="api_error", message=str(e))
            ]
        }


def _merge_node(state: TripAgentState) -> dict:
    """
    汇聚节点 — 检查三路结果完整性，决策重试/降级/继续
    
    路由规则:
    - 三路全失败 → stage=fallback
    - 部分失败 + retry < MAX_RETRIES → retry_count+1
    - 部分失败 + retry = MAX_RETRIES → 降级文本填充 → stage=planning
    - 全成功 → stage=planning
    """
    request = state["request"]
    retry_count = state.get("retry_count", 0)

    has_a = bool(state.get("attraction_output", "").strip())
    has_w = bool(state.get("weather_output", "").strip())
    has_h = bool(state.get("hotel_output", "").strip())

    # 全部失败
    if not has_a and not has_w and not has_h:
        return {"stage": "fallback"}

    # 部分失败 + 可重试
    failed = (0 if has_a else 1) + (0 if has_w else 1) + (0 if has_h else 1)
    if failed > 0 and retry_count < MAX_RETRIES:
        return {"retry_count": retry_count + 1}

    # 降级继续
    updates = {"stage": "planning"}
    if not has_a: updates["attraction_output"] = f"{request.city} 热门景点（自动降级补充）"
    if not has_w: updates["weather_output"] = f"{request.city} 天气（自动降级补充）"
    if not has_h: updates["hotel_output"] = f"{request.city} 酒店推荐（自动降级补充）"
    return updates


def _route_after_merge(state: TripAgentState) -> Literal["planner", "fallback"]:
    """汇聚后的条件路由"""
    return "fallback" if state.get("stage") == "fallback" else "planner"


async def _planner_node(state: TripAgentState) -> dict:
    """行程规划节点 — 汇总信息，LLM 生成 JSON 计划"""
    request = state["request"]
    llm = get_llm()

    attractions = _truncate_text(state.get("attraction_output", ""), 2000)
    weather = _truncate_text(state.get("weather_output", ""), 1500)
    hotels = _truncate_text(state.get("hotel_output", ""), 1500)

    # 构建JSON Schema示例，确保LLM输出正确字段名
    json_schema_template = """{
  "city": "城市名",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "当日行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿偏好",
      "attractions": [
        {
          "name": "景点名称",
          "address": "景点地址",
          "location": {"longitude": 116.40, "latitude": 39.90},
          "visit_duration": 120,
          "description": "景点描述",
          "category": "景点",
          "rating": 4.5,
          "ticket_price": 0
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐名称", "description": "推荐理由"},
        {"type": "lunch", "name": "午餐名称", "description": "推荐理由"},
        {"type": "dinner", "name": "晚餐名称", "description": "推荐理由"}
      ],
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.40, "latitude": 39.90},
        "price_range": "200-400元",
        "rating": "4.5",
        "distance": "距离市中心3km",
        "type": "经济型酒店",
        "estimated_cost": 300
      }
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 30,
      "night_temp": 20,
      "wind_direction": "南",
      "wind_power": "2-3级"
    }
  ],
  "overall_suggestions": "整体建议",
  "budget": {
    "total_attractions": 0,
    "total_hotels": 0,
    "total_meals": 0,
    "total_transportation": 0,
    "total": 0
  }
}"""

    prompt = "\n".join([
        f"请为 {request.city} 生成 {request.travel_days} 天旅行计划。",
        f"日期: {request.start_date}~{request.end_date}",
        f"交通: {request.transportation} | 住宿类型: {request.accommodation}",
        f"旅行偏好: {', '.join(request.preferences) if request.preferences else '无'}",
        *([f"额外要求: {request.free_text_input}"] if request.free_text_input else []),
        "",
        "=== 可用景点 ===", attractions,
        "=== 天气预报 ===", weather,
        "=== 酒店选项 ===", hotels,
        "",
        "【重要】你必须严格按以下JSON Schema返回（字段名必须精确匹配，city/start_date/end_date/days）：",
        json_schema_template,
        "",
        "要求：1) 每天安排2-3个景点 2) 景点经纬度使用真实坐标 3) 餐饮推荐当地特色 4) 预算合理估算",
        "只返回JSON，不要```json```标记，不要任何解释文字。"
    ])

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)]),
            timeout=TIMEOUT_PLANNER
        )
        plan_text = response.content if hasattr(response, 'content') else str(response)
        data = _parse_json_safe(plan_text)
        data = _normalize_trip_json(data)  # 字段映射兜底
        return {"plan_json": plan_text, "trip_plan": TripPlan(**data), "stage": "done"}
    except asyncio.TimeoutError:
        return {
            "stage": "fallback",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="planner", error_type="timeout",
                               message=f"超时({TIMEOUT_PLANNER}s)")
            ]
        }
    except Exception as e:
        return {
            "stage": "fallback",
            "errors": state.get("errors", []) + [
                AgentErrorInfo(agent_name="planner", error_type="api_error", message=str(e))
            ]
        }


def _fallback_node(state: TripAgentState) -> dict:
    """兜底节点 — 生成语义完整的降级计划"""
    plan = _create_fallback_plan(state["request"])
    plan.overall_suggestions = (
        f"部分服务不可用，已为{state['request'].city}生成基础行程。刷新可获取完整AI规划。"
    )
    return {"trip_plan": plan, "stage": "done"}


# ============ LangGraph 图构建 ============

class LangGraphTripPlanner:
    """
    LangGraph StateGraph 多智能体旅行规划器
    
    图结构 (6节点 + 条件边):
        START → attraction_agent / weather_agent / hotel_agent (三路并行)
              → merge (汇聚 + 容错决策)
              → planner (生成) / fallback (兜底)
              → END
    
    特性:
    - 条件边自动降级：超时/异常不阻塞全局
    - 独立超时控制：45-90s/Agent
    - SSE astream_events 流式输出
    - 三级JSON解析降级
    """

    def __init__(self):
        self.llm = get_llm()
        self.amap_tools = get_amap_tools()
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        print(f"[OK] LangGraph 规划器 v3.0")
        print(f"   工具: {[t.name for t in self.amap_tools]}")
        print(f"   超时: 景点{int(TIMEOUT_ATTRACTION)}s 天气{int(TIMEOUT_WEATHER)}s "
              f"酒店{int(TIMEOUT_HOTEL)}s 规划{int(TIMEOUT_PLANNER)}s")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(TripAgentState)

        workflow.add_node("attraction_agent", _attraction_node)
        workflow.add_node("weather_agent", _weather_node)
        workflow.add_node("hotel_agent", _hotel_node)
        workflow.add_node("merge", _merge_node)
        workflow.add_node("planner", _planner_node)
        workflow.add_node("fallback", _fallback_node)

        # START → 三路并行
        workflow.add_edge(START, "attraction_agent")
        workflow.add_edge(START, "weather_agent")
        workflow.add_edge(START, "hotel_agent")

        # 三路 → 汇聚
        workflow.add_edge("attraction_agent", "merge")
        workflow.add_edge("weather_agent", "merge")
        workflow.add_edge("hotel_agent", "merge")

        # 汇聚 → 条件路由
        workflow.add_conditional_edges(
            "merge", _route_after_merge,
            {"planner": "planner", "fallback": "fallback"}
        )

        workflow.add_edge("planner", END)
        workflow.add_edge("fallback", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        """标准模式 — 生成旅行计划"""
        initial_state: TripAgentState = {
            "request": request,
            "attraction_output": "", "weather_output": "", "hotel_output": "",
            "plan_json": None, "trip_plan": None,
            "stage": "init", "errors": [], "retry_count": 0,
            "human_approved": False, "human_feedback": ""
        }
        config = RunnableConfig(
            configurable={"thread_id": f"trip_{request.city}_{int(time.time())}"}
        )

        try:
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config),
                timeout=TOTAL_TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] 总超时({TOTAL_TIMEOUT}s)，返回兜底")
            return _create_fallback_plan(request)

        result = final_state.get("trip_plan")
        if result is None:
            return _create_fallback_plan(request)

        errors = final_state.get("errors", [])
        if errors:
            print(f"[INFO] 规划完成，{len(errors)}个非致命错误:")
            for e in errors:
                print(f"  - {e.agent_name}: {e.message}")

        return result

    async def plan_trip_streaming(self, request: TripRequest):
        """SSE 流式模式 — 逐步推送进度事件"""
        initial_state: TripAgentState = {
            "request": request,
            "attraction_output": "", "weather_output": "", "hotel_output": "",
            "plan_json": None, "trip_plan": None,
            "stage": "init", "errors": [], "retry_count": 0,
            "human_approved": False, "human_feedback": ""
        }
        config = RunnableConfig(
            configurable={"thread_id": f"trip_{request.city}_{int(time.time())}"}
        )

        node_names = {
            "attraction_agent": "正在搜索景点...",
            "weather_agent": "正在查询天气...",
            "hotel_agent": "正在搜索酒店...",
            "merge": "正在汇总信息...",
            "planner": "正在生成行程计划...",
            "fallback": "正在生成备用计划...",
            "__start__": "开始规划...",
        }

        try:
            async for event in self.graph.astream_events(initial_state, config, version="v2"):
                kind = event.get("event")
                name = event.get("name", "")

                if kind == "on_chain_start" and name in node_names:
                    yield {"stage": "agent_start", "agent": name, "message": node_names[name]}

                elif kind == "on_tool_start":
                    yield {"stage": "tool_call", "tool": event.get("name", "")}

                elif kind == "on_chain_end" and name == "merge":
                    output = event.get("data", {}).get("output", {})
                    yield {
                        "stage": "merge_complete",
                        "has_attraction": bool(output.get("attraction_output")),
                        "has_weather": bool(output.get("weather_output")),
                        "has_hotel": bool(output.get("hotel_output")),
                    }

                elif kind == "on_chain_end" and name == "planner":
                    output = event.get("data", {}).get("output", {})
                    if output.get("trip_plan"):
                        yield {"stage": "plan_complete", "trip_plan": output["trip_plan"].model_dump()}

                elif kind == "on_chain_end" and name == "fallback":
                    output = event.get("data", {}).get("output", {})
                    if output.get("trip_plan"):
                        yield {
                            "stage": "fallback",
                            "message": "部分服务不可用，已生成备用计划",
                            "trip_plan": output["trip_plan"].model_dump()
                        }

            yield {"stage": "done", "message": "规划完成"}

        except Exception as e:
            yield {
                "stage": "error",
                "message": str(e),
                "trip_plan": _create_fallback_plan(request).model_dump()
            }


# ============ 全局单例 ============

_langgraph_planner: Optional[LangGraphTripPlanner] = None


def get_langgraph_planner() -> LangGraphTripPlanner:
    global _langgraph_planner
    if _langgraph_planner is None:
        _langgraph_planner = LangGraphTripPlanner()
    return _langgraph_planner
