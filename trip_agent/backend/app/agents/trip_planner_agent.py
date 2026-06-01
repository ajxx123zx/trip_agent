"""多智能体旅行规划系统 - 基于LangChain框架"""

import json
import asyncio
from datetime import datetime, timedelta
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from ..services.llm_service import get_llm
from ..services.amap_service import get_amap_tools
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel

# 各Agent超时配置（秒）
AGENT_TIMEOUT_ATTRACTION = 60.0
AGENT_TIMEOUT_WEATHER = 45.0
AGENT_TIMEOUT_HOTEL = 60.0
AGENT_TIMEOUT_PLANNER = 120.0

# ============ Agent系统提示词 ============

ATTRACTION_SYSTEM_PROMPT = """你是景点搜索专家。你的任务是根据用户需求搜索合适的景点。

工作流程:
1. 分析用户想去哪个城市，偏好什么类型的景点
2. 使用 amap_text_search 工具搜索景点
3. 整理搜索结果，列出每个景点的名称、地址、坐标

注意: 搜索时使用具体的偏好关键词(如"历史文化"、"自然风光"、"美食")，而非笼统的"景点"。
每次只搜索一种类型的景点，返回最相关的10个结果。
以中文格式返回每个景点的: 名称、地址、经纬度坐标、类别、评分(如有)、门票价格(如有)。
"""

WEATHER_SYSTEM_PROMPT = """你是天气查询专家。你的任务是指查询指定城市的天气信息。

工作流程:
1. 使用 amap_weather 工具查询目标城市的天气
2. 以中文格式整理返回天气信息，包含每天的日期、白天天气、夜间天气、白天温度、夜间温度、风向、风力。
"""

HOTEL_SYSTEM_PROMPT = """你是酒店推荐专家。你的任务是根据城市和住宿偏好搜索酒店。

工作流程:
1. 分析用户的住宿偏好(经济型/舒适型/豪华/民宿)
2. 使用 amap_text_search 工具搜索对应类型的酒店
3. 整理酒店信息

注意: 根据用户偏好选择合适的关键词搜索，如:
- 经济型 → "经济型酒店" 或 "快捷酒店"
- 舒适型 → "商务酒店" 或 "三星级酒店"
- 豪华 → "五星级酒店" 或 "豪华酒店"
- 民宿 → "民宿" 或 "客栈"

以中文格式返回每个酒店的: 名称、地址、经纬度坐标、类型、评分(如有)、价格范围(如有)。
"""

PLANNER_SYSTEM_PROMPT = """你是行程规划专家。根据提供的景点、天气、酒店信息，生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划(不要包含其他文字):
```json
{{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {{
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {{"longitude": 116.397128, "latitude": 39.916527}},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离市中心2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      }},
      "attractions": [
        {{
          "name": "景点名称",
          "address": "详细地址",
          "location": {{"longitude": 116.397128, "latitude": 39.916527}},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐推荐", "description": "当地特色早餐", "estimated_cost": 30}},
        {{"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50}},
        {{"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}}
      ]
    }}
  ],
  "weather_info": [
    {{
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }}
  ],
  "overall_suggestions": "总体建议",
  "budget": {{
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }}
}}
```

规则:
1. weather_info数组必须包含每一天的天气
2. 温度必须是纯数字(不带°C)
3. 每天安排2-3个景点，考虑距离合理安排顺序
4. 每天必须包含早中晚三餐
5. 景点经纬度必须使用搜索到的真实坐标
6. 必须计算预算: 景点门票+酒店+餐饮+交通的总和
"""


def _extract_agent_output(result: dict) -> str:
    """从 create_agent 返回结果中提取最终输出文本"""
    messages = result.get("messages", [])
    # 取最后一条 AI 消息的内容
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.type == "ai":
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                # 内容可能是列表格式
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "\n".join(parts)
    return ""


class MultiAgentTripPlanner:
    """多智能体旅行规划系统 (LangChain版本)"""

    def __init__(self):
        """初始化多智能体系统"""
        print("[INIT] 开始初始化多智能体旅行规划系统 (LangChain)...")

        try:
            self.llm = get_llm()
            self.amap_tools = get_amap_tools()

            # --- 景点搜索Agent ---
            print("  - 创建景点搜索Agent...")
            self.attraction_agent = create_agent(
                model=self.llm,
                tools=self.amap_tools,
                system_prompt=ATTRACTION_SYSTEM_PROMPT,
            )

            # --- 天气查询Agent ---
            print("  - 创建天气查询Agent...")
            self.weather_agent = create_agent(
                model=self.llm,
                tools=self.amap_tools,
                system_prompt=WEATHER_SYSTEM_PROMPT,
            )

            # --- 酒店推荐Agent ---
            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = create_agent(
                model=self.llm,
                tools=self.amap_tools,
                system_prompt=HOTEL_SYSTEM_PROMPT,
            )

            # --- 行程规划Chain (无需工具,直接LLM调用) ---
            print("  - 创建行程规划Chain...")
            self.planner_chain = ChatPromptTemplate.from_messages([
                ("system", PLANNER_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]) | self.llm

            print(f"[OK] 多智能体系统初始化成功 (LangChain create_agent)")
            print(f"   景点Agent: {len(self.amap_tools)}个工具")
            print(f"   天气Agent: {len(self.amap_tools)}个工具")
            print(f"   酒店Agent: {len(self.amap_tools)}个工具")
            print(f"   规划器: 直接LLM调用")

        except Exception as e:
            print(f"[FAIL] 多智能体系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划（异步并行版本）
        """
        try:
            print(f"\n{'='*60}")
            print(f"[START] 开始多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")

            # ═══════════════════════════════════════════════════════
            # 步骤1-3: 并行执行(景点搜索 + 天气查询 + 酒店搜索)
            # ═══════════════════════════════════════════════════════
            print("[STEP 1-3] 并行搜索景点 + 查询天气 + 搜索酒店...\n")

            async def _run_attraction():
                query = self._build_attraction_query(request)
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.attraction_agent.invoke,
                        {"messages": [{"role": "user", "content": query}]}
                    ),
                    timeout=AGENT_TIMEOUT_ATTRACTION
                )
                return _extract_agent_output(result)

            async def _run_weather():
                query = f"请查询{request.city}的天气信息"
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.weather_agent.invoke,
                        {"messages": [{"role": "user", "content": query}]}
                    ),
                    timeout=AGENT_TIMEOUT_WEATHER
                )
                return _extract_agent_output(result)

            async def _run_hotel():
                query = f"请搜索{request.city}的{request.accommodation}"
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.hotel_agent.invoke,
                        {"messages": [{"role": "user", "content": query}]}
                    ),
                    timeout=AGENT_TIMEOUT_HOTEL
                )
                return _extract_agent_output(result)

            # 并行等待三个Agent完成
            attraction_text, weather_text, hotel_text = await asyncio.gather(
                _run_attraction(), _run_weather(), _run_hotel()
            )

            print(f"\n[RESULT] 景点搜索: {attraction_text[:200]}...")
            print(f"[RESULT] 天气查询: {weather_text[:200]}...")
            print(f"[RESULT] 酒店搜索: {hotel_text[:200]}...\n")

            # ═══════════════════════════════════════════════════════
            # 步骤4: 行程规划(串行,依赖前3步结果)
            # ═══════════════════════════════════════════════════════
            print("[STEP 4] 生成行程计划...")
            planner_input = self._build_planner_input(
                request, attraction_text, weather_text, hotel_text
            )
            planner_response = await asyncio.wait_for(
                asyncio.to_thread(self.planner_chain.invoke, {"input": planner_input}),
                timeout=AGENT_TIMEOUT_PLANNER
            )

            planner_text = planner_response.content if hasattr(planner_response, 'content') else str(planner_response)
            print(f"[RESULT] 行程规划: {planner_text[:300]}...\n")

            # 解析最终计划
            trip_plan = self._parse_response(planner_text, request)

            print(f"{'='*60}")
            print(f"[OK] 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except asyncio.TimeoutError as e:
            print(f"[TIMEOUT] Agent执行超时: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)
        except Exception as e:
            print(f"[FAIL] 生成旅行计划失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)

    def _build_attraction_query(self, request: TripRequest) -> str:
        """构建景点搜索查询"""
        keywords = request.preferences[0] if request.preferences else "热门景点"
        return (
            f"请使用amap_text_search工具搜索{request.city}的{keywords}相关景点。"
            f"搜索参数: keywords={keywords}, city={request.city}"
        )

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """截断过长的文本"""
        if len(text) <= max_chars:
            return text
        head_len = max_chars * 2 // 3
        tail_len = max_chars // 3
        return text[:head_len] + f"\n\n...(中间省略 {len(text) - max_chars} 字符)...\n\n" + text[-tail_len:]

    def _build_planner_input(self, request: TripRequest, attractions: str, weather: str, hotels: str) -> str:
        """构建行程规划输入"""
        attractions = self._truncate_text(attractions, 2000)
        weather = self._truncate_text(weather, 1500)
        hotels = self._truncate_text(hotels, 1500)

        input_parts = [
            f"请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:",
            "",
            "**基本信息:**",
            f"- 城市: {request.city}",
            f"- 日期: {request.start_date} 至 {request.end_date}",
            f"- 天数: {request.travel_days}天",
            f"- 交通方式: {request.transportation}",
            f"- 住宿偏好: {request.accommodation}",
            f"- 旅行偏好: {', '.join(request.preferences) if request.preferences else '无'}",
            "",
            "**景点信息:**",
            attractions,
            "",
            "**天气信息:**",
            weather,
            "",
            "**酒店信息:**",
            hotels,
            "",
            "**要求:**",
            "1. 每天安排2-3个景点, 合理利用天气信息安排室内外活动",
            "2. 每天必须包含早中晚三餐, 结合当地特色推荐",
            "3. 每天推荐一个具体酒店(从酒店信息中选取)",
            "4. 景点经纬度必须使用搜索到的真实坐标",
            "5. 直接返回JSON格式数据, 不要包含其他文字",
        ]

        if request.free_text_input:
            input_parts.insert(-1, f"**额外要求:** {request.free_text_input}")

        return "\n".join(input_parts)

    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """解析Agent响应中的JSON"""
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                raise ValueError("响应中未找到JSON数据")

            data = json.loads(json_str)
            return TripPlan(**data)

        except Exception as e:
            print(f"[WARN] 解析响应失败: {e}，使用备用方案")
            return self._create_fallback_plan(request)

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")

        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)

            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}推荐景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=116.4 + i * 0.01 + j * 0.005, latitude=39.9 + i * 0.01 + j * 0.005),
                        visit_duration=120,
                        description=f"{request.city}的热门景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner
