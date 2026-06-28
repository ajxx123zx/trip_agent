"""
LangGraph 旅行规划 API 路由

端点:
    POST /api/trip/plan        — 标准模式
    POST /api/trip/plan/stream — SSE 流式
    GET  /api/trip/health      — 健康检查
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...models.schemas import TripRequest, TripPlanResponse
from ...agents.langgraph_planner import get_langgraph_planner

router = APIRouter(prefix="/api/trip", tags=["旅行规划 (LangGraph)"])


@router.post("/plan", response_model=TripPlanResponse,
             summary="生成旅行计划 (LangGraph)",
             description="使用 LangGraph StateGraph 多Agent协作生成旅行计划")
async def plan_trip(request: TripRequest):
    try:
        print(f"[LangGraph] {request.city} | {request.travel_days}天")
        planner = get_langgraph_planner()
        trip_plan = await planner.plan_trip(request)
        print("[LangGraph] 完成")
        return TripPlanResponse(success=True, message="旅行计划生成成功", data=trip_plan)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"规划失败: {str(e)}")


@router.post("/plan/stream",
             summary="SSE流式生成 (LangGraph)",
             description="实时推送旅行规划进度，前端通过 EventSource 消费")
async def plan_trip_stream(request: TripRequest):
    """
    SSE 事件类型:
    - agent_start:   {"agent":"attraction_agent","message":"正在搜索景点..."}
    - tool_call:     {"tool":"amap_text_search"}
    - merge_complete:{"has_attraction":true,"has_weather":true,"has_hotel":true}
    - plan_complete: {"trip_plan":{...}}
    - fallback:      {"message":"...","trip_plan":{...}}
    - done/error
    """
    planner = get_langgraph_planner()

    async def event_generator():
        try:
            async for event in planner.plan_trip_streaming(request):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@router.get("/health", summary="健康检查 (LangGraph)")
async def health_check():
    try:
        planner = get_langgraph_planner()
        return {
            "status": "healthy",
            "service": "trip-planner-langgraph",
            "framework": "LangGraph",
            "version": "3.0.0",
            "tools": [t.name for t in planner.amap_tools],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
