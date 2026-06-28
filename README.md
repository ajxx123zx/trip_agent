# 🧳 Trip Agent — LangGraph 智能旅行规划助手

基于 **LangGraph StateGraph** 的多智能体旅行规划系统。3 路并行 Agent 搜索 → 汇聚决策 → LLM 生成结构化行程计划，SSE 流式输出，条件边自动容错兜底。

> 📌 原始 `LangChain create_agent` 版本：[trip_agent](../trip_agent)

---

## 🏗️ 系统架构

```
                    ┌─────────────────┐
                    │     START       │
                    └──────┬──────────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
  ┌────────────────┐ ┌───────────┐ ┌──────────────┐
  │ attraction     │ │ weather   │ │ hotel        │
  │ agent (60s)    │ │ agent(45s)│ │ agent (60s)  │
  │ LLM + 景点搜索  │ │ LLM + 天气 │ │ LLM + 酒店搜索 │
  └───────┬────────┘ └─────┬─────┘ └──────┬───────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                  ┌────────────────┐
                  │  merge 汇聚决策  │
                  │ 检查完整性/重试  │
                  └───────┬────────┘
                          │ 条件边
                 ┌────────┴────────┐
                 ▼                 ▼
        ┌──────────────┐  ┌──────────────┐
        │ planner (90s)│  │ fallback     │
        │ LLM 生成JSON  │  │ 语义完整兜底  │
        │ 结构化行程     │  │ 降级方案     │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               └────────┬────────┘
                        ▼
                   ┌────────┐
                   │  END   │
                   └────────┘
```

### 6 节点职责

| 节点 | 功能 | 工具 | 超时 |
|------|------|------|------|
| `attraction_agent` | LLM 驱动搜索目的地景点 | `amap_text_search` | 60s |
| `weather_agent` | 查询未来几天天气 | `amap_weather` | 45s |
| `hotel_agent` | 搜索住宿推荐 | `amap_text_search` | 60s |
| `merge` | 汇聚三路结果，决策重试/降级 | — | — |
| `planner` | 汇总信息，LLM 生成结构化 JSON 行程 | — | 90s |
| `fallback` | 全链路失败的语义完整兜底计划 | — | — |

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔀 **3 路并行 Agent** | 景点/天气/酒店同时搜索，互不阻塞 |
| 🧠 **LLM 驱动工具调用** | Agent 自主决定调用高德 API 搜索 |
| 🔗 **StateGraph 状态图** | 有向图编排，可视化工作流 |
| 🛡️ **4 级容错** | 单 Agent 超时 → 重试 → 降级文本 → 兜底计划 |
| 📡 **SSE 流式输出** | 实时推送进度（agent_start / tool_call / plan_complete） |
| 🗺️ **高德地图可视化** | 景点标记 + 信息窗 + 路线绘制 |
| 📸 **Unsplash 景点图片** | 自动为每个景点加载高质量实景照片 |
| 📥 **一键导出** | 行程导出为 PNG 图片 / PDF 文件 |
| ✏️ **行程可编辑** | 景点顺序调整、增删、修改 |

---

## 🛠️ 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| **Agent 框架** | LangGraph StateGraph + LangChain | ≥0.2.0 / ≥0.3.0 |
| **LLM** | DeepSeek V4 Pro (OpenAI 兼容) | — |
| **后端** | Python 3.10+ / FastAPI / Uvicorn | ≥0.115.0 |
| **数据模型** | Pydantic v2 + TypedDict | ≥2.0.0 |
| **地图服务** | 高德地图 Web API + JS API 2.0 | — |
| **图片服务** | Unsplash API | — |
| **前端** | Vue 3 + TypeScript + Vite 6 | ≥3.5 |
| **UI 框架** | Ant Design Vue 4 | ≥4.2 |
| **导出** | html2canvas + jsPDF | — |

---

## 📂 项目结构

```
trip_agent_langgraph/
├── README.md
├── backend/
│   ├── .env                        # 环境变量（需自行创建）
│   ├── requirements.txt
│   ├── run.py                      # 启动入口
│   └── app/
│       ├── __init__.py
│       ├── config.py               # 配置管理 (Pydantic Settings)
│       ├── api/
│       │   ├── main.py             # FastAPI 应用 + CORS
│       │   └── routes/
│       │       ├── trip.py         # /api/trip/plan + SSE 流式
│       │       └── poi.py          # /api/poi/photo
│       ├── agents/
│       │   └── langgraph_planner.py  # 🌟 LangGraph 图编排核心
│       ├── services/
│       │   ├── llm_service.py      # ChatOpenAI 单例
│       │   ├── amap_service.py     # 高德 HTTP API + LangChain Tools
│       │   └── unsplash_service.py # Unsplash 图片搜索
│       └── models/
│           └── schemas.py          # Pydantic 模型 + TripAgentState
└── frontend/
    ├── .env                        # 前端环境变量（需自行创建）
    ├── package.json
    ├── vite.config.ts              # Vite 配置 + API 代理
    ├── index.html
    └── src/
        ├── main.ts                 # Vue 入口
        ├── App.vue
        ├── types/index.ts          # TypeScript 类型定义
        ├── services/api.ts         # Axios 封装
        └── views/
            ├── Home.vue            # 首页 — 旅行表单（日期/偏好/额外要求）
            └── Result.vue          # 结果页 — 行程展示 + 高德地图 + 导出
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低版本 |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| 高德地图 API Key | Web 服务 + Web JS API（两个独立 Key） |
| LLM API Key | DeepSeek / OpenAI 兼容接口 |
| Unsplash API Key | Access Key（可选，景点图片服务） |

### 1. 配置环境变量

**后端 `backend/.env`：**

```bash
# LLM 配置
LLM_MODEL_ID=deepseek-v4-pro
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com
LLM_TIMEOUT=120

# 服务器配置
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# 高德地图 Web 服务 API Key
AMAP_API_KEY=your-amap-web-api-key

# Unsplash 图片 API（可选）
UNSPLASH_ACCESS_KEY=your-unsplash-access-key
UNSPLASH_SECRET_KEY=your-unsplash-secret-key
```

**前端 `frontend/.env`：**

```bash
# 后端 API 地址
VITE_API_BASE_URL=http://localhost:8000

# 高德地图 JS API Key（Web 端）
VITE_AMAP_WEB_JS_KEY=your-amap-js-api-key
```

> ⚠️ 高德地图 JS API Key 需在[高德控制台](https://console.amap.com/dev/key/app)配置域名白名单（开发环境可留空或填 `127.0.0.1:5173`）。

### 2. 安装依赖

```bash
# 后端
cd backend
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 3. 启动服务

```bash
# 终端 1 — 启动后端
cd backend
python run.py
# → http://localhost:8000
# → API 文档: http://localhost:8000/docs

# 终端 2 — 启动前端
cd frontend
npm run dev
# → http://localhost:5173
```

---

## 📡 API 文档

| 方法 | 路径 | Content-Type | 说明 |
|------|------|-------------|------|
| `GET` | `/health` | — | 服务健康检查 |
| `POST` | `/api/trip/plan` | `application/json` | 生成旅行计划 |
| `POST` | `/api/trip/plan/stream` | `application/json` | SSE 流式规划 |
| `GET` | `/api/trip/health` | — | 旅行规划服务状态 |
| `GET` | `/api/poi/photo?name=故宫` | — | 获取景点 Unsplash 图片 |

### 请求示例

```json
POST /api/trip/plan
Content-Type: application/json

{
  "city": "北京",
  "start_date": "2025-07-01",
  "end_date": "2025-07-03",
  "travel_days": 3,
  "transportation": "公共交通",
  "accommodation": "经济型酒店",
  "preferences": ["历史文化", "美食"],
  "free_text_input": "希望多安排博物馆"
}
```

### 响应示例

```json
{
  "success": true,
  "message": "旅行计划生成成功",
  "data": {
    "city": "北京",
    "start_date": "2025-07-01",
    "end_date": "2025-07-03",
    "days": [
      {
        "date": "2025-07-01",
        "day_index": 0,
        "description": "探索天安门广场区域的历史文化",
        "transportation": "地铁1号线",
        "accommodation": "经济型酒店",
        "attractions": [
          {
            "name": "中国国家博物馆",
            "address": "东长安街16号",
            "location": {"longitude": 116.401, "latitude": 39.905},
            "visit_duration": 180,
            "description": "中国最高历史文化殿堂...",
            "category": "博物馆",
            "rating": 4.9,
            "ticket_price": 0
          }
        ],
        "meals": [
          {"type": "breakfast", "name": "酒店早餐", "description": "自助早餐"},
          {"type": "lunch", "name": "前门炸酱面", "description": "老北京特色"}
        ],
        "hotel": {
          "name": "北京艺栈青年酒店",
          "address": "新源南路甲3号",
          "price_range": "200-400元",
          "rating": "4.8"
        }
      }
    ],
    "weather_info": [
      {
        "date": "2025-07-01",
        "day_weather": "晴",
        "day_temp": 35,
        "night_temp": 22
      }
    ],
    "overall_suggestions": "故宫需提前预约，建议上午8点前到达...",
    "budget": {
      "total_attractions": 180,
      "total_hotels": 600,
      "total_meals": 450,
      "total_transportation": 100,
      "total": 1330
    }
  }
}
```

### SSE 流式事件

```
data: {"stage":"agent_start","agent":"attraction_agent","message":"正在搜索景点..."}
data: {"stage":"tool_call","tool":"amap_text_search"}
data: {"stage":"merge_complete","has_attraction":true,"has_weather":true,"has_hotel":true}
data: {"stage":"plan_complete","trip_plan":{...}}
data: {"stage":"done","message":"规划完成"}
```

---

## 🛡️ 容错策略

| 场景 | 处理方式 |
|------|---------|
| 单个 Agent 超时 | merge 节点检测 → 重试 1 次 |
| 重试仍失败 | 降级文本填充（如"北京 热门景点"）→ 继续规划 |
| 三路全失败 | 条件边路由到 `fallback` → 生成语义完整兜底计划 |
| Planner LLM 失败 | 条件边路由到 `fallback` |
| 总超时 180s | `asyncio.wait_for` 兜底 |
| LLM JSON 字段名偏差 | `_normalize_trip_json()` 自动映射 |

**设计原则：绝不返回 500。** 任何异常路径都有语义完整的行程计划输出。

---

## 🆚 与原始版本对比

| 维度 | 原始 (create_agent) | LangGraph 版 |
|------|---------------------|-------------|
| 编排方式 | `asyncio.gather` 手动调度 | `StateGraph` 有向图声明式编排 |
| 状态管理 | 局部变量散落各处 | `TripAgentState` TypedDict 集中管理 |
| 容错机制 | 分散 try/except | 条件边自动路由 + 独立 fallback 节点 |
| 重试机制 | 无 | merge 节点决策重试 |
| 流式输出 | 无 | SSE `astream_events` 实时推送 |
| 可观测性 | 仅 print | 图事件钩子 + 节点级追踪 |
| 兜底方案 | API 层后处理 | 独立 fallback 节点 |

---

## 🔧 常见问题

### Q: 地图只显示网格？
A: 前端 `.env` 中的 `VITE_AMAP_WEB_JS_KEY` 未配置或域名白名单不包含当前访问地址。去[高德控制台](https://console.amap.com/dev/key/app)配置白名单，然后**重启前端 dev server**（Vite 只在启动时加载 `.env`）。

### Q: 行程总是"降级方案"？
A: LLM 返回的 JSON 字段名不匹配 Pydantic 模型。检查后端控制台是否有 `ValidationError` 日志，确认 `_normalize_trip_json()` 字段映射生效。

### Q: 景点图片显示不出来？
A: 检查 `UNSPLASH_ACCESS_KEY` 是否配置。景点图片为可选功能，加载失败时自动使用渐变色占位图。

### Q: CORS 报错？
A: 确保后端 `.env` 中 `CORS_ORIGINS` 包含前端地址（默认 `http://localhost:5173`）。

---

## 📝 License

MIT
