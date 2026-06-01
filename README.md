# 🧳 智能旅行规划助手 (Trip Agent)

基于 **LangChain 多智能体框架**的智能旅行规划系统，采用前后端分离架构。用户输入目的地、日期、偏好等需求后，多个 AI Agent 并行协作（搜索景点、查询天气、推荐酒店），最终生成一份包含每日行程、餐饮推荐和预算估算的完整旅行计划。

---

## ✨ 核心特性

- **🤖 多智能体协作** — 4 个 LangChain Agent（景点/天气/酒店/规划器）协同工作，前 3 个并行执行，效率更高
- **🗺️ 真实地理数据** — 集成高德地图 API，搜索真实的景点、酒店坐标与天气信息
- **🖼️ 景点配图** — 集成 Unsplash API，为景点自动匹配高质量图片
- **📊 预算估算** — 自动汇总门票、住宿、餐饮、交通费用
- **🛡️ 容错机制** — Agent 超时或失败时自动 fallback 兜底计划
- **📄 API 文档** — 基于 Swagger/ReDoc 自动生成交互式 API 文档
- **🎨 现代前端** — Vue 3 + TypeScript + Ant Design Vue，支持行程导出为 PDF

---

## 🏗️ 架构总览

```
┌─────────────────────────────────────────────────────┐
│                     前端 (Vue 3)                     │
│         Ant Design Vue + 高德地图 JSAPI              │
│               localhost:5173                         │
└────────────────────┬────────────────────────────────┘
                     │ HTTP / WebSocket
                     ▼
┌─────────────────────────────────────────────────────┐
│                 后端 (FastAPI)                       │
│               localhost:8000                         │
│                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ 景点搜索  │ │ 天气查询  │ │ 酒店推荐  │  ← 并行   │
│  │ Agent    │ │ Agent    │ │ Agent    │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       └──────────────┼──────────────┘               │
│                      ▼                               │
│              ┌──────────────┐                        │
│              │  行程规划器   │  ← 汇总前三者结果       │
│              └──────────────┘                        │
│                      │                               │
│         ┌────────────┼────────────┐                  │
│         ▼            ▼            ▼                  │
│    高德地图 API   LLM API    Unsplash API            │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- 高德地图 API Key（[免费申请](https://lbs.amap.com/)）
- LLM API Key（OpenAI 兼容接口）
- Unsplash Access Key（可选，[免费申请](https://unsplash.com/developers)）

### 1. 克隆项目

```bash
git clone <repo-url>
cd trip_agent
```

### 2. 后端配置

```bash
cd backend

# 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

创建 `backend/.env` 文件：

```env
# 高德地图 API
AMAP_API_KEY=your_amap_api_key_here

# LLM 配置 (OpenAI 兼容接口)
OPENAI_API_KEY=your_llm_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# 或使用独立环境变量覆盖
# LLM_API_KEY=xxx
# LLM_BASE_URL=xxx
# LLM_MODEL_ID=xxx

# Unsplash (可选)
UNSPLASH_ACCESS_KEY=your_unsplash_access_key_here
```

### 3. 启动后端

```bash
python run.py
```

后端运行在 `http://localhost:8000`，访问：
- Swagger 文档：http://localhost:8000/docs
- ReDoc 文档：http://localhost:8000/redoc

### 4. 前端配置与启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端运行在 `http://localhost:5173`，API 请求自动代理到后端。

---

## 📡 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/trip/plan` | 🎯 **核心接口** — 提交旅行需求，生成完整计划 |
| `GET` | `/api/trip/health` | 旅行规划服务健康检查 |
| `GET` | `/api/map/poi` | POI 关键词搜索 |
| `GET` | `/api/map/weather` | 城市天气查询 |
| `POST` | `/api/map/route` | 两点间路线规划 |
| `GET` | `/api/poi/detail/{id}` | POI 详细信息 |
| `GET` | `/api/poi/photo` | 景点 Unsplash 图片搜索 |

### 核心接口示例

**请求** — `POST /api/trip/plan`：

```json
{
  "city": "北京",
  "start_date": "2025-06-01",
  "end_date": "2025-06-03",
  "travel_days": 3,
  "transportation": "公共交通",
  "accommodation": "经济型酒店",
  "preferences": ["历史文化", "美食"],
  "free_text_input": "希望多安排一些博物馆"
}
```

**响应** — 包含每日行程、景点坐标、天气、餐饮、预算的完整 JSON：

```json
{
  "success": true,
  "message": "旅行计划生成成功",
  "data": {
    "city": "北京",
    "start_date": "2025-06-01",
    "end_date": "2025-06-03",
    "days": [
      {
        "date": "2025-06-01",
        "day_index": 0,
        "description": "第1天：探索古都历史文化",
        "attractions": [...],
        "meals": [
          {"type": "breakfast", "name": "护国寺小吃", "estimated_cost": 30},
          {"type": "lunch", "name": "四季民福烤鸭店", "estimated_cost": 80},
          {"type": "dinner", "name": "南门涮肉", "estimated_cost": 100}
        ],
        "hotel": {"name": "如家快捷酒店(王府井店)", "estimated_cost": 350}
      }
    ],
    "weather_info": [...],
    "budget": {
      "total_attractions": 180,
      "total_hotels": 1050,
      "total_meals": 630,
      "total_transportation": 150,
      "total": 2010
    }
  }
}
```

---

## 📂 项目结构

```
trip_agent/
├── README.md
├── backend/
│   ├── run.py                          # 启动入口
│   ├── requirements.txt                # Python 依赖
│   └── app/
│       ├── config.py                   # 配置管理 (.env)
│       ├── api/
│       │   ├── main.py                 # FastAPI 主应用 + CORS
│       │   └── routes/
│       │       ├── trip.py             # 旅行规划 API
│       │       ├── poi.py              # POI + Unsplash 图片 API
│       │       └── map.py              # 地图服务 API
│       ├── agents/
│       │   └── trip_planner_agent.py   # 🌟 多智能体编排核心
│       ├── services/
│       │   ├── llm_service.py          # LLM 封装 (LangChain ChatOpenAI)
│       │   ├── amap_service.py         # 高德地图 API + LangChain 工具
│       │   └── unsplash_service.py     # Unsplash 图片服务
│       └── models/
│           └── schemas.py              # Pydantic 数据模型
└── frontend/
    ├── package.json
    ├── vite.config.ts                  # Vite 配置 + API 代理
    └── src/
        ├── main.ts                     # Vue 入口 + 路由
        ├── App.vue
        ├── views/
        │   ├── Home.vue                # 首页 — 旅行表单
        │   └── Result.vue              # 结果页 — 行程展示
        └── types/
            └── index.ts                # TypeScript 类型定义
```

---

## 🔧 技术栈

### 后端

| 技术 | 用途 |
|------|------|
| **FastAPI** | Web 框架，异步支持，自动生成 API 文档 |
| **LangChain** | Agent 框架，`create_agent` 编排多智能体 |
| **LangChain ChatOpenAI** | LLM 调用（兼容 OpenAI 格式的任意模型） |
| **Pydantic** | 数据校验与序列化 |
| **httpx** | HTTP 客户端（高德/Unsplash API 调用） |
| **uvicorn** | ASGI 服务器 |

### 前端

| 技术 | 用途 |
|------|------|
| **Vue 3** | 前端框架（Composition API） |
| **TypeScript** | 类型安全 |
| **Vite** | 构建工具 |
| **Ant Design Vue** | UI 组件库 |
| **高德 JSAPI** | 地图展示与交互 |
| **jsPDF + html2canvas** | 行程 PDF 导出 |

---

## 🤖 Agent 工作流详解

1. **并行阶段**（`asyncio.gather`）：
   - **景点搜索 Agent** — 根据 `preferences` 中的偏好标签（如"历史文化"、"美食"）调用 `amap_text_search` 工具搜索景点
   - **天气查询 Agent** — 调用 `amap_weather` 工具获取目的地 7 天天气预报
   - **酒店推荐 Agent** — 根据 `accommodation`（经济型/舒适型/豪华/民宿）调用 `amap_text_search` 搜索酒店

2. **串行阶段**：
   - **行程规划器** — 汇总前三者的结果，通过 LLM 直接生成结构化 JSON（包含每日景点安排、三餐推荐、酒店推荐、预算明细）

3. **容错机制**：
   - 各 Agent 设有独立超时（60s~120s）
   - 超时或解析失败时自动生成 fallback 兜底计划
   - 搜索结果过长时自动截断（防止 token 溢出）

---

## ⚙️ 配置说明

所有配置通过 `backend/.env` 文件管理：

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `AMAP_API_KEY` | ✅ | 高德地图 Web 服务 API Key | - |
| `OPENAI_API_KEY` | ✅ | LLM API Key | - |
| `OPENAI_BASE_URL` | ❌ | LLM 服务地址 | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | ❌ | 模型名称 | `gpt-4` |
| `LLM_API_KEY` | ❌ | LLM Key（覆盖 `OPENAI_API_KEY`） | - |
| `LLM_BASE_URL` | ❌ | LLM 地址（覆盖 `OPENAI_BASE_URL`） | - |
| `LLM_MODEL_ID` | ❌ | 模型名（覆盖 `OPENAI_MODEL`） | - |
| `UNSPLASH_ACCESS_KEY` | ❌ | Unsplash API Key（用于景点图片） | - |
| `LLM_TIMEOUT` | ❌ | LLM 请求超时（秒） | `120` |

---

## 🛠️ 开发

```bash
# 后端 — 开发模式（热重载）
cd backend
python run.py

# 前端 — 开发模式（热重载）
cd frontend
npm run dev

# 前端 — 生产构建
npm run build
```

---

## 📝 License

MIT
