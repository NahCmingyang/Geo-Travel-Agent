# Geo-Travel-Agent
# 🗺️ AI 旅行小管家 (AI Travel Planner)

本项目是一个基于 **LangChain** 框架开发的垂直领域智能体（Agent）。它结合了 **DeepSeek** 大模型、**向量数据库 (RAG)** 和 **高德地图 API**，旨在为用户提供从景点推荐、路线优化到深度攻略生成的一站式旅行规划服务。

## 🌟 核心功能

* **双引擎景点获取**：
    * **RAG 推荐**：利用 `RAGEngine` 从本地向量数据库（爬取小红书旅游笔记搭建而成）检索特定城市的旅游攻略。
    * **精准搜索**：通过 `MapEngine` 调用高德地图 POI 联想 API，支持用户手动搜索并添加特定位置。
* **地理空间路径优化**：
    * **坐标转换**：将地名自动转换为经纬度坐标。
    * **路网计算**：获取真实路网下的行驶距离与预计耗时。
    * **贪心算法排序**：基于起点和目标点，自动计算物理路径上的最优游览顺序，减少折返。
* **智能行程规划**：
    * **多约束生成**：AI 在生成攻略时会考虑每日游玩上限（不超过 3 个大景点）、留白缓冲时间以及餐饮时间。
    * **动态分天**：根据景点数量和地理跨度自动将行程拆分为多天展示。

## 🛠️ 技术栈

* **大模型后端**: DeepSeek-V3 (via `langchain_deepseek`)
* **地理信息服务**: 高德地图 Web 服务 API
* **向量数据库**: ChromaDB (存储本地旅游知识库)
* **嵌入模型**: HuggingFace `bge-small-zh-v1.5`
* **应用框架**: Streamlit

## 📂 文件结构

* `build_db.py`: 向量数据库搭建，负责将爬到的小红书笔记（json格式）进行数据清晰并构建向量数据库。
* `vector_db_test.py`: 向量数据库搭建结果验证，检验数据库质量。
* `app.py`: Streamlit 前端交互界面，负责 Session State 管理与业务流程编排。
* `map_utils.py`: 地图引擎封装，处理地理编码、路网规划与路径排序算法。
* `rag_engine.py`: RAG 引擎封装，负责向量检索、景点提取及最终长文本攻略的 Prompt 调度。

## 🚀 快速开始

1.  **环境配置**：确保安装所需库：
    ```bash
    pip install streamlit langchain_deepseek langchain_chroma langchain_huggingface requests python-dotenv
    ```
2.  **准备数据**：在项目目录下放置预先构建好的向量库文件夹 `./xhs_vector_db`。
3.  **运行程序**：
    ```bash
    streamlit run app.py
    ```
4.  **配置 Key**：在页面侧边栏输入有效的高德地图 Key 和 DeepSeek API Key。
