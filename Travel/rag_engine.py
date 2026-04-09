import os
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RAGEngine:
    def __init__(self, db_path, api_key):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        self.vector_db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 8})

        # 使用传入的 api_key
        self.llm = ChatDeepSeek(
            model="deepseek-reasoner",
            temperature=0.3,
            api_key=api_key
        )

    def _get_valid_context(self, city, query):
        """校验检索到的内容是否属于该城市"""
        docs = self.retriever.invoke(query)
        if not docs: return ""

        full_text = "\n".join([d.page_content for d in docs])
        if city not in full_text:
            return ""
        return full_text

    def extract_pois(self, city):
        """
        合并功能：一次性提取景点名称及其极简介绍
        返回格式: { "景点名": "简介内容", ... }
        """
        context = self._get_valid_context(city, city)

        prompt = f"""
        任务：列出“{city}”值得去的经典景点，并为每个景点写一段极简介绍。
        参考资料：{context if context else "（请根据你的知识库回答）"}

        要求：
        1. 介绍必须控制在 20 字以内。
        2. 输出格式必须严格为：景点名:介绍内容;景点名:介绍内容;
        3. 不要包含任何换行符或额外文字。
        """
        res = self.llm.invoke(prompt).content

        # 兼容处理并解析
        clean_res = res.replace("；", ";").replace("\n", "").strip()
        poi_dict = {}
        for item in clean_res.split(";"):
            if ":" in item:
                name, brief = item.split(":", 1)
                poi_dict[name.strip()] = brief.strip()
            elif "：" in item:  # 兼容中文冒号
                name, brief = item.split("：", 1)
                poi_dict[name.strip()] = brief.strip()

        return poi_dict

    def generate_final_guide(self, city, ordered_route, route_details):
        # 1. 整理物理数据
        physics_report = ""
        total_min = 0
        for d in route_details:
            total_min += d.get('min', 0)
            physics_report += (f"- 从【{d['from']}】到【{d['to']}】：\n"
                               f"  距离：{d.get('km')} 公里, 预计耗时：{d.get('min')} 分钟, ")

        # 2. 获取背景资料
        query = f"{city} {' '.join(ordered_route[:3])} 旅游建议"
        context = self._get_valid_context(city, query)

        # 3. 编写更理智的 Prompt
        prompt = f"""
        你是一个极其专业且人性化的旅游管家。

        【核心任务】
        用户确定的目标城市是【{city}】，计划游览以下地点：{" -> ".join(ordered_route)}。

        【交通数据参考】
        {physics_report}

        【规划原则】
        1. **严控每日负荷**：
            - **核心准则**：每天安排的大型景点（游玩耗时>2小时的）严禁超过 3 个。
            - **强制留白**：在每两个景点之间，除了交通时间外，必须预留 30-60 分钟的“漫步/休息”缓冲时间。
        2. **自动分天**：
            - 如果目标地点总数超过 4 个，或者地理跨度极大，必须将行程拆分为“Day 1”、“Day 2”等。
            - 地理位置相近的点聚类在一起，减少奔波感。
        3. **强制生活化节奏**：
            - 行程中考虑【午餐】（11:30-13:30）和【下午茶/晚餐】（16:30-19:30）,如指定餐饮店，必须合理插入行程规划。
            - 默认行程不宜早于上午 9:30 开始，不宜晚于晚上 20:00 结束。

        【背景参考】
        {context if context else "请根据专业常识规划"}

        【输出格式】
        - 每一段路程必须保留【交通看板】（包含距离、预计时间、建议交通方式）。
        - 每一站都要有游玩建议和避雷提醒。
        - 如果需要分天旅游，必须以“--- Day x：[放松的主题] ---”作为分天标题。
        """
        return self.llm.stream(prompt)
