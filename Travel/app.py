import streamlit as st
from map_utils import MapEngine
from rag_engine import RAGEngine
from dotenv import load_dotenv

# 加载环境变量仅作为后端预备，前端输入默认置空
load_dotenv()

st.set_page_config(page_title="AI 旅行小管家", layout="wide")


# ==========================================
# 1. 初始化引擎
# ==========================================
@st.cache_resource
def init_engines(amap_key, ds_key):
    """
    根据传入的 Key 动态初始化引擎。
    """
    if not amap_key or not ds_key:
        return None, None

    try:
        map_en = MapEngine(amap_key)
        # 将用户输入的 DeepSeek Key 传给 RAG 引擎
        rag_en = RAGEngine("./xhs_vector_db", ds_key)
        return map_en, rag_en
    except Exception as e:
        st.error(f"引擎初始化失败，请检查 Key 是否有效: {e}")
        return None, None


# ==========================================
# 2. Session State 初始化与工具函数
# ==========================================
def init_session_state():
    defaults = {
        "pois": [],
        "briefs": {},
        "res": "",
        "current_city": "",
        "initialized": False,
        "selected_custom": [],
        "ordered_list": [],
        "route_details": None,
        "search_kw": "",
        "search_count": 0,
        "planning_done": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def clear_all_generated_data():
    """彻底清空所有生成的路线和文案数据"""
    st.session_state.res = ""
    st.session_state.ordered_list = []
    st.session_state.route_details = None
    st.session_state.planning_done = False


# ==========================================
# 3. 侧边栏：重新布局
# ==========================================
with st.sidebar:
    # --- 3.1 目的地设置 (置于上方) ---
    st.header("📍 目的地设置")
    city = st.text_input(
        "目标城市",
        placeholder="例如：绍兴",
        key="city_input"
    )

    start_loc = st.text_input(
        "出发地/起点",
        placeholder="例如：绍兴北站",
        key="start_input"
    )

    # 空白占位，将配置推向底部
    st.markdown("<br>" * 7, unsafe_allow_html=True)

    st.divider()

    # --- 3.2 系统配置 (置于左下角) ---
    st.header("⚙️ 系统配置")
    with st.expander("🔑 API Key 设置", expanded=False):  # 默认折叠，保持简洁
        input_ds_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=""
        )
        input_amap_key = st.text_input(
            "高德地图 Key",
            type="password",
            value=""
        )

    # 在所有输入组件定义完后再初始化引擎
    map_engine, rag_engine = init_engines(input_amap_key, input_ds_key)

    if st.button("🔍 获取推荐景点", use_container_width=True):
        if not map_engine:
            st.error("请先在下方配置有效的 API Key")
        elif not city.strip():
            st.warning("请先输入目标城市")
        else:
            # 切换城市时，强制重置所有状态
            if city != st.session_state.current_city:
                st.session_state.pois = []
                st.session_state.selected_custom = []
                clear_all_generated_data()

            st.session_state.initialized = True
            st.session_state.current_city = city

            with st.spinner("正在调取旅行笔记..."):
                try:
                    results_dict = rag_engine.extract_pois(city)
                    # 1. 把字典的键（景点名）转成列表给 pois
                    # 2. 把整个字典存入 briefs 供后续查询
                    st.session_state.pois = list(results_dict.keys())
                    st.session_state.briefs = results_dict

                except Exception as e:
                    st.error(f"获取失败：{e}")
            st.rerun()

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.title("🗺️ AI 旅行小管家")

# 检查权限：如果没有 Key，提示用户
if not map_engine or not rag_engine:
    st.info("👋 您好！请先在左侧侧边栏底部完成 **API Key 配置**。")
    st.stop()

if st.session_state.initialized and st.session_state.current_city:
    # --- 1. 推荐景点选择 ---
    st.markdown("## ① 选择推荐景点")
    if st.session_state.pois:
        selected_rec = st.pills(
            "从中挑选感兴趣的地点：",
            st.session_state.pois,
            selection_mode="multi"
        )
        if selected_rec:
            with st.container(border=True):
                st.caption("🔍 景点速览 (已根据本地资料更新)")
                briefs = st.session_state.get("briefs", {})
                for p in selected_rec:
                    # 从字典里拿介绍，拿不到则显示默认语
                    desc = briefs.get(p, "暂无详细介绍")
                    c1, c2 = st.columns([1, 4])
                    c1.markdown(f"**{p}**")
                    c2.markdown(f"*{desc}*")
    else:
        st.caption("暂无推荐，请手动搜索。")
        selected_rec = []

    # --- 2. 精准地点添加 ---
    st.markdown("---")
    st.markdown("## ② 补充精准地点")
    search_kw = st.text_input(
        "搜索特定位置：",
        value=st.session_state.search_kw,
        key=f"search_input_{st.session_state.search_count}"
    )

    if search_kw.strip():
        try:
            tips = map_engine.get_input_tips(search_kw, st.session_state.current_city)
            if tips:
                col1, col2 = st.columns([4, 1])
                with col1:
                    selected_tip = st.selectbox("请选择：", options=[""] + tips, key="tip_sel")
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("➕ 添加", use_container_width=True):
                        if selected_tip and selected_tip not in st.session_state.selected_custom:
                            st.session_state.selected_custom.append(selected_tip)
                            st.session_state.search_count += 1
                            clear_all_generated_data()
                            st.rerun()
        except Exception as e:
            st.error(f"联想失败：{e}")

    # --- 3. 管理已选地点 ---
    if st.session_state.selected_custom:
        for idx, place in enumerate(st.session_state.selected_custom):
            c1, c2 = st.columns([6, 1])
            c1.write(f"📍 {place}")
            if c2.button("删除", key=f"del_{idx}"):
                st.session_state.selected_custom.remove(place)
                clear_all_generated_data()  # 删除地点后清空旧路线
                st.rerun()

    # --- 4. 生成按钮逻辑 (强化清除逻辑) ---
    st.markdown("---")
    st.markdown("## ③ 规划行程")
    all_targets = list(dict.fromkeys((selected_rec if selected_rec else []) + st.session_state.selected_custom))
    st.markdown("### 📋 您的待规划清单")

    if not all_targets:
        st.warning("暂未选择任何地点")
    else:
        # 2. 界面呈现：使用 border=True 的容器包裹
        with st.container(border=True):
            cols = st.columns(3)
            for idx, p in enumerate(all_targets):
                with cols[idx % 3]:
                    # 这里的 info 组件就是你在页面上看到的蓝色小方块
                    st.info(f"📍 {p}")

    if all_targets:
        if st.button("🗺️ 生成优化路线与深度攻略", type="primary", use_container_width=True):
            if not start_loc.strip():
                st.error("请先填写左侧侧边栏的【出发地/起点】")
            else:
                # 核心改动：点击瞬间立即重置所有生成相关的状态
                clear_all_generated_data()

                with st.spinner("🚀 正在重新计算最优路径与 AI 攻略..."):
                    try:
                        # 1. 地图规划
                        ordered_list, details = map_engine.optimize_route(
                            st.session_state.current_city, start_loc, all_targets
                        )
                        st.session_state.ordered_list = ordered_list
                        st.session_state.route_details = details

                        # 2. AI 文案生成（流式预览）
                        st.markdown("### 🗓️ 生成中...")
                        stream_gen = rag_engine.generate_final_guide(
                            st.session_state.current_city,
                            st.session_state.ordered_list,
                            st.session_state.route_details
                        )
                        full_res = st.write_stream(stream_gen)

                        # 保存结果到 session，标记完成
                        st.session_state.res = full_res
                        st.session_state.planning_done = True
                        st.rerun()  # 刷新进入展示模式
                    except Exception as e:
                        st.error(f"规划失败：{e}")

    # --- 5. 结果展示区 ---
    if st.session_state.planning_done and st.session_state.res:
        st.divider()
        st.markdown(st.session_state.res)
        st.download_button(
            label="📥 导出攻略 (Markdown)",
            data=st.session_state.res,
            file_name=f"{st.session_state.current_city}_trip.md",
            use_container_width=True
        )
else:
    st.info("💡 请先在左侧输入 **目的地** 并获取景点推荐。")