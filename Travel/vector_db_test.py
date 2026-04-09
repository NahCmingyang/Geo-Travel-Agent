import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def test_search():
    # 1. 配置路径和模型（必须与构建库时完全一致）
    DB_PATH = "./xhs_vector_db"
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    if not os.path.exists(DB_PATH):
        print(f"错误：找不到数据库目录 {DB_PATH}，请先运行构建脚本。")
        return

    # 2. 初始化 Embedding 模型
    print(f"正在加载模型 {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 3. 加载现有的向量数据库
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # 4. 定义你想测试的问题列表
    test_queries = [
        "XX两天一夜怎么安排路线？",
        "XX附近有什么好吃的？",
        "XX旅游需要注意什么避雷？",
        "哪个景点人比较少？"
    ]

    print("\n" + "=" * 50)
    print("🚀 开始进行向量数据库检索测试")
    print("=" * 50)

    for query in test_queries:
        print(f"\n🔍 用户问题: {query}")

        # k=3 表示返回最相似的前 3 个片段
        # similarity_search_with_score 会返回 (文档, 分数)
        # 分数越小代表越相似（距离越近）
        results = db.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(results):
            print(f"\n  [匹配结果 {i + 1}] (相似度分数: {score:.4f})")
            print(f"  内容梗概: {doc.page_content[:150]}...")
            print(f"  元数据信息: note_id={doc.metadata.get('note_id')}, 点赞={doc.metadata.get('liked_count')}")

        print("-" * 30)


if __name__ == "__main__":
    test_search()
