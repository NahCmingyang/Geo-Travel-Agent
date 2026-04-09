import os
import json
import re
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# --- 第一步：定义清洗函数 ---
def clean_xhs_text(text):
    text = re.sub(r'\[话题\]', '', text)
    text = text.replace('#', ' ')
    # 保持中英文数字，过滤乱码和表情
    text = re.sub(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9^，。！？、]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- 第二步：处理 JSON 文件 ---
def process_json_to_vector(file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", " "]
    )

    for item in data:
        note_id = item.get("note_id")
        title = item.get("title", "")
        desc = item.get("desc", "")

        clean_title = clean_xhs_text(title)
        clean_desc = clean_xhs_text(desc)
        full_content = f"标题：{clean_title}。正文：{clean_desc}"

        chunks = text_splitter.split_text(full_content)
        for i, chunk in enumerate(chunks):
            metadata = {
                "note_id": note_id,
                "chunk_index": i,
                "liked_count": str(item.get("liked_count", "0"))
            }
            all_docs.append({"content": chunk, "metadata": metadata})
    return all_docs


# --- 第三步：持久化到向量数据库 ---
def save_to_db(docs):
    DB_PATH = "./xhs_vector_db"

    # 清理旧库
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # 1. 使用你本地已经下好的 BGE 模型
    print("正在加载 BGE 本地模型...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    # 2. 准备数据
    texts = [d["content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    # 3. 写入数据库
    print(f"正在生成向量并写入数据库 (共 {len(texts)} 条)...")
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_PATH
    )
    print(f"数据库构建成功！保存路径: {DB_PATH}")


if __name__ == "__main__":
    # 路径确保正确
    JSON_FILE = ''
    all_data = process_json_to_vector(JSON_FILE)
    if all_data:
        save_to_db(all_data)