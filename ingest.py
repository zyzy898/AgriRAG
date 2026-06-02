import os
import argparse
import config
import models
import milvus_client
import cache
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUMMARY_BATCH_SIZE = 3000     # 每批输入字符数（配合 RecursiveCharacterTextSplitter 在句子边界切分）
SUMMARY_BATCH_OVERLAP = 200   # 相邻批次重叠字符数（防止断句丢失上下文）
SKIP_SHORT_THRESHOLD = 3000   # 短于此字符数的文档跳过摘要生成

# 中文文本分隔符优先级：段落 → 换行 → 句号/问号/叹号 → 分号 → 逗号 → 空格 → 字符
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def call_summary_llm(system_prompt, user_prompt, max_tokens=1024):
    try:
        return models.call_llm_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=max_tokens,
            stream=False
        )
    except Exception as e:
        print(f"      [警告] API 调用失败: {e}")
        return ""


def split_text_batches(text, batch_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=batch_size,
        chunk_overlap=overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


def summarize_batch(text_batch, batch_idx, total_batches):
    """为单个文本批次生成短摘要（text1）和长摘要（text2），分别返回"""
    batch_label = f"第{batch_idx}/{total_batches}部分" if total_batches > 1 else ""

    system_short = "你是一个农业知识摘要引擎。只输出摘要文本本身，禁止任何前缀、后缀、解释或客套话。"
    user_short = (
        f"将以下文档{batch_label}总结为一句话（约20字），直接输出结果，不要任何额外文字：\n\n"
        f"{text_batch}"
    )
    short = call_summary_llm(system_short, user_short, max_tokens=128)

    system_long = (
        "你是一个农业知识结构化引擎。只输出markdown摘要内容本身。"
        "禁止输出任何开场白、过渡句、结语或解释，如'好的'、'本部分文档'、'以下是摘要'等。"
        "直接从正文标题开始输出。"
    )
    user_long = (
        f"将以下文档{batch_label}整理为markdown摘要，包含知识要点、分类和关键数据。"
        f"直接从内容开始，不要写任何介绍性或总结性文字：\n\n"
        f"{text_batch}"
    )
    long = call_summary_llm(system_long, user_long, max_tokens=1536)

    return short or "", long or ""


def load_and_process_knowledge_files():
    """
    加载知识文件，每批生成短摘要（text1→embedding）和长摘要（text2→LLM上下文）。
    短文档原文作为 text1+text2 直接入库。
    """
    documents = []
    knowledge_dir = config.KNOWLEDGE_DIR
    summary_dir = os.path.join(os.path.dirname(knowledge_dir), "summary")
    os.makedirs(summary_dir, exist_ok=True)

    if not os.path.exists(knowledge_dir):
        print(f"知识目录不存在: {knowledge_dir}")
        return documents

    for filename in sorted(os.listdir(knowledge_dir)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(knowledge_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            original_text = f.read().strip()

        if not original_text:
            print(f"跳过空文件: {filename}")
            continue

        print(f"\n{'=' * 50}")
        print(f"处理: {filename} ({len(original_text)} 字符)")

        if len(original_text) < SKIP_SHORT_THRESHOLD:
            # 短文档：原文作为 text1+text2 直接入库
            print(f"  短文档（<{SKIP_SHORT_THRESHOLD}字），原文直接入库")
            documents.append({
                "text1": original_text,
                "text2": original_text,
                "source": filename,
            })
        else:
            # 长文档：分批生成摘要
            batches = split_text_batches(original_text, SUMMARY_BATCH_SIZE, SUMMARY_BATCH_OVERLAP)
            print(f"  文本分割为 {len(batches)} 批 "
                  f"(每批≤{SUMMARY_BATCH_SIZE}字，重叠{SUMMARY_BATCH_OVERLAP}字)")

            all_shorts = []
            all_longs = []
            for i, batch_text in enumerate(batches):
                print(f"    处理第 {i+1}/{len(batches)} 批 ({len(batch_text)} 字符)...")
                short, long = summarize_batch(batch_text, i + 1, len(batches))
                if short or long:
                    all_shorts.append(short or "")
                    all_longs.append(long or "")

            # 保存完整摘要到 summary/ 目录
            combined = f"## {filename}\n\n"
            for j, (s, l) in enumerate(zip(all_shorts, all_longs)):
                combined += f"### {s}\n\n{l}\n\n---\n\n"
            summary_path = os.path.join(summary_dir, filename.replace(".txt", "_summary.md"))
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(combined)
            print(f"  摘要已保存: {summary_path}")

            # 每批摘要独立入库（text1=短摘要→embedding，text2=长摘要→LLM上下文）
            MAX_TEXT2 = 18000
            for s, l in zip(all_shorts, all_longs):
                documents.append({
                    "text1": s,
                    "text2": l[:MAX_TEXT2],
                    "source": filename,
                })
            print(f"  入库 {len(all_shorts)} 条摘要记录")

    return documents


def ingest(reset=False, build_graph=False):
    print("正在加载并处理知识文档...")
    documents = load_and_process_knowledge_files()

    if not documents:
        print("没有可入库的文档块")
        return

    print(f"\n共 {len(documents)} 条记录，正在生成 Embedding（基于 text1 短摘要）...")

    texts_for_emb = [doc["text1"] for doc in documents]
    embeddings = models.encode_texts(texts_for_emb)

    docs_with_embeddings = []
    for i, doc in enumerate(documents):
        doc["embedding"] = embeddings[i].tolist()
        docs_with_embeddings.append(doc)

    print("正在存储到 Milvus...")
    if reset:
        milvus_client.reset_collection()
    else:
        milvus_client.create_collection()

    count = milvus_client.insert_documents(docs_with_embeddings)
    print(f"入库完成，新增 {count} 条记录")

    if count > 0:
        cache.invalidate_all()

    if build_graph:
        print(f"\n{'=' * 50}")
        print("构建 Neo4j 知识图谱...")
        try:
            import entity_relation_extractor
            entity_relation_extractor.extract_and_build_graph(clear_first=reset)
        except ImportError:
            print("[图谱] entity_relation_extractor 模块不可用，跳过图谱构建")
        except Exception as e:
            print(f"[图谱] 构建失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识入库工具 — 双字段摘要存储 + 知识图谱构建")
    parser.add_argument("--reset", action="store_true", help="清空旧数据后重新入库")
    parser.add_argument("--build-graph", action="store_true", help="同时构建 Neo4j 知识图谱")
    args = parser.parse_args()
    ingest(reset=args.reset, build_graph=args.build_graph)
