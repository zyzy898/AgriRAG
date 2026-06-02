import os
import re
import json
import config
import models
from graph_retriever import get_neo4j_driver, GRAPH_DISEASE_LABEL


def _merge_duplicate_sections(sections):
    merged = {}
    for s in sections:
        key = (s["name"], s["source"])
        if key in merged:
            merged[key]["full_text"] += "\n" + s["full_text"]
        else:
            merged[key] = dict(s)
    return list(merged.values())


def _call_extraction_llm(system_prompt, user_prompt, max_tokens=1024):
    try:
        return models.call_llm_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=max_tokens,
            stream=False
        )
    except Exception as e:
        print(f"      [警告] 实体抽取 API 调用失败: {e}")
        return ""


def _load_knowledge_text(filename):
    filepath = os.path.join(config.KNOWLEDGE_DIR, filename)
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _classify_disease_names(candidate_names):
    system = """你是一个农业知识图谱专家。从给定的章节标题列表中，判断哪些是病害名或虫害名。

  规则:
  - 病害名如: 叶斑病、灰霉病、青枯病、番茄筋腐病、根结线虫 等
  - 虫害名如: 棉铃虫、蚜虫、烟青虫、二斑叶螨、天幕毛虫 等
  - 不是病害/虫害的标题: 主要病害、主要虫害、病害、虫害、害虫、危害、症状、
    防治方法、传播途径、简介、产生原因、肥料危害、风害、涝害 等泛化章节标签
  - 不是病害/虫害的危害描述: 肥料危害、药物危害、涝害、高温危害 等非生物胁迫

  只返回一个JSON数组，不要任何其他文字:
  ["病害名1", "虫害名1", "病害名2", ...]"""

    user = "标题列表:\n" + "\n".join(f"- {n}" for n in candidate_names)
    result = _call_extraction_llm(system, user, max_tokens=512)
    try:
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r"^```(?:json)?\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
        return json.loads(result)
    except json.JSONDecodeError:
        return []


def _heading_depth(number_str):
    return number_str.count(".") + 1


def _heading_fallback_filter(candidate_names):
    """LLM不可用时的规则回退：长度和通用后缀过滤。"""
    skip_suffixes = (
        "方法", "途径", "症状", "简介", "原因", "条件",
        "规律", "时期", "特性", "形态", "习性", "特点",
        "特征", "指标", "设施", "标准", "类型", "危害",
    )
    result = []
    for name in candidate_names:
        if len(name) > 12 or len(name) < 2:
            continue
        if name.endswith(skip_suffixes):
            continue
        result.append(name)
    return result


def _split_disease_sections(text):
    heading_re = re.compile(r"^(\d+(?:\.\d+)+)\s*[、.]?\s*(.+)")
    lines = text.split("\n")

    candidates = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = heading_re.match(stripped)
        if m:
            name = m.group(2).strip()
            if 2 <= len(name) <= 15:
                candidates.append((i, stripped, m.group(1), name))

    if not candidates:
        return []

    unique_names = list(dict.fromkeys(name for _, _, _, name in candidates))
    disease_names = set(_classify_disease_names(unique_names))
    if not disease_names:
        disease_names = set(_heading_fallback_filter(unique_names))
    if not disease_names:
        return []

    sections = []
    current_name = None
    current_lines = []
    current_depth = 0

    idx_to_name = {idx: name for idx, _, _, name in candidates}
    idx_is_disease = {idx: name in disease_names for idx, name in idx_to_name.items()}
    idx_to_depth = {idx: _heading_depth(num) for idx, _, num, _ in candidates}

    for i, line in enumerate(lines):
        stripped = line.strip()

        if i in idx_is_disease:
            is_disease = idx_is_disease[i]
            depth = idx_to_depth[i]
            name = idx_to_name[i]

            if is_disease:
                if current_name and current_lines:
                    sections.append({
                        "name": current_name,
                        "full_text": "\n".join(current_lines)
                    })
                current_name = name
                current_lines = [stripped]
                current_depth = depth
            elif current_name and depth <= current_depth:
                if current_lines:
                    sections.append({
                        "name": current_name,
                        "full_text": "\n".join(current_lines)
                    })
                current_name = None
                current_lines = []
                current_depth = 0
            elif current_name:
                current_lines.append(stripped)
        elif current_name:
            current_lines.append(stripped)

    if current_name and current_lines:
        sections.append({
            "name": current_name,
            "full_text": "\n".join(current_lines)
        })

    return sections


def _extract_relations_llm(disease_name, disease_text):
    system_prompt = """你是一个农业知识图谱抽取引擎。从给定病害文本中提取结构化信息，严格按JSON格式输出。

输出格式(只输出JSON，不要任何其他文字):
{
  "pathogen_type": "真菌/细菌/病毒/线虫/生理性/未知",
  "symptoms": ["症状1", "症状2"],
  "transmission_modes": ["传播途径1"],
  "control_agricultural": ["农业防治措施1"],
  "control_chemical": ["化学防治措施1"],
  "pesticides": [{"name": "药剂名", "concentration": "浓度", "usage": "用法"}],
  "related_diseases": [{"name": "关联病害名", "relation": "同治法/同传播/并发/同宿主"}]
}

注意:
1. pesticides只提取具体药剂名称(如戊唑醇、多菌灵)，不要泛泛描述
2. 如果某字段无信息，返回空数组[]
3. 只输出JSON，禁止任何其他文字"""

    user_prompt = f"病害名称: {disease_name}\n\n文本内容:\n{disease_text[:3000]}"

    result = _call_extraction_llm(system_prompt, user_prompt, max_tokens=1536)

    try:
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r"^```(?:json)?\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "pathogen_type": "未知",
            "symptoms": [],
            "transmission_modes": [],
            "control_agricultural": [],
            "control_chemical": [],
            "pesticides": [],
            "related_diseases": []
        }


def _generate_cypher(disease_name, source_file, extracted):
    statements = []
    safe_name = disease_name.replace("'", "\\'").replace('"', '\\"')
    pathogen = (extracted.get("pathogen_type") or "未知").replace("'", "\\'")

    statements.append(
        f"MERGE (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
        f"SET d.病原类型 = '{pathogen}', "
        f"d.来源 = '{source_file}'"
    )

    for symptom in extracted.get("symptoms", []):
        safe_symptom = symptom[:500].replace("'", "\\'").replace('"', '\\"')
        statements.append(
            f"MERGE (s:症状 {{描述: '{safe_symptom}'}}) "
            f"WITH s "
            f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
            f"MERGE (d)-[:有症状]->(s)"
        )

    for mode in extracted.get("transmission_modes", []):
        safe_mode = mode[:300].replace("'", "\\'").replace('"', '\\"')
        statements.append(
            f"MERGE (t:传播途径 {{方式: '{safe_mode}'}}) "
            f"WITH t "
            f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
            f"MERGE (d)-[:传播方式]->(t)"
        )

    for control in extracted.get("control_agricultural", []):
        safe_control = control[:500].replace("'", "\\'").replace('"', '\\"')
        statements.append(
            f"MERGE (c:防治方法 {{描述: '{safe_control}', 类型: 'agricultural'}}) "
            f"WITH c "
            f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
            f"MERGE (d)-[:防治措施]->(c)"
        )

    for control in extracted.get("control_chemical", []):
        safe_control = control[:500].replace("'", "\\'").replace('"', '\\"')
        statements.append(
            f"MERGE (c:防治方法 {{描述: '{safe_control}', 类型: 'chemical'}}) "
            f"WITH c "
            f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
            f"MERGE (d)-[:防治措施]->(c)"
        )

    for pesticide in extracted.get("pesticides", []):
        p_name = pesticide.get("name", "").replace("'", "\\'").replace('"', '\\"')
        if not p_name or len(p_name) > 50:
            continue
        p_conc = (pesticide.get("concentration") or "").replace("'", "\\'").replace('"', '\\"')
        p_usage = (pesticide.get("usage") or "").replace("'", "\\'").replace('"', '\\"')
        statements.append(
            f"MERGE (p:药剂 {{名称: '{p_name}'}}) "
            f"SET p.浓度 = '{p_conc}', p.用法 = '{p_usage}' "
            f"WITH p "
            f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
            f"MERGE (d)-[:用药]->(p)"
        )

    for related in extracted.get("related_diseases", []):
        r_name = related.get("name", "").replace("'", "\\'").replace('"', '\\"')
        r_rel = (related.get("relation") or "关联").replace("'", "\\'")
        if not r_name or len(r_name) > 50:
            continue
        statements.append(
            f"MERGE (rd:{GRAPH_DISEASE_LABEL} {{名称: '{r_name}'}}) "
            f"WITH rd "
            f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: '{safe_name}'}}) "
            f"MERGE (d)-[:关联 {{原因: '{r_rel}'}}]->(rd)"
        )

    return statements


def _auto_link_related_diseases(driver):
    auto_linked = 0

    try:
        with driver.session() as session:
            result = session.run(
                f"MATCH (d1:{GRAPH_DISEASE_LABEL})-[:用药]->(p:药剂)"
                f"<-[:用药]-(d2:{GRAPH_DISEASE_LABEL})"
                f"WHERE d1.名称 < d2.名称 AND NOT (d1)-[:关联]->(d2)"
                f"MERGE (d1)-[:关联 {{原因: '同药剂'}}]->(d2)"
                f"RETURN count(*) AS cnt"
            )
            count = result.single()["cnt"]
            auto_linked += count
            if count > 0:
                print(f"  [自动关联] 同药剂 → {count} 对病害")
    except Exception as e:
        print(f"  [自动关联] 同药剂规则执行失败: {e}")

    try:
        with driver.session() as session:
            result = session.run(
                f"MATCH (d1:{GRAPH_DISEASE_LABEL}), (d2:{GRAPH_DISEASE_LABEL})"
                f"WHERE d1.名称 < d2.名称 "
                f"AND d1.病原类型 = d2.病原类型 "
                f"AND d1.病原类型 <> '未知' AND d1.病原类型 IS NOT NULL "
                f"AND NOT (d1)-[:关联]->(d2)"
                f"MERGE (d1)-[:关联 {{原因: '同病原类型'}}]->(d2)"
                f"RETURN count(*) AS cnt"
            )
            count = result.single()["cnt"]
            auto_linked += count
            if count > 0:
                print(f"  [自动关联] 同病原类型 → {count} 对病害")
    except Exception as e:
        print(f"  [自动关联] 同病原类型规则执行失败: {e}")

    return auto_linked


def extract_and_build_graph(clear_first=False):
    driver = get_neo4j_driver()
    if driver is None:
        print("[图谱] Neo4j 未连接，跳过知识图谱构建")
        return 0

    if clear_first:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[图谱] 已清空旧图谱数据")

    with driver.session() as session:
        session.run("CREATE CONSTRAINT 病害名称唯一 IF NOT EXISTS "
                     f"FOR (d:{GRAPH_DISEASE_LABEL}) REQUIRE d.名称 IS UNIQUE")
        session.run("CREATE CONSTRAINT 药剂名称唯一 IF NOT EXISTS "
                     "FOR (p:药剂) REQUIRE p.名称 IS UNIQUE")
        session.run("CREATE CONSTRAINT 症状描述唯一 IF NOT EXISTS "
                     "FOR (s:症状) REQUIRE s.描述 IS UNIQUE")
        session.run("CREATE CONSTRAINT 传播途径方式唯一 IF NOT EXISTS "
                     "FOR (t:传播途径) REQUIRE t.方式 IS UNIQUE")
        session.run("CREATE CONSTRAINT 防治方法描述唯一 IF NOT EXISTS "
                     "FOR (c:防治方法) REQUIRE c.描述 IS UNIQUE")

    all_disease_sections = []
    for filename in sorted(os.listdir(config.KNOWLEDGE_DIR)):
        if not filename.endswith(".txt"):
            continue
        text = _load_knowledge_text(filename)
        if not text:
            continue
        sections = _split_disease_sections(text)
        for s in sections:
            s["source"] = filename
        all_disease_sections.extend(sections)
        print(f"[图谱] {filename}: 识别到 {len(sections)} 个病害/虫害段落")

    before_merge = len(all_disease_sections)
    all_disease_sections = _merge_duplicate_sections(all_disease_sections)
    after_merge = len(all_disease_sections)
    if before_merge != after_merge:
        print(f"\n[图谱] 合并同名段落: {before_merge} → {after_merge} (去重 {before_merge - after_merge} 个)")

    print(f"\n[图谱] 共 {len(all_disease_sections)} 个病害/虫害段落，开始LLM抽取关系...")

    total_statements = 0
    with driver.session() as session:
        for i, ds in enumerate(all_disease_sections):
            name = ds["name"]
            source = ds["source"]
            text = ds["full_text"]
            print(f"  [{i+1}/{len(all_disease_sections)}] {name} ({source}) ...", end=" ")
            extracted = _extract_relations_llm(name, text)
            statements = _generate_cypher(name, source, extracted)
            for stmt in statements:
                try:
                    session.run(stmt)
                    total_statements += 1
                except Exception as e:
                    print(f"\n    [警告] Cypher 执行失败: {e}")
            print(f"完成 ({len(statements)} 条关系)")

    if total_statements > 0:
        auto_linked = _auto_link_related_diseases(driver)
        print(f"[图谱] 自动发现 {auto_linked} 条跨病害关联关系")

    total_executed = total_statements + (auto_linked if total_statements > 0 else 0)
    print(f"\n[图谱] 知识图谱构建完成，共执行 {total_executed} 条Cypher语句")
    return total_statements


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="农业知识图谱抽取工具")
    parser.add_argument("--clear", action="store_true", help="清空旧图数据后重建")
    parser.add_argument("--test", action="store_true", help="仅抽取前5个病害做测试")
    args = parser.parse_args()

    if args.test:
        print("=== 测试模式: 仅处理前5个病害段落 ===")
        for filename in sorted(os.listdir(config.KNOWLEDGE_DIR)):
            if not filename.endswith(".txt"):
                continue
            text = _load_knowledge_text(filename)
            if not text:
                continue
            sections = _split_disease_sections(text)
            for s in sections[:5]:
                s["source"] = filename
                print(f"\n病害: {s['name']} ({s['source']})")
                print(f"文本长度: {len(s['full_text'])} 字符")
                extracted = _extract_relations_llm(s['name'], s['full_text'])
                print(f"抽取结果: {json.dumps(extracted, ensure_ascii=False, indent=2)}")
                stmts = _generate_cypher(s['name'], s['source'], extracted)
                print(f"生成 {len(stmts)} 条Cypher语句:")
                for st in stmts:
                    print(f"  {st}")
            break
    else:
        extract_and_build_graph(clear_first=args.clear)
