import os
import re
import config

GRAPH_DISEASE_LABEL = "病害"

_neo4j_driver = None
_disease_names_cache = None
_pesticide_names_cache = None
_symptom_descs_cache = None


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    try:
        from neo4j import GraphDatabase
        uri = config.NEO4J_URI
        auth = (config.NEO4J_USER, config.NEO4J_PASSWORD) if config.NEO4J_USER else None
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        _neo4j_driver = driver
        print(f"[图谱] Neo4j 连接成功: {uri}")
        return _neo4j_driver
    except ImportError:
        print("[图谱] neo4j 驱动未安装 (pip install neo4j)")
        return None
    except Exception as e:
        print(f"[图谱] Neo4j 连接失败: {e}")
        return None


def _load_disease_names():
    global _disease_names_cache
    if _disease_names_cache is not None:
        return _disease_names_cache

    driver = get_neo4j_driver()
    if driver is None:
        _disease_names_cache = []
        return _disease_names_cache

    with driver.session() as session:
        result = session.run(
            f"MATCH (d:{GRAPH_DISEASE_LABEL}) RETURN d.名称 AS name ORDER BY d.名称"
        )
        _disease_names_cache = [r["name"] for r in result]

    print(f"[图谱] 已加载 {len(_disease_names_cache)} 个病害名称")
    return _disease_names_cache


def _load_pesticide_names():
    global _pesticide_names_cache
    if _pesticide_names_cache is not None:
        return _pesticide_names_cache

    driver = get_neo4j_driver()
    if driver is None:
        _pesticide_names_cache = []
        return _pesticide_names_cache

    with driver.session() as session:
        result = session.run("MATCH (p:药剂) RETURN p.名称 AS name ORDER BY p.名称")
        _pesticide_names_cache = [r["name"] for r in result]

    print(f"[图谱] 已加载 {len(_pesticide_names_cache)} 个药剂名称")
    return _pesticide_names_cache


def _load_symptom_descs():
    global _symptom_descs_cache
    if _symptom_descs_cache is not None:
        return _symptom_descs_cache

    driver = get_neo4j_driver()
    if driver is None:
        _symptom_descs_cache = []
        return _symptom_descs_cache

    with driver.session() as session:
        result = session.run("MATCH (s:症状) RETURN DISTINCT s.描述 AS desc ORDER BY s.描述")
        _symptom_descs_cache = [r["desc"] for r in result if r["desc"]]

    print(f"[图谱] 已加载 {len(_symptom_descs_cache)} 条症状描述")
    return _symptom_descs_cache


def _match_entities(query):
    disease_names = _load_disease_names()
    pesticide_names = _load_pesticide_names()
    symptom_descs = _load_symptom_descs()

    result = {"diseases": [], "pesticides": [], "symptoms": []}

    if disease_names:
        disease_names.sort(key=lambda x: len(x), reverse=True)
        matched = set()
        for name in disease_names:
            if name in query:
                matched.add(name)
        result["diseases"] = _dedupe_substrings(matched)

    if pesticide_names:
        pesticide_names.sort(key=lambda x: len(x), reverse=True)
        matched = set()
        for name in pesticide_names:
            if name in query:
                matched.add(name)
        result["pesticides"] = _dedupe_substrings(matched)

    if symptom_descs:
        symptom_descs.sort(key=lambda x: len(x), reverse=True)
        matched = set()
        for desc in symptom_descs:
            if desc in query:
                matched.add(desc)
        result["symptoms"] = _dedupe_substrings(matched)

    return result


def _dedupe_substrings(matched_set):
    if len(matched_set) <= 1:
        return list(matched_set)
    result = []
    for name in matched_set:
        if not any(len(other) > len(name) and name in other for other in matched_set):
            result.append(name)
    return result


def _build_disease_doc(disease_name, pathogen, source, symptom_list,
                       pesticide_list, transmission_list, control_list, related_list):
    text2_parts = [f"病害名称: {disease_name}"]
    if pathogen and pathogen != "未知":
        text2_parts.append(f"病原类型: {pathogen}")

    symptoms = [s for s in (symptom_list or []) if s]
    if symptoms:
        text2_parts.append(f"症状: {'; '.join(symptoms)}")

    transmissions = [t for t in (transmission_list or []) if t]
    if transmissions:
        text2_parts.append(f"传播途径: {'; '.join(transmissions)}")

    pesticides = [p for p in (pesticide_list or []) if p and p.get("名称")]
    if pesticides:
        pest_strs = []
        for p in pesticides:
            parts = [p["名称"]]
            if p.get("浓度"):
                parts.append(p["浓度"])
            if p.get("用法"):
                parts.append(p["用法"])
            pest_strs.append(" ".join(parts))
        text2_parts.append(f"防治药剂: {'; '.join(pest_strs)}")

    controls = [c for c in (control_list or []) if c and c.get("描述")]
    if controls:
        agri = [c["描述"] for c in controls if c.get("类型") == "agricultural"]
        chem = [c["描述"] for c in controls if c.get("类型") == "chemical"]
        if agri:
            text2_parts.append(f"农业防治: {'; '.join(agri[:3])}")
        if chem:
            text2_parts.append(f"化学防治: {'; '.join(chem[:3])}")

    related = [r for r in (related_list or []) if r and r.get("名称")]
    if related:
        rel_strs = [f"{r['名称']}({r.get('原因', '关联')})" for r in related[:5]]
        text2_parts.append(f"关联病害: {'; '.join(rel_strs)}")

    text2 = "\n".join(text2_parts)
    text1 = f"病害:{disease_name}"
    if symptoms:
        text1 += f" 症状:{(symptoms[0])[:40]}"

    return {
        "id": f"graph:{disease_name}",
        "text1": text1[:200],
        "text2": text2[:8000],
        "text": text2[:8000],
        "source": f"知识图谱/{source}",
        "score": 1.0,
    }


def _batch_serialize_disease_subgraphs(session, disease_names):
    if not disease_names:
        return {}

    query_cypher = (
        f"UNWIND $disease_names AS name "
        f"MATCH (d:{GRAPH_DISEASE_LABEL} {{名称: name}}) "
        f"RETURN d.名称 AS disease_name, "
        f"d.病原类型 AS pathogen, "
        f"d.来源 AS source, "
        f"[(d)-[:有症状]->(s:症状) | s.描述] AS symptoms, "
        f"[(d)-[:用药]->(p:药剂) | {{名称: p.名称, 浓度: p.浓度, 用法: p.用法}}] AS pesticides, "
        f"[(d)-[:传播方式]->(t:传播途径) | t.方式] AS transmissions, "
        f"[(d)-[:防治措施]->(c:防治方法) | {{描述: c.描述, 类型: c.类型}}] AS controls, "
        f"[(d)-[:关联]->(rd:{GRAPH_DISEASE_LABEL}) | {{名称: rd.名称, 原因: rd.原因}}] AS related"
    )

    docs = {}
    for record in session.run(query_cypher, disease_names=list(disease_names)):
        name = record["disease_name"]
        docs[name] = _build_disease_doc(
            disease_name=name,
            pathogen=record.get("pathogen", "未知"),
            source=record.get("source", ""),
            symptom_list=record.get("symptoms"),
            pesticide_list=record.get("pesticides"),
            transmission_list=record.get("transmissions"),
            control_list=record.get("controls"),
            related_list=record.get("related"),
        )
    return docs


def _serialize_disease_subgraph(session, disease_name):
    docs = _batch_serialize_disease_subgraphs(session, [disease_name])
    return docs.get(disease_name)


def _batch_reverse_lookup(session, pesticide_names, symptom_descs):
    disease_names = set()
    parts = []
    params = {}

    if pesticide_names:
        params["pesticide_names"] = list(pesticide_names)
        parts.append(
            "MATCH (d:病害)-[:用药]->(p:药剂) "
            "WHERE p.名称 IN $pesticide_names "
            "RETURN DISTINCT d.名称 AS disease_name"
        )

    if symptom_descs:
        params["symptom_descs"] = list(symptom_descs)
        parts.append(
            "UNWIND $symptom_descs AS symptom_desc "
            "MATCH (d:病害)-[:有症状]->(s:症状) "
            "WHERE s.描述 CONTAINS symptom_desc "
            "RETURN DISTINCT d.名称 AS disease_name"
        )

    if not parts:
        return disease_names

    cypher = " UNION ".join(parts)
    for rec in session.run(cypher, params):
        disease_names.add(rec["disease_name"])
    return disease_names


def search(query, top_k=None):
    if not config.GRAPH_RECALL_ENABLED:
        return []

    if top_k is None:
        top_k = config.GRAPH_RECALL_K

    matched = _match_entities(query)
    disease_matches = matched.get("diseases", [])
    pesticide_matches = matched.get("pesticides", [])
    symptom_matches = matched.get("symptoms", [])

    if not disease_matches and not pesticide_matches and not symptom_matches:
        return []

    entity_parts = []
    if disease_matches:
        entity_parts.append(f"病害:{disease_matches}")
    if pesticide_matches:
        entity_parts.append(f"药剂:{pesticide_matches}")
    if symptom_matches:
        entity_parts.append(f"症状:{[s[:20] for s in symptom_matches]}")
    print(f"  [图谱] 识别到实体: {', '.join(entity_parts)}")

    driver = get_neo4j_driver()
    if driver is None:
        return []

    all_disease_names = set(disease_matches)

    with driver.session() as session:
        if pesticide_matches or symptom_matches:
            rev_diseases = _batch_reverse_lookup(
                session, pesticide_matches, symptom_matches
            )
            all_disease_names.update(rev_diseases)

        docs_map = _batch_serialize_disease_subgraphs(
            session, all_disease_names
        )

    results = []
    for disease_name in all_disease_names:
        doc = docs_map.get(disease_name)
        if doc:
            if disease_name not in disease_matches and pesticide_matches:
                matching_pests = [
                    p for p in pesticide_matches
                    if p in doc.get("text2", "")
                ]
                if matching_pests:
                    pest_label = "/".join(matching_pests)
                    doc["text2"] = (
                        f"[药剂反查: {pest_label} 可防治此病害]\n"
                        f"{doc['text2']}"
                    )
            elif disease_name not in disease_matches and symptom_matches:
                matching_symptoms = [
                    s for s in symptom_matches
                    if s in doc.get("text2", "")
                ]
                if matching_symptoms:
                    symptom_label = "; ".join(m[:30] for m in matching_symptoms)
                    doc["text2"] = (
                        f"[症状反查: {symptom_label}]\n"
                        f"{doc['text2']}"
                    )
            results.append(doc)

    if results:
        print(f"  [图谱] 召回 {len(results)} 条图检索结果")
    else:
        print(f"  [图谱] 未检索到图结果")

    return results[:top_k]


def rebuild_cache():
    global _disease_names_cache, _pesticide_names_cache, _symptom_descs_cache
    _disease_names_cache = None
    _pesticide_names_cache = None
    _symptom_descs_cache = None
    _load_disease_names()
    _load_pesticide_names()
    _load_symptom_descs()


if __name__ == "__main__":
    test_queries = [
        "樱桃叶斑病如何防治？",
        "灰霉病用什么药？",
        "戊唑醇能治什么病？",
        "石硫合剂能防治哪些病害？",
        "番茄晚疫病的症状是什么？",
        "叶片上出现黄色斑点是什么病害？",
        "果实腐烂变软是什么病？",
    ]
    for q in test_queries:
        print(f"\n查询: {q}")
        results = search(q)
        for r in results:
            print(f"  → {r['text2'][:200]}...")
