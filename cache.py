import hashlib
import json
import os
import csv
import time

import config

_redis_client = None
_redis_available = True

ANSWER_PREFIX = "rag:answer:"
COUNTER_PREFIX = "rag:counter:"


def _ensure_cache_dir():
    os.makedirs(config.CACHE_CSV_DIR, exist_ok=True)


def _get_redis():
    global _redis_client, _redis_available
    if not _redis_available or not config.REDIS_CACHE_ENABLED:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        kwargs = {
            "host": config.REDIS_HOST,
            "port": config.REDIS_PORT,
            "db": config.REDIS_DB,
            "decode_responses": True,
            "socket_connect_timeout": 2,
            "socket_timeout": 2,
        }
        if config.REDIS_PASSWORD:
            kwargs["password"] = config.REDIS_PASSWORD
        _redis_client = redis.Redis(**kwargs)
        _redis_client.ping()
        print(f"[缓存] Redis 已连接 ({config.REDIS_HOST}:{config.REDIS_PORT}, db={config.REDIS_DB})")
        return _redis_client
    except ImportError:
        _redis_available = False
        return None
    except Exception as e:
        print(f"[缓存] Redis 不可用 ({config.REDIS_HOST}:{config.REDIS_PORT}): {e}")
        _redis_available = False
        return None


def _make_answer_key(question):
    h = hashlib.md5(question.strip().encode("utf-8")).hexdigest()
    return f"{ANSWER_PREFIX}{h}"


def _make_counter_key(question):
    h = hashlib.md5(question.strip().encode("utf-8")).hexdigest()
    return f"{COUNTER_PREFIX}{h}"


def get_cached_answer(question):
    r = _get_redis()
    if r is None:
        return None
    try:
        key = _make_answer_key(question)
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        print(f"[缓存] 读取失败: {e}")
        return None


def increment_counter(question):
    r = _get_redis()
    if r is None:
        return 0
    try:
        key = _make_counter_key(question)
        count = r.incr(key)
        r.expire(key, config.REDIS_TTL * 2, nx=False)
        return count
    except Exception as e:
        print(f"[缓存] 计数器更新失败: {e}")
        return 0


def get_count(question):
    r = _get_redis()
    if r is None:
        return 0
    try:
        key = _make_counter_key(question)
        val = r.get(key)
        return int(val) if val else 0
    except Exception:
        return 0


def set_cached_answer(question, answer, source_legend):
    r = _get_redis()
    if r is None:
        return
    try:
        key = _make_answer_key(question)
        data = json.dumps({
            "question": question,
            "answer": answer,
            "source_legend": source_legend,
            "cached_at": time.time(),
        }, ensure_ascii=False)
        r.setex(key, config.REDIS_TTL, data)
        _sync_csv()
        print(f"[缓存] 已写入 Redis 并同步 CSV")
    except Exception as e:
        print(f"[缓存] 写入失败: {e}")


def _sync_csv():
    _ensure_cache_dir()
    r = _get_redis()
    if r is None:
        return
    csv_path = os.path.join(config.CACHE_CSV_DIR, "faq_answers.csv")
    try:
        rows = []
        for key in r.scan_iter(f"{ANSWER_PREFIX}*"):
            raw = r.get(key)
            if raw is None:
                continue
            try:
                cached = json.loads(raw)
                q = cached.get("question", "")
                a = cached.get("answer", "")
                if q:
                    rows.append((q, a))
            except json.JSONDecodeError:
                continue

        rows.sort(key=lambda x: x[0])

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["经常被提及的问题", "最后一次回答的答案"])
            for q, a in rows:
                writer.writerow([q, a])
    except Exception as e:
        print(f"[缓存] CSV 同步失败: {e}")


def invalidate_all():
    r = _get_redis()
    if r is None:
        return
    try:
        answer_keys = list(r.scan_iter(f"{ANSWER_PREFIX}*"))
        counter_keys = list(r.scan_iter(f"{COUNTER_PREFIX}*"))
        all_keys = answer_keys + counter_keys
        if all_keys:
            r.delete(*all_keys)
            print(f"[缓存] 已清除 {len(answer_keys)} 条答案缓存 + {len(counter_keys)} 条计数")
        _sync_csv()
    except Exception as e:
        print(f"[缓存] 清除失败: {e}")
