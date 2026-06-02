import hashlib
import json
import time

import config

_redis_client = None
_redis_available = True

ANSWER_PREFIX = "rag:answer:"


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


def _make_key(question):
    q = question.strip()
    h = hashlib.md5(q.encode("utf-8")).hexdigest()
    return f"{ANSWER_PREFIX}{h}"


def get_cached_answer(question):
    r = _get_redis()
    if r is None:
        return None
    try:
        key = _make_key(question)
        raw = r.get(key)
        if raw is None:
            return None
        cached = json.loads(raw)
        return cached
    except Exception as e:
        print(f"[缓存] 读取失败: {e}")
        return None


def set_cached_answer(question, answer, source_legend):
    r = _get_redis()
    if r is None:
        return
    try:
        key = _make_key(question)
        data = json.dumps({
            "answer": answer,
            "source_legend": source_legend,
            "cached_at": time.time(),
        }, ensure_ascii=False)
        r.setex(key, config.REDIS_TTL, data)
    except Exception as e:
        print(f"[缓存] 写入失败: {e}")


def invalidate_all():
    r = _get_redis()
    if r is None:
        return
    try:
        keys = list(r.scan_iter(f"{ANSWER_PREFIX}*"))
        if keys:
            r.delete(*keys)
            print(f"[缓存] 已清除 {len(keys)} 条答案缓存")
    except Exception as e:
        print(f"[缓存] 清除失败: {e}")
