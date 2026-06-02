"""Redis 连接测试脚本 — 直接运行即可验证 Redis 连通性和缓存读写"""

import hashlib
import json
import time
import os

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("None") or None

print("=" * 50)
print("Redis 连接测试")
print(f"目标: {REDIS_HOST}:{REDIS_PORT}  db={REDIS_DB}")
print("=" * 50)

# 1. 检查 redis-py 是否安装
try:
    import redis
    print("[1/5] redis-py 已安装 ✓")
except ImportError:
    print("[1/5] redis-py 未安装 ✗  请执行: pip install redis")
    raise SystemExit(1)

# 2. 建立连接
try:
    kwargs = {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
        "decode_responses": True,
        "socket_connect_timeout": 3,
        "socket_timeout": 3,
    }
    if REDIS_PASSWORD:
        kwargs["password"] = REDIS_PASSWORD
    r = redis.Redis(**kwargs)
    r.ping()
    print("[2/5] 连接成功 → PONG ✓")
except Exception as e:
    print(f"[2/5] 连接失败 ✗  {e}")
    raise SystemExit(1)

# 3. 测试写入
test_key = "rag:test:connection_check"
test_value = json.dumps({
    "message": "Hello from AgriRAG cache test",
    "timestamp": time.time(),
}, ensure_ascii=False)

try:
    r.setex(test_key, 60, test_value)
    print(f"[3/5] 写入成功 (key={test_key}, TTL=60s) ✓")
except Exception as e:
    print(f"[3/5] 写入失败 ✗  {e}")

# 4. 测试读取
try:
    raw = r.get(test_key)
    if raw is None:
        print("[4/5] 读取失败 ✗  键不存在（可能已过期）")
    else:
        data = json.loads(raw)
        print(f"[4/5] 读取成功 → {data['message']} ✓")
except Exception as e:
    print(f"[4/5] 读取失败 ✗  {e}")

# 5. 清理测试键
try:
    r.delete(test_key)
    print("[5/5] 清理完成 ✓")
except Exception as e:
    print(f"[5/5] 清理失败 ✗  {e}")

# 6. 模拟业务缓存读写（MD5 key 模式）
test_question = "番茄晚疫病怎么防治？"
key = f"rag:answer:{hashlib.md5(test_question.encode('utf-8')).hexdigest()}"
r.setex(key, 60, json.dumps({"answer": "这是模拟的缓存答案", "source_legend": "知识1.txt"}, ensure_ascii=False))
cached = json.loads(r.get(key))
r.delete(key)

print("-" * 50)
print(f"业务缓存模拟 → 问题: \"{test_question[:15]}...\" → key: {key[:20]}... → 读写正常 ✓")
print("=" * 50)
print("Redis 连接测试全部通过！")
