# adapted from https://github.com/tinygrad/tinygrad/blob/7ef02d0e1c7d35b536d935ad9da6ba5d8b619ecf/tinygrad/helpers.py#L169

import sqlite3, contextlib, time
from typing import Any

import cbor2

from .. import SYSTEM_PATH, PERSIST_PATH

EPHEMERAL_DB = SYSTEM_PATH / "keyvalue.db"
PERSIST_DB = PERSIST_PATH / "persist.db"
PERSISTED_TABLES: set[str] = set()
def kv_persist(table:str): PERSISTED_TABLES.add(table)

_connections: dict = {}
def kv_connection(table=None):
  path = PERSIST_DB if table in PERSISTED_TABLES else EPHEMERAL_DB
  conn = _connections.get(path)
  if conn is None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    with contextlib.suppress(sqlite3.OperationalError): conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.execute("PRAGMA busy_timeout=2000")
    _connections[path] = conn
  return conn

def _retry_locked(op):
  delay = 0.005
  for i in range(12):
    try:
      return op()
    except sqlite3.OperationalError as e:
      if "locked" not in str(e) or i == 11: raise
      time.sleep(delay)
      delay = min(delay * 2, 0.2)

def kv_reset():
  global _connections, _db_tables
  _connections = {}
  _db_tables = set()

def kv_clear(table:str):
  kv_connection(table).execute(f"DROP TABLE IF EXISTS '{table}'")

def kv_checkpoint():
  for conn in _connections.values():
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

def kv_get(table:str, key:Any) -> Any:
  def op():
    cur = kv_connection(table).cursor()
    try:
      # fetchall drains the statement so no read txn is left open on the shared connection
      res = cur.execute(f"SELECT val FROM '{table}' WHERE key = ?", (cbor2.dumps(key),)).fetchall()
    except sqlite3.OperationalError as e:
      if "no such table" in str(e): return None  # table doesn't exist
      raise
    finally:
      cur.close()
    return cbor2.loads(res[0][0]) if res else None
  return _retry_locked(op)

def kv_getall(table:str) -> dict:
  def op():
    cur = kv_connection(table).cursor()
    try:
      res = cur.execute(f"SELECT * FROM '{table}'").fetchall()
    except sqlite3.OperationalError as e:
      if "no such table" in str(e): return {}  # table doesn't exist
      raise
    finally:
      cur.close()
    return {cbor2.loads(row[0]): cbor2.loads(row[1]) for row in res}
  return _retry_locked(op)

_db_tables = set()
def kv_put(table:str, key:Any, val:Any):
  def op():
    cur = kv_connection(table).cursor()
    try:
      if table not in _db_tables:
        cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}' (key blob, val blob, PRIMARY KEY (key))")
        _db_tables.add(table)
      cur.execute(f"REPLACE INTO '{table}' (key, val) VALUES (?, ?)", (cbor2.dumps(key), cbor2.dumps(val)))
    finally:
      cur.close()
    return val
  return _retry_locked(op)
