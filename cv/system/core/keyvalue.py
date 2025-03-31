# adapted from https://github.com/tinygrad/tinygrad/blob/7ef02d0e1c7d35b536d935ad9da6ba5d8b619ecf/tinygrad/helpers.py#L169

import sqlite3, contextlib
from typing import Any

import cbor2

from .. import SYSTEM_PATH

KVPATH = SYSTEM_PATH / "keyvalue.db"

VERSION = 0
_kv_connection = None
def kv_connection():
  global _kv_connection
  if _kv_connection is None:
    _kv_connection = sqlite3.connect(KVPATH, timeout=2, isolation_level="IMMEDIATE")
    # another connection has set it already or is in the process of setting it
    # that connection will lock the database
    with contextlib.suppress(sqlite3.OperationalError): _kv_connection.execute("PRAGMA journal_mode=WAL").fetchone()
  return _kv_connection

def kv_clear(table:str):
  cur = kv_connection().cursor()
  cur.executescript(f"DROP TABLE IF EXISTS '{table}_{VERSION}';" + "VACUUM;")

def kv_get(table:str, key:Any) -> Any:
  cur = kv_connection().cursor()
  try:
    res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE key = ?", cbor2.dumps(key))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return cbor2.loads(val[0])
  return None

def kv_getall(table:str) -> dict:
  cur = kv_connection().cursor()
  try:
    res = cur.execute(f"SELECT * FROM '{table}_{VERSION}'")
  except sqlite3.OperationalError:
    return {}  # table doesn't exist
  return {cbor2.loads(row[0]): cbor2.loads(row[1]) for row in res.fetchall()}

_db_tables = set()
def kv_put(table:str, key:Any, val:Any):
  conn = kv_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' (key blob, val blob, PRIMARY KEY (key))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO '{table}_{VERSION}' (key, val) VALUES (?, ?)", (cbor2.dumps(key), cbor2.dumps(val)))
  conn.commit()
  cur.close()
  return val
