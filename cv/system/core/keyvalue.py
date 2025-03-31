# adapted from https://github.com/tinygrad/tinygrad/blob/7ef02d0e1c7d35b536d935ad9da6ba5d8b619ecf/tinygrad/helpers.py#L169

import sqlite3, contextlib, pickle
from typing import Any

from .. import SYSTEM_PATH

KVPATH = SYSTEM_PATH / "keyvalue.db"

VERSION = 0
_kv_connection = None
def kv_connection():
  global _kv_connection
  if _kv_connection is None:
    _kv_connection = sqlite3.connect(KVPATH, timeout=5, isolation_level="IMMEDIATE")
    # another connection has set it already or is in the process of setting it
    # that connection will lock the database
    with contextlib.suppress(sqlite3.OperationalError): _kv_connection.execute("PRAGMA journal_mode=WAL").fetchone()
  return _kv_connection

def kv_clear():
  cur = kv_connection().cursor()
  drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
  cur.executescript("\n".join([s[0] for s in drop_tables] + ["VACUUM;"]))

def kv_get(table:str, key:dict|str|int) -> Any:
  if isinstance(key, (str,int)): key = {"key": key}
  cur = kv_connection().cursor()
  try:
    res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return None

def kv_getall(table:str) -> dict:
  cur = kv_connection().cursor()
  try:
    res = cur.execute(f"SELECT * FROM '{table}_{VERSION}'")
  except sqlite3.OperationalError:
    return {}  # table doesn't exist
  return {tuple(row[:-1]): pickle.loads(row[-1]) for row in res.fetchall()}

_db_tables = set()
def kv_put(table:str, key:dict|str|int, val:Any, prepickled=False):
  if isinstance(key, (str,int)): key = {"key": key}
  conn = kv_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  vals = (pickle.dumps(val) if not prepickled else val,)
  cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key))}, ?)", tuple(key.values()) + vals)
  conn.commit()
  cur.close()
  return val
