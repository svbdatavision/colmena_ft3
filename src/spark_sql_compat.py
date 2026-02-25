"""
Compatibility layer to run legacy DB-API style code on Spark SQL.

This module is intentionally minimal and keeps the existing calling pattern
(`conn.cursor().execute(...)`) while executing everything with `spark.sql(...)`.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Iterable, List, Optional, Sequence, Tuple


_INSERT_QMARK_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+(?P<table>.+?)\s*\((?P<columns>.+?)\)\s*VALUES\s*\((?P<values>.+?)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def get_spark_session():
    """Return active SparkSession (or create one if needed)."""
    try:
        from pyspark.sql import SparkSession
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No module named 'pyspark'. "
            "Este pipeline debe ejecutarse dentro de Databricks Runtime (Spark)."
        ) from exc

    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    return spark


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _to_sql_literal(value: Any) -> str:
    value = _normalize_scalar(value)
    if value is None or _is_nan(value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (datetime, date)):
        literal = value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else value.isoformat()
        return f"'{literal}'"

    text = str(value)
    text = text.replace("\\", "\\\\").replace("'", "''")
    return f"'{text}'"


def _apply_qmark_parameters(query: str, parameters: Sequence[Any]) -> str:
    if parameters is None:
        return query

    parts = query.split("?")
    expected = len(parts) - 1
    if expected != len(parameters):
        raise ValueError(
            f"Cantidad de parámetros inválida: esperados={expected}, recibidos={len(parameters)}"
        )

    rebuilt = parts[0]
    for idx, value in enumerate(parameters):
        rebuilt += _to_sql_literal(value) + parts[idx + 1]
    return rebuilt


class SparkSQLCursor:
    """Very small cursor wrapper over Spark SQL."""

    def __init__(self, spark) -> None:
        self.spark = spark
        self._rows: List[Tuple[Any, ...]] = []
        self._fetch_idx = 0
        self.description: Optional[List[Tuple[Any, ...]]] = None
        self.rowcount: int = -1

    def execute(self, query: str, parameters: Optional[Sequence[Any]] = None):
        statement = _apply_qmark_parameters(query, parameters) if parameters is not None else query
        df = self.spark.sql(statement)
        columns = list(df.columns)
        collected = df.collect()
        self._rows = [tuple(row[col] for col in columns) for row in collected]
        self._fetch_idx = 0
        self.description = [(col, None, None, None, None, None, None) for col in columns]
        self.rowcount = len(self._rows)
        return self

    def executemany(self, query: str, seq_of_parameters: Iterable[Sequence[Any]]):
        params_list = list(seq_of_parameters)
        if not params_list:
            self.rowcount = 0
            return self

        match = _INSERT_QMARK_RE.match(query)
        if match:
            table = match.group("table").strip()
            columns = match.group("columns").strip()
            values_sql = []
            for params in params_list:
                values_sql.append("(" + ", ".join(_to_sql_literal(p) for p in params) + ")")
            insert_sql = f"INSERT INTO {table} ({columns}) VALUES " + ", ".join(values_sql)
            self.execute(insert_sql)
            self.rowcount = len(params_list)
            return self

        affected = 0
        for params in params_list:
            self.execute(query, params)
            affected += 1
        self.rowcount = affected
        return self

    def fetchone(self):
        if self._fetch_idx >= len(self._rows):
            return None
        row = self._rows[self._fetch_idx]
        self._fetch_idx += 1
        return row

    def fetchmany(self, size: int = 1):
        if size <= 0:
            return []
        start = self._fetch_idx
        end = min(start + size, len(self._rows))
        batch = self._rows[start:end]
        self._fetch_idx = end
        return batch

    def fetchall(self):
        if self._fetch_idx >= len(self._rows):
            return []
        rows = self._rows[self._fetch_idx :]
        self._fetch_idx = len(self._rows)
        return rows

    def close(self):
        self._rows = []
        self._fetch_idx = 0
        self.description = None


class SparkSQLConnection:
    """Connection-like object to preserve no-downtime compatibility."""

    def __init__(self, spark) -> None:
        self.spark = spark

    def cursor(self):
        return SparkSQLCursor(self.spark)

    def commit(self):
        # Spark SQL autocommits statements.
        return None

    def rollback(self):
        # Spark SQL autocommits statements.
        return None

    def close(self):
        # Session lifecycle is managed by Databricks runtime.
        return None
