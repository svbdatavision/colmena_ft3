#!/usr/bin/env python3
"""Run the FastTrack 3.0 daily pipeline using a single Snowflake session."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

from FT3_dia import apply_model_to_all_licenses
from src.data_loader import SnowflakeDataLoader

BASE_DIR = Path(__file__).resolve().parent


def _load_environment() -> None:
    """Load environment variables from .env if present."""
    dotenv_candidate = os.environ.get("DOTENV_PATH", "/app/.env")
    dotenv_path = Path(dotenv_candidate)
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        load_dotenv()


def _build_conn() -> snowflake.connector.SnowflakeConnection:
    """Create a Snowflake connection with keep-alive enabled."""
    required = [
        "SF_USER",
        "SF_PASSWORD",
        "SF_ACCOUNT",
        "SF_WAREHOUSE",
        "SF_DATABASE",
        "SF_SCHEMA",
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

    return snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA"),
        role=os.getenv("SF_ROLE"),
        client_session_keep_alive=True,
        application="FT3_DAILY_PIPELINE",
    )


def _execute_sql(conn: snowflake.connector.SnowflakeConnection,
                 sql_path: Path,
                 label: str,
                 summary: str) -> None:
    """Execute a SQL file inside the shared Snowflake session."""
    print(f"\nPASO {label}: Ejecutando {sql_path.name}...")
    print("----------------------------------------")

    if not sql_path.is_file():
        raise FileNotFoundError(f"No se encuentra {sql_path}")

    with sql_path.open("r", encoding="utf-8") as handle:
        query_content = handle.read()

    cursor = conn.cursor()
    start_time = datetime.now()
    try:
        cursor.execute("USE DATABASE OPX")
        cursor.execute("USE SCHEMA P_DDV_OPX_MDPREDICTIVO")
        cursor.execute(query_content, num_statements=0)
        conn.commit()
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ {sql_path.name} ejecutada en {elapsed:.1f} segundos")
        print(f"   {summary}")
    except Exception as exc:
        conn.rollback()
        print(f"❌ Error en {sql_path.name}: {exc}")
        raise
    finally:
        cursor.close()


def main() -> int:
    print("=" * 42)
    print("INICIANDO PIPELINE DIARIO FT3")
    print(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 42)

    _load_environment()

    try:
        conn = _build_conn()
    except Exception as exc:
        print(f"❌ No fue posible conectar a Snowflake: {exc}")
        return 1

    loader = SnowflakeDataLoader()
    loader.conn = conn

    step_summaries = []

    try:
        day_of_week = datetime.now().isoweekday()
        if day_of_week == 1:
            _execute_sql(
                conn,
                BASE_DIR / "query_lunes.sql",
                "1a",
                "Licencias del fin de semana cargadas",
            )
            step_summaries.append("1a. query_lunes.sql: Licencias del fin de semana cargadas")

            _execute_sql(
                conn,
                BASE_DIR / "query_diaria.sql",
                "1b",
                "Licencias del día anterior cargadas",
            )
            step_summaries.append("1b. query_diaria.sql: Licencias del día anterior cargadas")
        else:
            _execute_sql(
                conn,
                BASE_DIR / "query_diaria.sql",
                "1",
                "Licencias del día anterior cargadas",
            )
            step_summaries.append("1. query_diaria.sql: Licencias del día anterior cargadas")

        _execute_sql(
            conn,
            BASE_DIR / "query_2.sql",
            "2",
            "Tabla MODELO_LM_202507_TRAIN actualizada",
        )
        step_summaries.append("2. query_2.sql: Tabla MODELO_LM_202507_TRAIN actualizada")

        print("\nPASO 3: Ejecutando FT3_dia.py...")
        print("----------------------------------------")
        report_file = apply_model_to_all_licenses(loader=loader)
        step_summaries.append("3. FT3_dia.py: Predicciones generadas")

        print("\n" + "=" * 42)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("Resumen del pipeline:")
        for item in step_summaries:
            print(f"  {item}")
        if report_file:
            print(f"  Reporte generado: {report_file}")
        print(f"Hora finalización: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 42)

        return 0

    except Exception as exc:
        print("\n❌ Pipeline abortado")
        print(f"Motivo: {exc}")
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _entrypoint() -> None:
    status = main()
    sys.exit(status)


if __name__ == "__main__":
    _entrypoint()
