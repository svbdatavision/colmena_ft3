#!/usr/bin/env python3
"""Run the FastTrack 3.0 daily pipeline using a single Databricks session."""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
import json

try:
    import databricks.sql as databricks_sql
except ModuleNotFoundError:
    databricks_sql = None
from dotenv import load_dotenv

from FT3_dia import apply_model_to_all_licenses
from src.data_loader import SnowflakeDataLoader

BASE_DIR = Path(__file__).resolve().parent


class _TeeStream:
    """Write stdout both to console and in-memory buffer."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _get_teams_webhook_url() -> str:
    """Return Teams webhook URL from environment, if configured."""
    return (
        os.getenv("TEAMS_WEBHOOK_URL", "").strip()
        or os.getenv("MSTEAMS_WEBHOOK_URL", "").strip()
    )


def _send_teams_alert(webhook_url: str, message: str) -> None:
    """Send a plain text notification to Microsoft Teams incoming webhook."""
    if not webhook_url:
        return

    payload = json.dumps({"text": message}).encode("utf-8")
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15):
            pass
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"⚠ No se pudo enviar alerta a Teams: {exc}")


def _truncate_for_teams(text: str, max_chars: int = 26000) -> str:
    """Trim oversized messages to avoid webhook payload issues."""
    if len(text) <= max_chars:
        return text
    suffix = "\n\n...[mensaje truncado por longitud]..."
    return text[: max_chars - len(suffix)] + suffix


def _build_teams_final_message(status: int, started_at: datetime, finished_at: datetime, output: str) -> str:
    """Build final Teams message with status and pipeline output."""
    status_label = "COMPLETADO EXITOSAMENTE" if status == 0 else "FALLÓ"
    status_icon = "✅" if status == 0 else "❌"
    duration = (finished_at - started_at).total_seconds()
    header = (
        f"{status_icon} PIPELINE DIARIO FT3 {status_label}\n"
        f"Inicio: {started_at:%Y-%m-%d %H:%M:%S}\n"
        f"Fin: {finished_at:%Y-%m-%d %H:%M:%S}\n"
        f"Duración: {duration:.1f}s\n\n"
    )
    return _truncate_for_teams(header + output.strip())


def _load_environment() -> None:
    """Load environment variables from .env if present."""
    dotenv_candidate = os.environ.get("DOTENV_PATH", "/app/.env")
    dotenv_path = Path(dotenv_candidate)
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        load_dotenv()


def _build_conn():
    """Create a Databricks SQL connection."""
    if databricks_sql is None:
        raise ModuleNotFoundError(
            "No module named 'databricks.sql'. "
            "Instale dependencias con: python3 -m pip install -r requirements.txt "
            "o ejecute: bash scripts/setup_cloud_env.sh"
        )

    required = [
        "DATABRICKS_SERVER_HOSTNAME",
        "DATABRICKS_HTTP_PATH",
    ]
    missing = [var for var in required if not os.getenv(var)]
    if not (os.getenv("DATABRICKS_ACCESS_TOKEN") or os.getenv("DATABRICKS_TOKEN")):
        missing.append("DATABRICKS_ACCESS_TOKEN/DATABRICKS_TOKEN")
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

    conn_kwargs = {
        "server_hostname": os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        "http_path": os.getenv("DATABRICKS_HTTP_PATH"),
        "access_token": os.getenv("DATABRICKS_ACCESS_TOKEN") or os.getenv("DATABRICKS_TOKEN"),
        "_user_agent_entry": "FT3_DAILY_PIPELINE",
    }
    if os.getenv("DATABRICKS_CATALOG"):
        conn_kwargs["catalog"] = os.getenv("DATABRICKS_CATALOG")
    if os.getenv("DATABRICKS_SCHEMA"):
        conn_kwargs["schema"] = os.getenv("DATABRICKS_SCHEMA")
    return databricks_sql.connect(**conn_kwargs)


def _split_sql_statements(query_content: str) -> list[str]:
    """Split SQL script into executable statements."""
    statements = []
    buffer = []
    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    n = len(query_content)

    while i < n:
        ch = query_content[i]
        nxt = query_content[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                buffer.append(ch)
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_single_quote and not in_double_quote and not in_backtick:
            if ch == "-" and nxt == "-":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if ch == "'" and not in_double_quote and not in_backtick:
            if in_single_quote and nxt == "'":
                buffer.append(ch)
                buffer.append(nxt)
                i += 2
                continue
            in_single_quote = not in_single_quote
            buffer.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single_quote and not in_backtick:
            in_double_quote = not in_double_quote
            buffer.append(ch)
            i += 1
            continue

        if ch == "`" and not in_single_quote and not in_double_quote:
            in_backtick = not in_backtick
            buffer.append(ch)
            i += 1
            continue

        if ch == ";" and not in_single_quote and not in_double_quote and not in_backtick:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            i += 1
            continue

        buffer.append(ch)
        i += 1

    trailing = "".join(buffer).strip()
    if trailing:
        statements.append(trailing)

    return statements


def _read_text_with_retry(path: Path, retries: int = 3, base_wait_seconds: float = 2.0) -> str:
    """Read text file with retries to mitigate intermittent workspace I/O errors."""
    for attempt in range(1, retries + 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            is_last_attempt = attempt == retries
            if is_last_attempt:
                raise
            wait_seconds = base_wait_seconds * attempt
            print(
                f"⚠ Error de I/O leyendo {path.name} "
                f"(intento {attempt}/{retries}): {exc}. Reintentando en {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"No fue posible leer el archivo SQL: {path}")


def _execute_sql(conn,
                 sql_path: Path,
                 label: str,
                 summary: str) -> None:
    """Execute a SQL file inside the shared Databricks session."""
    print(f"\nPASO {label}: Ejecutando {sql_path.name}...")
    print("----------------------------------------")

    if not sql_path.is_file():
        raise FileNotFoundError(f"No se encuentra {sql_path}")

    query_content = _read_text_with_retry(sql_path)

    statements = _split_sql_statements(query_content)
    if not statements:
        print(f"⚠️  {sql_path.name} no contiene sentencias ejecutables")
        return

    cursor = conn.cursor()
    start_time = datetime.now()
    try:
        for statement in statements:
            cursor.execute(statement)
        try:
            conn.commit()
        except Exception:
            # Some DB-API connectors autocommit.
            pass
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ {sql_path.name} ejecutada en {elapsed:.1f} segundos ({len(statements)} sentencias)")
        print(f"   {summary}")
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"❌ Error en {sql_path.name}: {exc}")
        raise
    finally:
        cursor.close()


def main() -> int:
    _load_environment()
    # Reduce noisy Py4J INFO logs that flood Databricks job output.
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)
    teams_webhook_url = _get_teams_webhook_url()
    started_at = datetime.now()

    if teams_webhook_url:
        _send_teams_alert(
            teams_webhook_url,
            (
                "🚀 INICIANDO PIPELINE DIARIO FT3\n"
                f"Fecha: {started_at:%Y-%m-%d %H:%M:%S}"
            ),
        )

    captured_stdout = io.StringIO()
    status = 1
    with redirect_stdout(_TeeStream(sys.stdout, captured_stdout)):
        print("=" * 42)
        print("INICIANDO PIPELINE DIARIO FT3")
        print(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 42)

        try:
            conn = _build_conn()
        except Exception as exc:
            print(f"❌ No fue posible conectar a Databricks SQL: {exc}")
            status = 1
        else:
            loader = SnowflakeDataLoader()
            loader.conn = conn
            try:
                loader.cursor = conn.cursor()
            except Exception:
                loader.cursor = None

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
                status = 0

            except Exception as exc:
                print("\n❌ Pipeline abortado")
                print(f"Motivo: {exc}")
                status = 1
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    if teams_webhook_url:
        finished_at = datetime.now()
        message = _build_teams_final_message(
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            output=captured_stdout.getvalue(),
        )
        _send_teams_alert(teams_webhook_url, message)

    return status


def _entrypoint() -> None:
    status = main()
    sys.exit(status)


if __name__ == "__main__":
    _entrypoint()
