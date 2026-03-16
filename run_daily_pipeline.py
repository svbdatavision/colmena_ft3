#!/usr/bin/env python3
"""Run the FastTrack 3.0 daily pipeline using a single Databricks session."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
import json

from dotenv import load_dotenv

from FT3_dia import apply_model_to_all_licenses
from src.data_loader import SnowflakeDataLoader
from src.spark_sql_compat import get_spark_session

BASE_DIR = Path(__file__).resolve().parent
_SPARK = None


def _running_in_notebook() -> bool:
    """Return True when executed inside a notebook kernel."""
    try:
        from IPython import get_ipython
    except Exception:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


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


def _get_spark():
    """Get active Spark session from Databricks runtime."""
    global _SPARK
    if _SPARK is None:
        _SPARK = get_spark_session()
    return _SPARK


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


def _execute_sql(sql_path: Path,
                 label: str,
                 summary: str) -> None:
    """Execute a SQL file using shared Spark session."""
    print(f"\nPASO {label}: Ejecutando {sql_path.name}...")
    print("----------------------------------------")

    if not sql_path.is_file():
        raise FileNotFoundError(f"No se encuentra {sql_path}")

    with sql_path.open("r", encoding="utf-8") as handle:
        query_content = handle.read()

    statements = _split_sql_statements(query_content)
    if not statements:
        print(f"⚠️  {sql_path.name} no contiene sentencias ejecutables")
        return

    spark = _get_spark()
    start_time = datetime.now()
    try:
        for statement in statements:
            spark.sql(statement)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ {sql_path.name} ejecutada en {elapsed:.1f} segundos ({len(statements)} sentencias)")
        print(f"   {summary}")
    except Exception as exc:
        print(f"❌ Error en {sql_path.name}: {exc}")
        raise


def main() -> int:
    _load_environment()
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
        # Ensure relative paths (models/, results/, etc.) resolve from project root.
        os.chdir(BASE_DIR)

        print("=" * 42)
        print("INICIANDO PIPELINE DIARIO FT3")
        print(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 42)

        try:
            spark = _get_spark()
        except Exception as exc:
            print(f"❌ No fue posible iniciar Spark en Databricks: {exc}")
            status = 1
        else:
            loader = SnowflakeDataLoader()
            try:
                loader.connect()
                # Keep explicit binding for shared-session consistency.
                loader.spark = spark
                if loader.conn is not None:
                    loader.conn.spark = spark
            except Exception as exc:
                print(f"❌ No fue posible inicializar loader Spark: {exc}")
                status = 1
            else:
                step_summaries = []
                try:
                    day_of_week = datetime.now().isoweekday()
                    if day_of_week == 1:
                        _execute_sql(
                            BASE_DIR / "query_lunes.sql",
                            "1a",
                            "Licencias del fin de semana cargadas",
                        )
                        step_summaries.append("1a. query_lunes.sql: Licencias del fin de semana cargadas")

                        _execute_sql(
                            BASE_DIR / "query_diaria.sql",
                            "1b",
                            "Licencias del día anterior cargadas",
                        )
                        step_summaries.append("1b. query_diaria.sql: Licencias del día anterior cargadas")
                    else:
                        _execute_sql(
                            BASE_DIR / "query_diaria.sql",
                            "1",
                            "Licencias del día anterior cargadas",
                        )
                        step_summaries.append("1. query_diaria.sql: Licencias del día anterior cargadas")

                    _execute_sql(
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
                        loader.disconnect()
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


if __name__ == "__main__":
    if _running_in_notebook():
        main()
    else:
        sys.exit(main())
