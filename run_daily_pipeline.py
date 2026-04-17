#!/usr/bin/env python3
"""Run the FastTrack 3.0 daily pipeline using a single Databricks session."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from dotenv import load_dotenv

from FT3_dia import apply_model_to_all_licenses
from src.data_loader import SnowflakeDataLoader
from src.spark_sql_compat import get_spark_session

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    # When running in Databricks notebook, __file__ is not defined.
    BASE_DIR = Path(os.getcwd())
_SPARK = None


class _TeeStream:
    """Replica stdout a consola y a un buffer en memoria."""

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
    """Send notification to Teams webhook with payload fallbacks.

    Supports both:
    - Legacy Incoming Webhook connectors.
    - Teams Workflows/Power Automate webhook endpoints.
    """
    if not webhook_url:
        return

    def _post_payload(payload_obj: Any) -> tuple[bool, str]:
        payload = json.dumps(payload_obj).encode("utf-8")
        req = Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=20):
                return True, ""
        except HTTPError as exc:
            # Intentar leer body del error para diagnóstico.
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            return False, f"HTTP {exc.code}: {detail or str(exc)}"
        except (URLError, TimeoutError) as exc:
            return False, str(exc)

    # Payloads en orden de preferencia/fallback.
    payload_candidates = [
        # Incoming webhook clásico
        {"text": message},
        # Connector card
        {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "FT3 pipeline",
            "text": message.replace("\n", "<br>"),
        },
        # Workflows / Power Automate (Adaptive Card envelope)
        {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "FT3 Pipeline",
                                "weight": "Bolder",
                                "wrap": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": message,
                                "wrap": True,
                            },
                        ],
                    },
                }
            ],
        },
    ]

    last_error = ""
    for payload_obj in payload_candidates:
        ok, error = _post_payload(payload_obj)
        if ok:
            return
        last_error = error

    print(f"⚠ No se pudo enviar alerta a Teams: {last_error}")


def _split_text_for_teams(text: str, max_chars: int = 3000) -> list[str]:
    """Divide texto largo en bloques aptos para webhook de Teams."""
    if not text:
        return [""]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(line):
                chunks.append(line[start:start + max_chars])
                start += max_chars
            continue

        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks


def _send_teams_log(webhook_url: str, title: str, full_log: str) -> None:
    """Envía el log completo en uno o varios mensajes a Teams."""
    chunks = _split_text_for_teams(full_log.strip(), max_chars=3000)
    total = len(chunks)
    if total == 0:
        _send_teams_alert(webhook_url, title)
        return

    for idx, chunk in enumerate(chunks, start=1):
        header = f"{title}\n\nLog pipeline ({idx}/{total}):"
        message = f"{header}\n{chunk}"
        _send_teams_alert(webhook_url, message)


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
                 summary: str,
                 query_content_override: str | None = None,
                 sql_display_name: str | None = None) -> None:
    """Execute a SQL file using shared Spark session."""
    display_name = sql_display_name or sql_path.name
    print(f"\nPASO {label}: Ejecutando {display_name}...")
    print("----------------------------------------")

    if query_content_override is None:
        if not sql_path.is_file():
            raise FileNotFoundError(f"No se encuentra {sql_path}")
        with sql_path.open("r", encoding="utf-8") as handle:
            query_content = handle.read()
    else:
        query_content = query_content_override

    statements = _split_sql_statements(query_content)
    if not statements:
        print(f"⚠️  {display_name} no contiene sentencias ejecutables")
        return

    spark = _get_spark()
    start_time = datetime.now()
    try:
        for statement in statements:
            spark.sql(statement)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ {display_name} ejecutada en {elapsed:.1f} segundos ({len(statements)} sentencias)")
        print(f"   {summary}")
    except Exception as exc:
        print(f"❌ Error en {display_name}: {exc}")
        raise


def _render_query_diaria_for_date(query_content: str, processing_date: date) -> str:
    """Render query_diaria.sql replacing 'ayer' expression with a fixed date."""
    target_expr = "date_add(current_date(), -1)"
    replacement = f"DATE('{processing_date.isoformat()}')"
    return query_content.replace(target_expr, replacement)


def _find_missing_base_dates(spark, lookback_days: int = 30) -> list[date]:
    """Return dates with ALFIL data but no rows inserted in BASE."""
    diagnostics_sql = f"""
    WITH fechas AS (
      SELECT explode(sequence(date_add(current_date(), -{lookback_days}), date_add(current_date(), -1), interval 1 day)) AS fecha
    ),
    alfil AS (
      SELECT DATE(TO_DATE(FECHA_RECEPCION)) AS fecha, COUNT(*) AS n_alfil
      FROM OPX.P_DDV_OPX_MDPREDICTIVO.SBN_LM_INPUT_DIARIO_ALFIL
      WHERE DATE(TO_DATE(FECHA_RECEPCION)) >= date_add(current_date(), -{lookback_days})
      GROUP BY DATE(TO_DATE(FECHA_RECEPCION))
    ),
    base AS (
      SELECT DATE(FECHA_RECEPCION) AS fecha, COUNT(*) AS n_base
      FROM OPX.P_DDV_OPX_MDPREDICTIVO.MODELO_LM_202507_BASE
      WHERE DATE(FECHA_RECEPCION) >= date_add(current_date(), -{lookback_days})
      GROUP BY DATE(FECHA_RECEPCION)
    )
    SELECT f.fecha
    FROM fechas f
    LEFT JOIN alfil a ON f.fecha = a.fecha
    LEFT JOIN base b ON f.fecha = b.fecha
    WHERE COALESCE(a.n_alfil, 0) > 0
      AND COALESCE(b.n_base, 0) = 0
    ORDER BY f.fecha
    """

    rows = spark.sql(diagnostics_sql).collect()
    parsed_dates: list[date] = []
    for row in rows:
        raw = row["fecha"]
        if isinstance(raw, datetime):
            parsed_dates.append(raw.date())
        elif isinstance(raw, date):
            parsed_dates.append(raw)
        elif raw is None:
            continue
        else:
            parsed_dates.append(datetime.strptime(str(raw)[:10], "%Y-%m-%d").date())
    return parsed_dates


def _execute_query_diaria_backfill(spark, sql_path: Path, lookback_days: int = 30) -> list[date]:
    """Execute query_diaria with fixed dates to backfill missing BASE days."""
    missing_dates = _find_missing_base_dates(spark, lookback_days=lookback_days)
    if not missing_dates:
        print("\nPASO 0: Sin brechas detectadas entre ALFIL y BASE en ventana reciente.")
        return []

    with sql_path.open("r", encoding="utf-8") as handle:
        raw_query = handle.read()

    print(f"\nPASO 0: Detectadas {len(missing_dates)} fecha(s) faltantes para backfill en BASE.")
    for idx, day in enumerate(missing_dates, start=1):
        rendered_query = _render_query_diaria_for_date(raw_query, day)
        _execute_sql(
            sql_path=sql_path,
            label=f"0.{idx}",
            summary=f"Backfill ejecutado para FECHA_RECEPCION = {day.isoformat()}",
            query_content_override=rendered_query,
            sql_display_name=f"{sql_path.name} [backfill {day.isoformat()}]",
        )
    return missing_dates


def main() -> int:
    # 1) Cargar variables de entorno (.env opcional) y webhook Teams.
    _load_environment()
    teams_webhook_url = _get_teams_webhook_url()
    started_at = datetime.now()
    step_summaries: list[str] = []
    report_file: str | None = None
    failure_reason = ""

    # 2) Notificar inicio a Teams (si hay webhook configurado).
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
    loader = None
    with redirect_stdout(_TeeStream(sys.stdout, captured_stdout)):
        # 3) Asegurar rutas relativas desde el directorio del proyecto.
        os.chdir(BASE_DIR)

        print("=" * 42)
        print("INICIANDO PIPELINE DIARIO FT3")
        print(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 42)

        # 4) Inicializar Spark y loader.
        try:
            spark = _get_spark()
        except Exception as exc:
            failure_reason = f"No fue posible iniciar Spark en Databricks: {exc}"
            print(f"❌ {failure_reason}")
            status = 1
        else:
            loader = SnowflakeDataLoader()
            try:
                loader.connect()
                # Mantener binding explícito a la misma sesión Spark.
                loader.spark = spark
                if loader.conn is not None:
                    loader.conn.spark = spark
            except Exception as exc:
                failure_reason = f"No fue posible inicializar loader Spark: {exc}"
                print(f"❌ {failure_reason}")
                status = 1
            else:
                # 5) Ejecutar pasos SQL + scoring FT3.
                try:
                    backfill_days = _execute_query_diaria_backfill(
                        spark=spark,
                        sql_path=BASE_DIR / "query_diaria.sql",
                        lookback_days=30,
                    )
                    if backfill_days:
                        step_summaries.append(
                            "0. query_diaria.sql backfill: "
                            f"{len(backfill_days)} fecha(s) reprocesadas "
                            f"({backfill_days[0]} a {backfill_days[-1]})"
                        )

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
                    failure_reason = str(exc)
                    print("\n❌ Pipeline abortado")
                    print(f"Motivo: {exc}")
                    status = 1
                finally:
                    # 6) Cierre de recursos.
                    try:
                        loader.disconnect()
                    except Exception:
                        pass

    # 7) Notificar resultado final a Teams con log completo (dinámico).
    if teams_webhook_url:
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()
        if status == 0:
            title = (
                "✅ PIPELINE DIARIO FT3 COMPLETADO EXITOSAMENTE\n"
                f"Inicio: {started_at:%Y-%m-%d %H:%M:%S}\n"
                f"Fin: {finished_at:%Y-%m-%d %H:%M:%S}\n"
                f"Duración: {duration:.1f}s"
            )
        else:
            title = (
                "❌ PIPELINE DIARIO FT3 FALLÓ\n"
                f"Inicio: {started_at:%Y-%m-%d %H:%M:%S}\n"
                f"Fallo: {finished_at:%Y-%m-%d %H:%M:%S}\n"
                f"Duración hasta fallo: {duration:.1f}s\n"
                f"Motivo: {failure_reason or 'Error no especificado'}"
            )
        _send_teams_log(teams_webhook_url, title=title, full_log=captured_stdout.getvalue())

    return status


if __name__ == "__main__":
    main()
