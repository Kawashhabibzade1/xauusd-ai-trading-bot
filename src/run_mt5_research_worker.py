"""
Poll the MT5 exporter CSV and rerun the MT5 research/paper-trading pipeline whenever a new bar arrives.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mt5_client import resolve_export_file_path
from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT,
    DEFAULT_MT5_RESEARCH_NOTIFICATION_STATE_OUTPUT,
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    DEFAULT_MT5_RESEARCH_WORKER_STATE_OUTPUT,
    json_dump,
    resolve_repo_path,
)
from run_mt5_research_pipeline import run_mt5_research_pipeline
from twelvedata_client import resolve_env_value


DEFAULT_TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
DEFAULT_TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=5, help="How often to check the MT5 exporter CSV.")
    parser.add_argument("--once", action="store_true", help="Run only one cycle and exit.")
    parser.add_argument("--report-output", default=DEFAULT_MT5_RESEARCH_REPORT_OUTPUT, help="MT5 research report output path.")
    parser.add_argument("--worker-state-output", default=DEFAULT_MT5_RESEARCH_WORKER_STATE_OUTPUT, help="Worker state JSON output.")
    parser.add_argument(
        "--predictions-output",
        default=DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT,
        help="Research predictions CSV used to detect the newest trade-ready signal.",
    )
    parser.add_argument(
        "--notification-state-output",
        default=DEFAULT_MT5_RESEARCH_NOTIFICATION_STATE_OUTPUT,
        help="Notification dedupe state JSON output.",
    )
    parser.add_argument(
        "--telegram-bot-token-env",
        default=DEFAULT_TELEGRAM_BOT_TOKEN_ENV,
        help="Environment variable containing the Telegram bot token.",
    )
    parser.add_argument(
        "--telegram-chat-id-env",
        default=DEFAULT_TELEGRAM_CHAT_ID_ENV,
        help="Environment variable containing the Telegram chat id.",
    )
    return parser.parse_args()


def run_cycle(report_output: str) -> dict:
    report = run_mt5_research_pipeline(report_output=report_output)
    latest_signal = report.get("latest_signal", {})
    live_source = report.get("mt5_live_source", {})
    paper = report.get("paper_trading", {})
    learning = report.get("learning", {})
    return {
        "last_run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_output": report_output,
        "symbol": live_source.get("symbol", "XAUUSD"),
        "latest_signal_time": latest_signal.get("time"),
        "latest_signal": latest_signal.get("signal"),
        "latest_signal_gate_status": latest_signal.get("gate_status"),
        "latest_signal_session_name": latest_signal.get("session_name"),
        "latest_signal_setup_score": latest_signal.get("setup_score", 0.0),
        "latest_signal_expected_value": latest_signal.get("expected_value", 0.0),
        "latest_signal_entry_price": latest_signal.get("entry_price", 0.0),
        "latest_signal_reason_blocked": latest_signal.get("reason_blocked", ""),
        "paper_trade_count": paper.get("trade_count", 0),
        "paper_precision": paper.get("precision", 0.0),
        "paper_profit_factor": paper.get("profit_factor", 0.0),
        "learning_closed_trades": learning.get("closed_trades", 0),
        "learning_retrain_ready": learning.get("retrain_ready", False),
        "learning_blockers": learning.get("retrain_blockers", []),
    }


def read_last_bar_time(source: Path) -> str | None:
    last_line = ""
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped

    if not last_line:
        return None

    first_field = last_line.split(",", 1)[0].strip()
    return first_field or None


def load_notification_state(path_like: str) -> dict[str, Any]:
    path = resolve_repo_path(path_like)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def parse_signal_time(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def load_latest_trade_ready_signal(predictions_output: str) -> dict[str, Any] | None:
    path = resolve_repo_path(predictions_output)
    if not path.exists():
        return None

    latest_row: dict[str, str] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            latest_row = row

    if latest_row is None:
        return None

    signal = str(latest_row.get("recommended_trade", "")).strip().upper()
    gate_status = str(latest_row.get("gate_status", "")).strip().upper()
    paper_status = str(latest_row.get("paper_status", "")).strip().upper()
    signal_time_text = str(latest_row.get("time", "")).strip()
    signal_time = parse_signal_time(signal_time_text)
    if signal not in {"LONG", "SHORT"} or gate_status != "READY" or paper_status != "SIGNAL_READY" or signal_time is None:
        return None

    return {
        "signal": signal,
        "gate_status": gate_status,
        "paper_status": paper_status,
        "signal_time": signal_time_text,
        "symbol": str(latest_row.get("symbol", "")).strip(),
        "session_name": str(latest_row.get("session_name", "")).strip(),
        "entry_price": float(latest_row.get("entry_price", 0.0) or 0.0),
        "setup_score": float(latest_row.get("setup_score", 0.0) or 0.0),
        "expected_value": float(latest_row.get("expected_value", 0.0) or 0.0),
    }


def escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def notify_desktop(title: str, subtitle: str, message: str) -> tuple[str, str]:
    osascript = shutil.which("osascript")
    if osascript is None:
        return "unavailable", "osascript not found"

    script = (
        f'display notification "{escape_applescript(message)}" '
        f'with title "{escape_applescript(title)}" '
        f'subtitle "{escape_applescript(subtitle)}"'
    )
    try:
        subprocess.run([osascript, "-e", script], check=True, capture_output=True, text=True)
    except Exception as exc:
        return "failed", str(exc)
    return "sent", ""


def notify_telegram(bot_token: str, chat_id: str, message: str) -> tuple[str, str]:
    payload = urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return "failed", str(exc)

    if not payload.get("ok", False):
        return "failed", str(payload.get("description", "Telegram API request failed."))
    return "sent", ""


def maybe_send_trade_notification(
    state: dict[str, Any],
    predictions_output: str,
    notification_state_output: str,
    telegram_bot_token_env: str,
    telegram_chat_id_env: str,
) -> dict[str, Any]:
    symbol = str(state.get("symbol", "XAUUSD")).strip() or "XAUUSD"
    latest_signal_time = str(state.get("latest_signal_time", "")).strip()
    latest_signal = str(state.get("latest_signal", "")).strip().upper()
    ready_signal = load_latest_trade_ready_signal(predictions_output)

    notification_state = load_notification_state(notification_state_output)
    notification_state.setdefault("last_notified_signal_time", "")
    notification_state.setdefault("last_notified_signal", "")
    notification_state.setdefault("last_notified_at", "")
    notification_state.setdefault("last_notification_status", "idle")
    notification_state.setdefault("last_notification_message", "")

    if ready_signal is None:
        notification_state["last_evaluated_signal_time"] = latest_signal_time
        notification_state["last_evaluated_signal"] = latest_signal
        notification_state["last_notification_status"] = "idle"
        notification_state["last_notification_message"] = "No new trade-ready signal."
        json_dump(notification_state, notification_state_output)
        return notification_state

    signal = str(ready_signal.get("signal", "")).strip().upper()
    gate_status = str(ready_signal.get("gate_status", "")).strip().upper()
    signal_time = str(ready_signal.get("signal_time", "")).strip()
    signal_time_dt = parse_signal_time(signal_time)
    last_notified_signal_time = str(notification_state.get("last_notified_signal_time", "")).strip()
    last_notified_signal = str(notification_state.get("last_notified_signal", "")).strip().upper()
    last_notified_signal_time_dt = parse_signal_time(last_notified_signal_time)

    if signal not in {"LONG", "SHORT"} or gate_status != "READY" or not signal_time or signal_time_dt is None:
        notification_state["last_evaluated_signal_time"] = latest_signal_time
        notification_state["last_evaluated_signal"] = latest_signal
        notification_state["last_notification_status"] = "idle"
        notification_state["last_notification_message"] = "No new trade-ready signal."
        json_dump(notification_state, notification_state_output)
        return notification_state

    if last_notified_signal_time_dt is not None and signal_time_dt < last_notified_signal_time_dt:
        notification_state["last_evaluated_signal_time"] = latest_signal_time or signal_time
        notification_state["last_evaluated_signal"] = latest_signal or signal
        notification_state["last_notification_status"] = "idle"
        notification_state["last_notification_message"] = "Latest trade-ready signal is older than the most recent notified signal."
        json_dump(notification_state, notification_state_output)
        return notification_state

    if last_notified_signal_time_dt is not None and signal_time_dt == last_notified_signal_time_dt and last_notified_signal == signal:
        notification_state["last_evaluated_signal_time"] = latest_signal_time or signal_time
        notification_state["last_evaluated_signal"] = latest_signal or signal
        notification_state["last_notification_status"] = "duplicate_skipped"
        notification_state["last_notification_message"] = "Trade-ready signal was already notified."
        json_dump(notification_state, notification_state_output)
        return notification_state

    already_notified = (
        notification_state.get("last_notified_signal_time") == signal_time
        and notification_state.get("last_notified_signal") == signal
    )
    if already_notified:
        notification_state["last_evaluated_signal_time"] = signal_time
        notification_state["last_evaluated_signal"] = signal
        notification_state["last_notification_status"] = "duplicate_skipped"
        notification_state["last_notification_message"] = "Trade-ready signal was already notified."
        json_dump(notification_state, notification_state_output)
        return notification_state

    setup_score = float(ready_signal.get("setup_score", 0.0) or 0.0)
    expected_value = float(ready_signal.get("expected_value", 0.0) or 0.0)
    entry_price = float(ready_signal.get("entry_price", 0.0) or 0.0)
    session_name = str(ready_signal.get("session_name", "")).strip() or "Active session"

    title = "XAUUSD AI Bot"
    subtitle = f"{symbol} {signal} setup ready"
    message = (
        f"Time {signal_time} | Entry {entry_price:.2f} | "
        f"Score {setup_score:.1%} | EV {expected_value:.4f} | {session_name}"
    )
    telegram_message = (
        f"{symbol} {signal} setup ready\n"
        f"Time: {signal_time}\n"
        f"Entry: {entry_price:.2f}\n"
        f"Setup score: {setup_score:.1%}\n"
        f"Expected value: {expected_value:.4f}\n"
        f"Session: {session_name}"
    )
    print(f"ALERT: {subtitle} :: {message}")
    desktop_status, desktop_message = notify_desktop(title, subtitle, message)

    bot_token = resolve_env_value(telegram_bot_token_env)
    chat_id = resolve_env_value(telegram_chat_id_env)
    if bot_token and chat_id:
        telegram_status, telegram_message_status = notify_telegram(bot_token, chat_id, telegram_message)
    else:
        telegram_status = "unconfigured"
        telegram_message_status = (
            f"Set {telegram_bot_token_env} and {telegram_chat_id_env} to enable Telegram alerts."
        )

    telegram_configured = bool(bot_token and chat_id)
    delivered = telegram_status == "sent" if telegram_configured else desktop_status == "sent"
    status_message = (
        f"desktop={desktop_status}; telegram={telegram_status}"
    )
    detail_message = " | ".join(
        part
        for part in [
            desktop_message and f"desktop: {desktop_message}",
            telegram_message_status and f"telegram: {telegram_message_status}",
        ]
        if part
    ) or message

    notification_state["last_evaluated_signal_time"] = latest_signal_time or signal_time
    notification_state["last_evaluated_signal"] = latest_signal or signal
    notification_state["last_notification_status"] = status_message
    notification_state["last_notification_message"] = detail_message
    notification_state["last_notification_channels"] = {
        "desktop": {"status": desktop_status, "message": desktop_message},
        "telegram": {"status": telegram_status, "message": telegram_message_status},
    }
    if delivered:
        notification_state["last_notified_signal_time"] = signal_time
        notification_state["last_notified_signal"] = signal
        notification_state["last_notified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(notification_state, notification_state_output)
    return notification_state


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("MT5 RESEARCH WORKER")
    print("=" * 70)
    print(f"Report output : {args.report_output}")
    print(f"Poll seconds  : {args.poll_seconds}")
    print()

    last_mtime = None
    last_bar_time = None
    while True:
        source = Path(resolve_export_file_path())
        if source.exists():
            current_mtime = source.stat().st_mtime
            current_bar_time = read_last_bar_time(source)
            should_run = (
                last_bar_time is None or current_bar_time != last_bar_time
                if current_bar_time is not None
                else last_mtime is None or current_mtime > last_mtime
            )
            if should_run:
                try:
                    state = run_cycle(args.report_output)
                except Exception as exc:
                    error_state = {
                        "last_run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "report_output": args.report_output,
                        "source_file": str(source),
                        "source_mtime": current_mtime,
                        "source_last_bar_time": current_bar_time,
                        "worker_error": str(exc),
                    }
                    json_dump(error_state, args.worker_state_output)
                    print(f"[{error_state['last_run_time']}] Worker cycle failed: {exc}")
                    if args.once:
                        raise
                    time.sleep(max(5, args.poll_seconds))
                    continue
                state["source_file"] = str(source)
                state["source_mtime"] = current_mtime
                state["source_last_bar_time"] = current_bar_time
                notification_state = maybe_send_trade_notification(
                    state,
                    args.predictions_output,
                    args.notification_state_output,
                    args.telegram_bot_token_env,
                    args.telegram_chat_id_env,
                )
                state["notification_status"] = notification_state.get("last_notification_status", "idle")
                state["notification_message"] = notification_state.get("last_notification_message", "")
                state["last_notified_signal_time"] = notification_state.get("last_notified_signal_time", "")
                state["last_notified_signal"] = notification_state.get("last_notified_signal", "")
                json_dump(state, args.worker_state_output)
                print(f"[{state['last_run_time']}] Updated report from MT5 exporter. latest={state['latest_signal']} trades={state['paper_trade_count']}")
                last_mtime = current_mtime
                last_bar_time = current_bar_time
                if args.once:
                    break
        elif args.once:
            raise SystemExit(f"MT5 exporter CSV not found: {source}")

        if args.once:
            break
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
