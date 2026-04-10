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
from env_utils import resolve_env_value
from run_mt5_research_pipeline import run_mt5_research_pipeline


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
    broker_execution = report.get("broker_execution", {}) or {}
    latest_event = broker_execution.get("latest_event", {}) or {}
    current_position = broker_execution.get("current_position", {}) or {}
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
        "broker_trading_enabled": broker_execution.get("broker_trading_enabled", False),
        "broker_block_reason": broker_execution.get("broker_block_reason", ""),
        "broker_trade_count": broker_execution.get("broker_trade_count", 0),
        "broker_closed_trade_count": broker_execution.get("broker_closed_trade_count", 0),
        "broker_consecutive_losses": broker_execution.get("consecutive_losses", 0),
        "broker_live_ready": broker_execution.get("live_ready", False),
        "broker_live_ready_blockers": broker_execution.get("live_ready_blockers", []),
        "broker_latest_event_key": latest_event.get("event_key", ""),
        "broker_latest_event_action": latest_event.get("action", ""),
        "broker_latest_event_time": latest_event.get("time_utc", ""),
        "broker_latest_event_direction": latest_event.get("direction", ""),
        "broker_latest_event_note": latest_event.get("note", ""),
        "broker_latest_event_exit_reason": latest_event.get("exit_reason", ""),
        "broker_latest_event_realized_pnl_cash": latest_event.get("realized_pnl_cash", 0.0),
        "broker_position_state": current_position.get("state", ""),
        "broker_position_direction": current_position.get("direction", ""),
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

    latest_ready_signal: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            signal = str(row.get("recommended_trade", "")).strip().upper()
            gate_status = str(row.get("gate_status", "")).strip().upper()
            paper_status = str(row.get("paper_status", "")).strip().upper()
            signal_time_text = str(row.get("time", "")).strip()
            signal_time = parse_signal_time(signal_time_text)
            if signal not in {"LONG", "SHORT"} or gate_status != "READY" or paper_status != "SIGNAL_READY" or signal_time is None:
                continue
            latest_ready_signal = {
                "signal": signal,
                "gate_status": gate_status,
                "paper_status": paper_status,
                "signal_time": signal_time_text,
                "symbol": str(row.get("symbol", "")).strip(),
                "session_name": str(row.get("session_name", "")).strip(),
                "entry_price": float(row.get("entry_price", 0.0) or 0.0),
                "setup_score": float(row.get("setup_score", 0.0) or 0.0),
                "expected_value": float(row.get("expected_value", 0.0) or 0.0),
            }

    return latest_ready_signal


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
    notification_state = load_notification_state(notification_state_output)
    notification_state.setdefault("last_seen_event_key", "")
    notification_state.setdefault("last_seen_event_action", "")
    notification_state.setdefault("last_autopilot_enabled", None)
    notification_state.setdefault("last_notified_at", "")
    notification_state.setdefault("last_notification_status", "idle")
    notification_state.setdefault("last_notification_message", "")
    notification_state.setdefault("last_notification_type", "")

    current_autopilot_enabled = bool(state.get("broker_trading_enabled", False))
    last_autopilot_enabled = notification_state.get("last_autopilot_enabled", None)
    latest_event_key = str(state.get("broker_latest_event_key", "")).strip()
    latest_event_action = str(state.get("broker_latest_event_action", "")).strip().upper()
    latest_event_time = str(state.get("broker_latest_event_time", "")).strip()
    latest_event_direction = str(state.get("broker_latest_event_direction", "")).strip().upper()
    latest_event_note = str(state.get("broker_latest_event_note", "")).strip()
    latest_event_exit_reason = str(state.get("broker_latest_event_exit_reason", "")).strip().upper()
    latest_event_realized_pnl_cash = float(state.get("broker_latest_event_realized_pnl_cash", 0.0) or 0.0)
    broker_block_reason = str(state.get("broker_block_reason", "")).strip() or "No block reason supplied."

    event_type = ""
    title = "XAUUSD AI Bot"
    subtitle = ""
    message = ""
    telegram_message = ""

    if last_autopilot_enabled is None:
        notification_state["last_autopilot_enabled"] = current_autopilot_enabled
    elif last_autopilot_enabled and not current_autopilot_enabled:
        event_type = "AUTOPILOT_DISABLED"
        subtitle = f"{symbol} autopilot disabled"
        message = broker_block_reason
        telegram_message = (
            f"{symbol} autopilot disabled\n"
            f"Reason: {broker_block_reason}"
        )
        notification_state["last_autopilot_enabled"] = current_autopilot_enabled
    elif (not last_autopilot_enabled) and current_autopilot_enabled:
        event_type = "AUTOPILOT_RESTORED"
        subtitle = f"{symbol} autopilot restored"
        message = "Broker execution is enabled again."
        telegram_message = (
            f"{symbol} autopilot restored\n"
            f"Broker execution is enabled again."
        )
        notification_state["last_autopilot_enabled"] = current_autopilot_enabled
    elif latest_event_action in {"OPEN", "CLOSE", "BREAKEVEN"} and latest_event_key:
        last_seen_event_key = str(notification_state.get("last_seen_event_key", "")).strip()
        if latest_event_key != last_seen_event_key:
            event_type = latest_event_action
            notification_state["last_seen_event_key"] = latest_event_key
            notification_state["last_seen_event_action"] = latest_event_action
            if latest_event_action == "OPEN":
                subtitle = f"{symbol} {latest_event_direction or 'TRADE'} opened"
                message = f"Time {latest_event_time} | {latest_event_note or 'Broker demo trade opened.'}"
                telegram_message = (
                    f"{symbol} {latest_event_direction or 'TRADE'} opened\n"
                    f"Time: {latest_event_time}\n"
                    f"{latest_event_note or 'Broker demo trade opened.'}"
                )
            elif latest_event_action == "CLOSE":
                subtitle = f"{symbol} {latest_event_direction or 'TRADE'} closed"
                message = (
                    f"Time {latest_event_time} | "
                    f"Reason {latest_event_exit_reason or 'UNKNOWN'} | "
                    f"PnL {latest_event_realized_pnl_cash:+.2f}"
                )
                telegram_message = (
                    f"{symbol} {latest_event_direction or 'TRADE'} closed\n"
                    f"Time: {latest_event_time}\n"
                    f"Reason: {latest_event_exit_reason or 'UNKNOWN'}\n"
                    f"PnL: {latest_event_realized_pnl_cash:+.2f}"
                )
            else:
                subtitle = f"{symbol} {latest_event_direction or 'TRADE'} breakeven"
                message = f"Time {latest_event_time} | Position protection moved after TP1."
                telegram_message = (
                    f"{symbol} {latest_event_direction or 'TRADE'} breakeven\n"
                    f"Time: {latest_event_time}\n"
                    f"Position protection moved after TP1."
                )

    if not event_type:
        notification_state["last_notification_status"] = "idle"
        notification_state["last_notification_message"] = "No new broker execution or autopilot state change."
        notification_state["last_notification_type"] = ""
        json_dump(notification_state, notification_state_output)
        return notification_state

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

    notification_state["last_notification_status"] = status_message
    notification_state["last_notification_message"] = detail_message
    notification_state["last_notification_type"] = event_type
    notification_state["last_notification_channels"] = {
        "desktop": {"status": desktop_status, "message": desktop_message},
        "telegram": {"status": telegram_status, "message": telegram_message_status},
    }
    notification_state["last_notified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if delivered and event_type in {"OPEN", "CLOSE", "BREAKEVEN"}:
        notification_state["last_notified_event_key"] = latest_event_key
    if delivered:
        notification_state["last_notification_delivered"] = True
    else:
        notification_state["last_notification_delivered"] = False
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
                    time.sleep(max(1, args.poll_seconds))
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
                state["notification_type"] = notification_state.get("last_notification_type", "")
                state["last_seen_event_key"] = notification_state.get("last_seen_event_key", "")
                state["last_autopilot_enabled"] = notification_state.get("last_autopilot_enabled", None)
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
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
