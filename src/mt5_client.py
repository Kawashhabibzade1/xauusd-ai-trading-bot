"""
Minimal MetaTrader 5 client helpers for local live XAUUSD research flows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    from pandas.errors import EmptyDataError, ParserError
except Exception:  # pragma: no cover - optional for installer-only paths
    pd = None  # type: ignore[assignment]
    EmptyDataError = ValueError  # type: ignore[assignment]
    ParserError = ValueError  # type: ignore[assignment]
from env_utils import resolve_env_value
from pipeline_contract import display_path, ensure_parent_dir, resolve_repo_path


DEFAULT_MT5_TERMINAL_ENV = "MT5_TERMINAL_PATH"
DEFAULT_MT5_LOGIN_ENV = "MT5_LOGIN"
DEFAULT_MT5_PASSWORD_ENV = "MT5_PASSWORD"
DEFAULT_MT5_SERVER_ENV = "MT5_SERVER"
DEFAULT_MT5_EXPORT_ENV = "MT5_EXPORT_FILE"
DEFAULT_MT5_EXPORT_FILENAME = "xauusd_mt5_live.csv"
DEFAULT_MT5_ACCOUNT_SNAPSHOT_ENV = "MT5_ACCOUNT_SNAPSHOT_FILE"
DEFAULT_MT5_ACCOUNT_SNAPSHOT_FILENAME = "config/mt5_account_snapshot.csv"
DEFAULT_MT5_WINE_PREFIX = (
    Path.home() / "Library" / "Application Support" / "net.metaquotes.wine.metatrader5"
)
DEFAULT_MT5_APP_BUNDLE = Path("/Applications/MetaTrader 5.app")
DEFAULT_MT5_PROGRAM_DIR = (
    DEFAULT_MT5_WINE_PREFIX / "drive_c" / "Program Files" / "MetaTrader 5"
)
DEFAULT_MT5_FILES_DIR = DEFAULT_MT5_PROGRAM_DIR / "MQL5" / "Files"
DEFAULT_MT5_EXPERTS_DIR = DEFAULT_MT5_PROGRAM_DIR / "MQL5" / "Experts"
DEFAULT_MT5_METAEDITOR_EXE = DEFAULT_MT5_PROGRAM_DIR / "MetaEditor64.exe"
DEFAULT_MT5_TERMINAL_EXE = DEFAULT_MT5_PROGRAM_DIR / "terminal64.exe"
DEFAULT_MT5_WINE64 = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "pandas is not available in this environment. "
            "Install the project requirements before using MT5 data helpers."
        )
    return pd


def _atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        shutil.copy2(source, temp_target)
        os.replace(temp_target, target)
    finally:
        if temp_target.exists():
            temp_target.unlink()


def _import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local platform/package
        raise RuntimeError(
            "MetaTrader5 Python package is not available in this environment. "
            "This MT5 live path only works on a local machine with the MetaTrader terminal and the MetaTrader5 Python package installed."
        ) from exc
    return mt5


def resolve_terminal_path(env_name: str = DEFAULT_MT5_TERMINAL_ENV) -> str | None:
    return resolve_env_value(env_name)


def resolve_login(env_name: str = DEFAULT_MT5_LOGIN_ENV) -> int | None:
    value = resolve_env_value(env_name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"MT5 login in {env_name} must be numeric.") from exc


def resolve_password(env_name: str = DEFAULT_MT5_PASSWORD_ENV) -> str | None:
    return resolve_env_value(env_name)


def resolve_server(env_name: str = DEFAULT_MT5_SERVER_ENV) -> str | None:
    return resolve_env_value(env_name)


def normalize_mt5_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "").replace("_", "")


def get_mt5_wine_prefix() -> Path | None:
    env_path = resolve_env_value("MT5_WINE_PREFIX")
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.exists() else None
    return DEFAULT_MT5_WINE_PREFIX if DEFAULT_MT5_WINE_PREFIX.exists() else None


def get_mt5_program_dir() -> Path | None:
    prefix = get_mt5_wine_prefix()
    if prefix is None:
        return None
    program_dir = prefix / "drive_c" / "Program Files" / "MetaTrader 5"
    return program_dir if program_dir.exists() else None


def get_mt5_files_dir() -> Path | None:
    program_dir = get_mt5_program_dir()
    if program_dir is None:
        return None
    files_dir = program_dir / "MQL5" / "Files"
    return files_dir if files_dir.exists() else None


def get_mt5_experts_dir() -> Path | None:
    program_dir = get_mt5_program_dir()
    if program_dir is None:
        return None
    experts_dir = program_dir / "MQL5" / "Experts"
    return experts_dir if experts_dir.exists() else None


def get_mt5_scripts_dir() -> Path | None:
    program_dir = get_mt5_program_dir()
    if program_dir is None:
        return None
    scripts_dir = program_dir / "MQL5" / "Scripts"
    return scripts_dir if scripts_dir.exists() else None


def get_mt5_metaeditor_exe() -> Path | None:
    program_dir = get_mt5_program_dir()
    if program_dir is None:
        return None
    editor = program_dir / "MetaEditor64.exe"
    return editor if editor.exists() else None


def get_mt5_terminal_exe() -> Path | None:
    env_path = resolve_terminal_path()
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
    program_dir = get_mt5_program_dir()
    if program_dir is None:
        return None
    terminal = program_dir / "terminal64.exe"
    return terminal if terminal.exists() else None


def get_mt5_wine64() -> Path | None:
    return DEFAULT_MT5_WINE64 if DEFAULT_MT5_WINE64.exists() else None


def get_mt5_app_bundle() -> Path | None:
    return DEFAULT_MT5_APP_BUNDLE if DEFAULT_MT5_APP_BUNDLE.exists() else None


def resolve_export_file_path(
    path_like: str | Path | None = None,
    env_name: str = DEFAULT_MT5_EXPORT_ENV,
    filename: str = DEFAULT_MT5_EXPORT_FILENAME,
) -> Path:
    if path_like:
        return Path(path_like).expanduser()

    env_path = resolve_env_value(env_name)
    if env_path:
        return Path(env_path).expanduser()

    files_dir = get_mt5_files_dir()
    if files_dir is None:
        raise RuntimeError(
            "Could not find the local MT5 MQL5/Files directory. "
            "Set MT5_EXPORT_FILE to the CSV written by the MT5 exporter."
        )
    return files_dir / filename


def resolve_account_snapshot_file_path(
    path_like: str | Path | None = None,
    env_name: str = DEFAULT_MT5_ACCOUNT_SNAPSHOT_ENV,
    filename: str = DEFAULT_MT5_ACCOUNT_SNAPSHOT_FILENAME,
) -> Path:
    if path_like:
        return Path(path_like).expanduser()

    env_path = resolve_env_value(env_name)
    if env_path:
        return Path(env_path).expanduser()

    files_dir = get_mt5_files_dir()
    if files_dir is None:
        raise RuntimeError(
            "Could not find the local MT5 MQL5/Files directory. "
            "Set MT5_ACCOUNT_SNAPSHOT_FILE to the CSV written by the MT5 exporter."
        )
    return files_dir / filename


def _build_payload_from_normalized(
    normalized: pd.DataFrame,
    symbol: str,
    timeframe: str,
    provider: str,
    volume_source: str,
    volume_note: str,
    source_path: str | None = None,
) -> dict:
    values = []
    for _, row in normalized.iterrows():
        values.append(
            {
                "datetime": row["time"].isoformat(sep=" "),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    payload = {
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe.strip().upper(),
            "provider": provider,
        },
        "values": values,
        "frame": normalized,
        "has_volume": True,
        "volume_source": volume_source,
        "volume_note": volume_note,
    }
    if source_path is not None:
        payload["meta"]["source_path"] = source_path
    return payload


def _initialize_connection(
    terminal_path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> Any:
    mt5 = _import_mt5()
    kwargs: dict[str, Any] = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    if login is not None:
        kwargs["login"] = login
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server

    initialized = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
    if not initialized:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    return mt5


def _timeframe_constant(mt5: Any, timeframe: str) -> Any:
    attr = f"TIMEFRAME_{timeframe.strip().upper()}"
    if not hasattr(mt5, attr):
        raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
    return getattr(mt5, attr)


def fetch_recent_rates(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    count: int = 5000,
    terminal_path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    prefer_real_volume: bool = False,
) -> dict:
    normalized_symbol = normalize_mt5_symbol(symbol)
    mt5 = _initialize_connection(
        terminal_path=terminal_path,
        login=login,
        password=password,
        server=server,
    )
    try:
        if not mt5.symbol_select(normalized_symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed for {normalized_symbol}: {mt5.last_error()}")

        rates = mt5.copy_rates_from_pos(
            normalized_symbol,
            _timeframe_constant(mt5, timeframe),
            0,
            int(count),
        )
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 copy_rates_from_pos returned no data: {mt5.last_error()}")

        frame = pd.DataFrame(rates)
    finally:
        mt5.shutdown()

    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    volume_source = "tick_volume"
    if prefer_real_volume and "real_volume" in frame.columns and (frame["real_volume"].astype(float) > 0).any():
        volume_series = pd.to_numeric(frame["real_volume"], errors="coerce")
        volume_source = "real_volume"
    else:
        volume_series = pd.to_numeric(frame["tick_volume"], errors="coerce")
        if volume_series.fillna(0).le(0).all() and "real_volume" in frame.columns:
            fallback = pd.to_numeric(frame["real_volume"], errors="coerce")
            if fallback.fillna(0).gt(0).any():
                volume_series = fallback
                volume_source = "real_volume"

    normalized = pd.DataFrame(
        {
            "time": frame["time"],
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": volume_series,
        }
    )
    if normalized.isna().any().any():
        summary = normalized.isna().sum()
        summary = summary[summary > 0].to_dict()
        raise ValueError(f"MT5 frame contains nulls after normalization: {summary}")

    normalized = normalized.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    return _build_payload_from_normalized(
        normalized=normalized,
        symbol=normalized_symbol,
        timeframe=timeframe,
        provider="mt5_local",
        volume_source=volume_source,
        volume_note=(
            "MT5 local feed is using real_volume."
            if volume_source == "real_volume"
            else "MT5 local feed is using tick_volume as the model's live volume input."
        ),
    )


def fetch_mt5_account_context(
    symbol: str = "XAUUSD",
    terminal_path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> dict[str, Any]:
    normalized_symbol = normalize_mt5_symbol(symbol)
    mt5 = _initialize_connection(
        terminal_path=terminal_path,
        login=login,
        password=password,
        server=server,
    )
    try:
        if not mt5.symbol_select(normalized_symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed for {normalized_symbol}: {mt5.last_error()}")

        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError(f"MT5 account_info failed: {mt5.last_error()}")

        symbol_info = mt5.symbol_info(normalized_symbol)
        if symbol_info is None:
            raise RuntimeError(f"MT5 symbol_info failed for {normalized_symbol}: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    return {
        "symbol": normalized_symbol,
        "login": int(getattr(account_info, "login", 0) or 0),
        "server": str(getattr(account_info, "server", "") or ""),
        "currency": str(getattr(account_info, "currency", "") or ""),
        "balance": float(getattr(account_info, "balance", 0.0) or 0.0),
        "equity": float(getattr(account_info, "equity", 0.0) or 0.0),
        "leverage": int(getattr(account_info, "leverage", 0) or 0),
        "contract_size": float(getattr(symbol_info, "trade_contract_size", 0.0) or 0.0),
        "volume_min": float(getattr(symbol_info, "volume_min", 0.0) or 0.0),
        "volume_max": float(getattr(symbol_info, "volume_max", 0.0) or 0.0),
        "volume_step": float(getattr(symbol_info, "volume_step", 0.0) or 0.0),
    }


def load_mt5_account_snapshot(
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    snapshot_path = resolve_account_snapshot_file_path(input_path)
    if not snapshot_path.exists():
        raise RuntimeError(
            f"MT5 account snapshot CSV not found at {snapshot_path}. "
            "Run the MT5 exporter/bot with account snapshot export enabled first."
        )

    frame = pd.read_csv(snapshot_path)
    if frame.empty:
        raise RuntimeError(f"MT5 account snapshot CSV at {snapshot_path} is empty.")

    latest = frame.iloc[-1].to_dict()
    return {
        "source": "mt5_snapshot_csv",
        "snapshot_path": str(snapshot_path),
        "time_utc": str(latest.get("time_utc", "")),
        "symbol": str(latest.get("symbol", "") or ""),
        "login": int(pd.to_numeric(latest.get("login", 0), errors="coerce") or 0),
        "server": str(latest.get("server", "") or ""),
        "currency": str(latest.get("currency", "") or ""),
        "balance": float(pd.to_numeric(latest.get("balance", 0.0), errors="coerce") or 0.0),
        "equity": float(pd.to_numeric(latest.get("equity", 0.0), errors="coerce") or 0.0),
        "leverage": int(pd.to_numeric(latest.get("leverage", 0), errors="coerce") or 0),
        "contract_size": float(pd.to_numeric(latest.get("contract_size", 0.0), errors="coerce") or 0.0),
        "volume_min": float(pd.to_numeric(latest.get("volume_min", 0.0), errors="coerce") or 0.0),
        "volume_max": float(pd.to_numeric(latest.get("volume_max", 0.0), errors="coerce") or 0.0),
        "volume_step": float(pd.to_numeric(latest.get("volume_step", 0.0), errors="coerce") or 0.0),
    }


def load_exported_rates_csv(
    input_path: str | Path | None = None,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    max_read_attempts: int = 5,
    retry_delay_seconds: float = 0.25,
) -> dict:
    from validate_merged_data import load_and_standardize

    export_path = resolve_export_file_path(input_path)
    if not export_path.exists():
        raise RuntimeError(
            f"MT5 export CSV not found at {export_path}. "
            "Run the MT5 exporter first or set MT5_EXPORT_FILE to the correct CSV path."
        )

    last_error: Exception | None = None
    normalized: pd.DataFrame | None = None
    for attempt in range(max(1, int(max_read_attempts))):
        try:
            normalized = load_and_standardize(str(export_path))
            break
        except (EmptyDataError, ParserError, ValueError) as exc:
            error_text = str(exc)
            is_transient_null_parse = "contains nulls after parsing" in error_text or "Failed to parse" in error_text
            if not isinstance(exc, (EmptyDataError, ParserError)) and not is_transient_null_parse:
                raise
            last_error = exc
            if attempt >= max_read_attempts - 1:
                break
            time.sleep(max(0.0, float(retry_delay_seconds)))
    if normalized is None:
        raise RuntimeError(
            f"MT5 export CSV at {export_path} could not be read after {max_read_attempts} attempts. "
            "The exporter may be rewriting the file right now."
        ) from last_error

    return _build_payload_from_normalized(
        normalized=normalized,
        symbol=normalize_mt5_symbol(symbol),
        timeframe=timeframe,
        provider="mt5_export",
        volume_source="volume",
        volume_note="MT5 exporter feed is using the volume column written by the MT5 live exporter.",
        source_path=str(export_path),
    )


def fetch_and_write_rates_csv(
    output_path: str,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    count: int = 5000,
    terminal_path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    prefer_real_volume: bool = False,
) -> dict:
    payload = fetch_recent_rates(
        symbol=symbol,
        timeframe=timeframe,
        count=count,
        terminal_path=terminal_path,
        login=login,
        password=password,
        server=server,
        prefer_real_volume=prefer_real_volume,
    )
    frame = payload["frame"].copy()
    output_file = ensure_parent_dir(output_path)
    frame.to_csv(output_file, index=False)
    return {
        "output_path": output_file,
        "output_display": display_path(output_file),
        "symbol": payload["meta"]["symbol"],
        "interval": payload["meta"]["timeframe"],
        "timezone": "UTC",
        "has_source_volume": payload["has_volume"],
        "volume_mode": payload["volume_source"],
        "volume_note": payload["volume_note"],
        "rows": len(frame),
        "start": str(frame["time"].min()),
        "end": str(frame["time"].max()),
    }


def copy_exported_rates_csv(
    output_path: str,
    input_path: str | Path | None = None,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
) -> dict:
    payload = load_exported_rates_csv(input_path=input_path, symbol=symbol, timeframe=timeframe)
    frame = payload["frame"].copy()
    output_file = ensure_parent_dir(output_path)
    frame.to_csv(output_file, index=False)
    source_path = payload["meta"].get("source_path")
    return {
        "output_path": output_file,
        "output_display": display_path(output_file),
        "symbol": payload["meta"]["symbol"],
        "interval": payload["meta"]["timeframe"],
        "timezone": "UTC",
        "has_source_volume": payload["has_volume"],
        "volume_mode": payload["volume_source"],
        "volume_note": payload["volume_note"],
        "rows": len(frame),
        "start": str(frame["time"].min()),
        "end": str(frame["time"].max()),
        "provider": payload["meta"]["provider"],
        "source_path": source_path,
    }


def install_exporter_source(
    source_path: str | Path,
    relative_dir: str = "OpenAI",
) -> dict:
    experts_dir = get_mt5_experts_dir()
    if experts_dir is None:
        raise RuntimeError("Could not find the MT5 MQL5/Experts directory on this machine.")

    source = resolve_repo_path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)

    target_dir = experts_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / source.name
    shutil.copy2(source, target_file)
    return {
        "source": str(source),
        "target": str(target_file),
    }


def restart_mt5_terminal(
    relaunch_delay_seconds: float = 3.0,
    settle_seconds: float = 5.0,
) -> dict[str, Any]:
    app_bundle = get_mt5_app_bundle()
    if app_bundle is None:
        raise RuntimeError("Could not find /Applications/MetaTrader 5.app on this machine.")

    osascript = shutil.which("osascript")
    open_cmd = shutil.which("open")
    if osascript is None or open_cmd is None:
        raise RuntimeError("Both osascript and open are required to restart MetaTrader 5 on macOS.")

    bundle_id = "net.metaquotes.wine.MetaTrader5"
    quit_result = subprocess.run(
        [osascript, "-e", f'tell application id "{bundle_id}" to quit'],
        capture_output=True,
        text=True,
        check=False,
    )
    time.sleep(max(float(relaunch_delay_seconds), 0.0))
    launch_result = subprocess.run(
        [open_cmd, "-a", str(app_bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    time.sleep(max(float(settle_seconds), 0.0))
    return {
        "app_bundle": str(app_bundle),
        "quit_returncode": int(quit_result.returncode),
        "quit_stdout": quit_result.stdout.strip(),
        "quit_stderr": quit_result.stderr.strip(),
        "launch_returncode": int(launch_result.returncode),
        "launch_stdout": launch_result.stdout.strip(),
        "launch_stderr": launch_result.stderr.strip(),
        "success": launch_result.returncode == 0,
    }


def wait_for_mt5_file(
    relative_path: str,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    files_dir = get_mt5_files_dir()
    if files_dir is None:
        raise RuntimeError("Could not find the MT5 MQL5/Files directory on this machine.")

    target = files_dir / relative_path
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while time.monotonic() <= deadline:
        if target.exists():
            stat = target.stat()
            return {
                "path": str(target),
                "exists": True,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        time.sleep(max(float(poll_seconds), 0.1))

    return {
        "path": str(target),
        "exists": target.exists(),
    }


def sync_mt5_file_artifact(path_like: str | Path) -> dict[str, str]:
    files_dir = get_mt5_files_dir()
    if files_dir is None:
        raise RuntimeError("Could not find the MT5 MQL5/Files directory on this machine.")

    source = resolve_repo_path(path_like)
    if not source.exists():
        raise FileNotFoundError(source)

    repo_mt5_root = resolve_repo_path("mt5_expert_advisor/Files")
    try:
        relative = source.relative_to(repo_mt5_root)
    except ValueError as exc:
        raise ValueError(
            f"{display_path(source)} is not inside mt5_expert_advisor/Files and cannot be synced to MT5."
        ) from exc

    target = files_dir / relative
    if target.exists() and target.resolve() == source.resolve():
        return {
            "source": str(source),
            "target": str(target),
            "relative": str(relative),
        }
    _atomic_copy_file(source, target)
    return {
        "source": str(source),
        "target": str(target),
        "relative": str(relative),
    }


def sync_mt5_file_artifacts(paths: list[str | Path]) -> list[dict[str, str]]:
    return [sync_mt5_file_artifact(path_like) for path_like in paths]
