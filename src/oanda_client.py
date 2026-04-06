"""
Minimal OANDA candle client helpers for XAUUSD live research flows.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from pipeline_contract import display_path, ensure_parent_dir
from twelvedata_client import resolve_env_value


DEFAULT_OANDA_TOKEN_ENV = "OANDA_API_TOKEN"
DEFAULT_OANDA_MODE_ENV = "OANDA_ENV"
DEFAULT_OANDA_URL_ENV = "OANDA_API_URL"
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"


def normalize_oanda_instrument(instrument: str) -> str:
    return instrument.strip().upper().replace("/", "_")


def display_instrument(instrument: str) -> str:
    return normalize_oanda_instrument(instrument).replace("_", "/")


def resolve_api_token(env_name: str = DEFAULT_OANDA_TOKEN_ENV) -> str | None:
    return resolve_env_value(env_name)


def resolve_rest_url(
    url_env_name: str = DEFAULT_OANDA_URL_ENV,
    mode_env_name: str = DEFAULT_OANDA_MODE_ENV,
) -> str:
    explicit_url = resolve_env_value(url_env_name)
    if explicit_url:
        return explicit_url.rstrip("/")

    mode = (resolve_env_value(mode_env_name) or "practice").strip().lower()
    if mode == "live":
        return OANDA_LIVE_URL
    return OANDA_PRACTICE_URL


def fetch_instrument_candles(
    api_token: str,
    instrument: str = "XAU_USD",
    granularity: str = "M1",
    count: int = 5000,
    price: str = "M",
    rest_url: str | None = None,
) -> dict:
    instrument = normalize_oanda_instrument(instrument)
    rest_url = (rest_url or resolve_rest_url()).rstrip("/")
    params = urlencode(
        {
            "price": price,
            "granularity": granularity,
            "count": min(int(count), 5000),
        }
    )
    request = Request(
        f"{rest_url}/v3/instruments/{instrument}/candles?{params}",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    candles = payload.get("candles") or []
    if not candles:
        raise RuntimeError("OANDA returned no candles for the requested instrument.")

    price_key = {"M": "mid", "B": "bid", "A": "ask"}.get(price.upper(), "mid")
    normalized_values = []
    for candle in candles:
        if not candle.get("complete", True):
            continue
        ohlc = candle.get(price_key) or {}
        if not all(key in ohlc for key in ("o", "h", "l", "c")):
            continue
        normalized_values.append(
            {
                "datetime": candle["time"],
                "open": float(ohlc["o"]),
                "high": float(ohlc["h"]),
                "low": float(ohlc["l"]),
                "close": float(ohlc["c"]),
                "volume": float(candle.get("volume", 0.0)),
            }
        )

    if not normalized_values:
        raise RuntimeError("OANDA returned candles, but none were complete and parseable.")

    return {
        "meta": {
            "symbol": display_instrument(instrument),
            "instrument": instrument,
            "granularity": granularity,
            "price": price,
            "provider": "oanda",
        },
        "values": normalized_values,
        "has_volume": True,
        "volume_note": (
            "OANDA candle volume reflects the number of price updates in the candle, "
            "not centralized exchange volume."
        ),
    }


def build_ohlcv_frame(candles_payload: dict) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(candles_payload["values"]).rename(columns={"datetime": "time"})
    frame = frame.loc[:, ["time", "open", "high", "low", "close", "volume"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.isna().any().any():
        summary = frame.isna().sum()
        summary = summary[summary > 0].to_dict()
        raise ValueError(f"OANDA frame contains nulls after normalization: {summary}")

    frame = frame.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    return frame, {
        "rows": len(frame),
        "start": str(frame["time"].min()),
        "end": str(frame["time"].max()),
        "volume_mode": "source",
    }


def fetch_and_write_candles_csv(
    api_token: str,
    output_path: str,
    instrument: str = "XAU_USD",
    granularity: str = "M1",
    count: int = 5000,
    price: str = "M",
    rest_url: str | None = None,
) -> dict:
    candles_payload = fetch_instrument_candles(
        api_token=api_token,
        instrument=instrument,
        granularity=granularity,
        count=count,
        price=price,
        rest_url=rest_url,
    )
    frame, frame_stats = build_ohlcv_frame(candles_payload)
    output_file = ensure_parent_dir(output_path)
    frame.to_csv(output_file, index=False)
    return {
        "output_path": output_file,
        "output_display": display_path(output_file),
        "symbol": candles_payload["meta"]["symbol"],
        "instrument": candles_payload["meta"]["instrument"],
        "interval": candles_payload["meta"]["granularity"],
        "timezone": "UTC",
        "has_source_volume": candles_payload["has_volume"],
        "volume_note": candles_payload["volume_note"],
        **frame_stats,
    }
