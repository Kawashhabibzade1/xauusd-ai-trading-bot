"""
macOS Keychain helpers for storing MetaTrader 5 demo credentials locally.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass

DEFAULT_MT5_LOGIN_ENV = "MT5_LOGIN"
DEFAULT_MT5_PASSWORD_ENV = "MT5_PASSWORD"
DEFAULT_MT5_SERVER_ENV = "MT5_SERVER"
DEFAULT_MT5_KEYCHAIN_SERVICE = "xauusd-ai-trading-bot.mt5"
NOT_FOUND_SNIPPETS = (
    "could not be found",
    "The specified item could not be found in the keychain",
)
KEYCHAIN_RECORDS = {
    "login": {
        "account": "mt5_login",
        "label": "XAUUSD AI Bot MT5 Login",
        "env_name": DEFAULT_MT5_LOGIN_ENV,
    },
    "password": {
        "account": "mt5_password",
        "label": "XAUUSD AI Bot MT5 Password",
        "env_name": DEFAULT_MT5_PASSWORD_ENV,
    },
    "server": {
        "account": "mt5_server",
        "label": "XAUUSD AI Bot MT5 Server",
        "env_name": DEFAULT_MT5_SERVER_ENV,
    },
}


@dataclass(frozen=True)
class MT5KeychainCredentials:
    login: str | None
    password: str | None
    server: str | None

    def as_env_map(self) -> dict[str, str]:
        env_map: dict[str, str] = {}
        if self.login:
            env_map[DEFAULT_MT5_LOGIN_ENV] = self.login
        if self.password:
            env_map[DEFAULT_MT5_PASSWORD_ENV] = self.password
        if self.server:
            env_map[DEFAULT_MT5_SERVER_ENV] = self.server
        return env_map

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.login:
            missing.append("login")
        if not self.password:
            missing.append("password")
        if not self.server:
            missing.append("server")
        return missing


def _run_security_command(args: list[str], allow_missing: bool = False) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The macOS `security` CLI is not available on this machine. "
            "The MT5 Keychain helper only works on macOS with Keychain access."
        ) from exc

    if result.returncode == 0:
        return result.stdout.rstrip("\n")

    combined_output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part).strip()
    if allow_missing and any(snippet in combined_output for snippet in NOT_FOUND_SNIPPETS):
        return None

    raise RuntimeError(
        f"macOS Keychain command failed with exit code {result.returncode}: {' '.join(args)}\n{combined_output}"
    )


def store_keychain_value(field: str, value: str, service: str = DEFAULT_MT5_KEYCHAIN_SERVICE) -> None:
    if field not in KEYCHAIN_RECORDS:
        raise KeyError(f"Unsupported MT5 credential field: {field}")
    if not value:
        raise ValueError(f"Cannot store an empty MT5 {field} value in Keychain.")

    record = KEYCHAIN_RECORDS[field]
    _run_security_command(
        [
            "security",
            "add-generic-password",
            "-U",
            "-s",
            service,
            "-a",
            str(record["account"]),
            "-l",
            str(record["label"]),
            "-w",
            value,
        ]
    )


def read_keychain_value(field: str, service: str = DEFAULT_MT5_KEYCHAIN_SERVICE) -> str | None:
    if field not in KEYCHAIN_RECORDS:
        raise KeyError(f"Unsupported MT5 credential field: {field}")

    record = KEYCHAIN_RECORDS[field]
    return _run_security_command(
        [
            "security",
            "find-generic-password",
            "-w",
            "-s",
            service,
            "-a",
            str(record["account"]),
        ],
        allow_missing=True,
    )


def delete_keychain_value(field: str, service: str = DEFAULT_MT5_KEYCHAIN_SERVICE) -> bool:
    if field not in KEYCHAIN_RECORDS:
        raise KeyError(f"Unsupported MT5 credential field: {field}")

    record = KEYCHAIN_RECORDS[field]
    deleted = _run_security_command(
        [
            "security",
            "delete-generic-password",
            "-s",
            service,
            "-a",
            str(record["account"]),
        ],
        allow_missing=True,
    )
    return deleted is not None


def load_mt5_credentials(service: str = DEFAULT_MT5_KEYCHAIN_SERVICE) -> MT5KeychainCredentials:
    return MT5KeychainCredentials(
        login=read_keychain_value("login", service=service),
        password=read_keychain_value("password", service=service),
        server=read_keychain_value("server", service=service),
    )


def store_mt5_credentials(
    login: str,
    password: str,
    server: str,
    service: str = DEFAULT_MT5_KEYCHAIN_SERVICE,
) -> MT5KeychainCredentials:
    store_keychain_value("login", login, service=service)
    store_keychain_value("password", password, service=service)
    store_keychain_value("server", server, service=service)
    return load_mt5_credentials(service=service)


def delete_mt5_credentials(service: str = DEFAULT_MT5_KEYCHAIN_SERVICE) -> dict[str, bool]:
    return {field: delete_keychain_value(field, service=service) for field in KEYCHAIN_RECORDS}


@contextmanager
def temporary_mt5_env(credentials: MT5KeychainCredentials):
    previous = {
        DEFAULT_MT5_LOGIN_ENV: os.environ.get(DEFAULT_MT5_LOGIN_ENV),
        DEFAULT_MT5_PASSWORD_ENV: os.environ.get(DEFAULT_MT5_PASSWORD_ENV),
        DEFAULT_MT5_SERVER_ENV: os.environ.get(DEFAULT_MT5_SERVER_ENV),
    }
    try:
        for env_name, value in credentials.as_env_map().items():
            os.environ[env_name] = value
        yield
    finally:
        for env_name, prior in previous.items():
            if prior is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = prior
