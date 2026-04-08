"""
Store MT5 credentials in macOS Keychain and launch the MT5 live pipeline without putting secrets in repo files.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from mt5_keychain import (
    DEFAULT_MT5_KEYCHAIN_SERVICE,
    delete_mt5_credentials,
    load_mt5_credentials,
    store_mt5_credentials,
    temporary_mt5_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service",
        default=DEFAULT_MT5_KEYCHAIN_SERVICE,
        help="macOS Keychain service name used to store the MT5 credentials.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    store_parser = subparsers.add_parser("store", help="Save or update the MT5 login, password, and server in Keychain.")
    store_parser.add_argument("--login", required=True, help="Numeric MT5 account login.")
    store_parser.add_argument("--server", required=True, help="Broker server name, for example MetaQuotes-Demo.")
    store_parser.add_argument("--password", default="", help="Optional MT5 password. If omitted, a secure prompt is used.")

    subparsers.add_parser("status", help="Show whether MT5 login, password, and server are present in Keychain.")
    subparsers.add_parser("delete", help="Delete the stored MT5 login, password, and server from Keychain.")

    run_parser = subparsers.add_parser(
        "run-live-pipeline",
        help="Launch src/run_live_mt5_pipeline.py with MT5 credentials loaded from Keychain only for this process.",
    )
    run_parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to src/run_live_mt5_pipeline.py. Prefix them with -- when calling this wrapper.",
    )

    research_parser = subparsers.add_parser(
        "run-research-pipeline",
        help="Launch src/run_mt5_research_pipeline.py with MT5 credentials loaded from Keychain only for this process.",
    )
    research_parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to src/run_mt5_research_pipeline.py. Prefix them with -- when calling this wrapper.",
    )
    return parser.parse_args()


def _normalize_pipeline_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def handle_store(service: str, login: str, server: str, password: str) -> None:
    cleaned_login = login.strip()
    if not cleaned_login.isdigit():
        raise SystemExit("MT5 login must be numeric.")

    cleaned_server = server.strip()
    if not cleaned_server:
        raise SystemExit("MT5 server cannot be empty.")

    secret = password or getpass.getpass("MT5 password: ")
    if not secret:
        raise SystemExit("MT5 password cannot be empty.")

    store_mt5_credentials(
        login=cleaned_login,
        password=secret,
        server=cleaned_server,
        service=service,
    )
    print("MT5 credentials stored in macOS Keychain.")
    print(f"Service : {service}")
    print(f"Login   : {cleaned_login}")
    print(f"Server  : {cleaned_server}")
    print("Password: stored")


def handle_status(service: str) -> None:
    credentials = load_mt5_credentials(service=service)
    print("MT5 Keychain status")
    print(f"Service : {service}")
    print(f"Login   : {credentials.login or 'missing'}")
    print(f"Server  : {credentials.server or 'missing'}")
    print(f"Password: {'stored' if credentials.password else 'missing'}")
    missing = credentials.missing_fields()
    if missing:
        print(f"Ready   : no ({', '.join(missing)} missing)")
    else:
        print("Ready   : yes")


def handle_delete(service: str) -> None:
    results = delete_mt5_credentials(service=service)
    print("Removed MT5 Keychain entries:")
    print(f"Service : {service}")
    for field, deleted in results.items():
        print(f"{field:<8}: {'deleted' if deleted else 'not found'}")


def handle_run_live_pipeline(service: str, pipeline_args: list[str]) -> None:
    credentials = load_mt5_credentials(service=service)
    missing = credentials.missing_fields()
    if missing:
        raise SystemExit(
            "MT5 Keychain is missing required fields for the secure launcher: "
            f"{', '.join(missing)}. Run `python src/mt5_keychain_cli.py store --login <id> --server <name>` first."
        )

    from run_live_mt5_pipeline import parse_args as parse_live_pipeline_args
    from run_live_mt5_pipeline import print_summary, run_from_args

    forwarded_args = _normalize_pipeline_args(pipeline_args)
    parsed_pipeline_args = parse_live_pipeline_args(forwarded_args)

    print("=" * 70)
    print("XAUUSD MT5 LIVE PIPELINE (KEYCHAIN)")
    print("=" * 70)
    print(f"Keychain service : {service}")
    print(f"MT5 login        : {credentials.login}")
    print(f"MT5 server       : {credentials.server}")
    print("MT5 password     : loaded from Keychain for this process only")
    print()

    with temporary_mt5_env(credentials):
        report = run_from_args(parsed_pipeline_args)
    print_summary(report)


def handle_run_research_pipeline(service: str, pipeline_args: list[str]) -> None:
    credentials = load_mt5_credentials(service=service)
    missing = credentials.missing_fields()
    if missing:
        raise SystemExit(
            "MT5 Keychain is missing required fields for the secure launcher: "
            f"{', '.join(missing)}. Run `python src/mt5_keychain_cli.py store --login <id> --server <name>` first."
        )

    from run_mt5_research_pipeline import parse_args as parse_research_args
    from run_mt5_research_pipeline import print_research_summary, run_from_args

    forwarded_args = _normalize_pipeline_args(pipeline_args)
    parsed_pipeline_args = parse_research_args(forwarded_args)

    print("=" * 70)
    print("XAUUSD MT5 RESEARCH PIPELINE (KEYCHAIN)")
    print("=" * 70)
    print(f"Keychain service : {service}")
    print(f"MT5 login        : {credentials.login}")
    print(f"MT5 server       : {credentials.server}")
    print("MT5 password     : loaded from Keychain for this process only")
    print()

    with temporary_mt5_env(credentials):
        report = run_from_args(parsed_pipeline_args)
    print_research_summary(report)


def main() -> None:
    args = parse_args()
    if args.command == "store":
        handle_store(service=args.service, login=args.login, server=args.server, password=args.password)
        return
    if args.command == "status":
        handle_status(service=args.service)
        return
    if args.command == "delete":
        handle_delete(service=args.service)
        return
    if args.command == "run-live-pipeline":
        handle_run_live_pipeline(service=args.service, pipeline_args=args.pipeline_args)
        return
    if args.command == "run-research-pipeline":
        handle_run_research_pipeline(service=args.service, pipeline_args=args.pipeline_args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
