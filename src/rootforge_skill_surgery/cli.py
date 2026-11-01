"""Command-line entry points for stage-gated research experiments."""
from __future__ import annotations

import argparse
import json
import sys
from .contracts import ContractError, load_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rootforge-skill-surgery")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a Stage 0 experiment contract")
    validate.add_argument("contract", help="JSON contract path")
    validate.add_argument("--minimum-probes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "validate":
        try:
            report = load_contract(args.contract, args.minimum_probes)
        except (ContractError, OSError, json.JSONDecodeError) as error:
            print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
            return 2
        print(json.dumps({"valid": True, **report}, sort_keys=True))
        return 0
    return 2
