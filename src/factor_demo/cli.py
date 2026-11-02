"""Modern raw-data research CLI; historical feature-only commands live under legacy."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def main():
    # Configure SDK metadata before importing NOOA, including serve/research paths.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if len(sys.argv) > 1 and sys.argv[1] == "legacy":
        from .legacy.cli import main as legacy_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return legacy_main()
    parser = argparse.ArgumentParser(
        description="NOOA schema-driven industrial data research and explicit simulation benchmarks"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--warm-font-cache", action="store_true")
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("probe")
    p.add_argument("--provider", default="glm")
    p = sub.add_parser("generate-etch", aliases=["generate"])
    p.add_argument("--data", type=Path, default=Path("data/etching-v3"))
    p.add_argument("--runs", type=int, default=360)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--sample-period", type=float, default=0.5)
    p.add_argument("--profile", choices=["balanced_v3", "legacy_v2"], default="balanced_v3")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument(
        "--scenario",
        default="edge_leak_contact",
        choices=[
            "edge_leak_contact",
            "pump_restriction",
            "surface_condition",
            "edge_ring",
            "common_mode_only",
            "stable_control",
        ],
    )
    for name in ["validate-etch", "preview-etch"]:
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("data/etching-v3"))
    for name, aliases in [("research", ["autoresearch"]), ("resume", ["autoresearch-chat"])]:
        p = sub.add_parser(name, aliases=aliases)
        p.add_argument("--run", type=Path, required=True)
        if name == "research":
            p.add_argument("--data", type=Path, required=True)
        p.add_argument("--provider", default="glm")
        p.add_argument("--message", default="")
        p.add_argument("--message-file", type=Path)
        p.add_argument("--mode", choices=["research", "discuss"], default="research")
        p.add_argument("--speaker", choices=["user", "simulated_expert"], default="user")
        p.add_argument("--settings", type=Path, default=Path("configs/runtime.json"))
        p.add_argument("--timeout", type=int, default=3600)
    for name, aliases in [
        ("audit", ["autoresearch-audit"]),
        ("finalize-etch", []),
        ("evaluation", []),
    ]:
        p = sub.add_parser(name, aliases=aliases)
        p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        from .native_sandbox import sandbox_status
        from .sandboxing import dotenv_warnings, execution_info, prepare_font_cache

        info = execution_info()
        info["orchestration_sandbox"] = sandbox_status()
        info["warnings"] = dotenv_warnings()
        if args.warm_font_cache:
            info.update(prepare_font_cache())
        print(json.dumps(info, indent=2))
        return
    if args.command == "serve":
        from .web import serve

        return serve(args.host, args.port)
    if args.command == "probe":
        from .providers import load_provider, probe

        result = probe(load_provider(args.provider))
    elif args.command in {"generate", "generate-etch"}:
        from .etch_dataset import generate_etching

        result = generate_etching(
            args.data,
            args.runs,
            args.seed,
            args.sample_period,
            not args.no_controls,
            scenario=args.scenario,
            profile=args.profile,
        )
    elif args.command == "validate-etch":
        from .etch_validation import validate_etching

        result = validate_etching(args.data)
    elif args.command == "preview-etch":
        from .etch_preview import preview_etching

        result = str(preview_etching(args.data))
    elif args.command in {"research", "autoresearch", "resume", "autoresearch-chat"}:
        from .autoresearch import research_turn

        message = args.message_file.read_text() if args.message_file else args.message
        result = asyncio.run(
            research_turn(
                args.run,
                args.provider,
                message,
                args.mode,
                args.settings,
                args.timeout,
                args.data if args.command in {"research", "autoresearch"} else None,
                args.speaker,
            )
        )
    elif args.command in {"audit", "autoresearch-audit"}:
        from .research_audit import audit_session

        result = audit_session(args.run)
    elif args.command == "finalize-etch":
        from .evaluation import finalize_evaluation

        result = asyncio.run(finalize_evaluation(args.run))
    else:
        from .evaluation import evaluation_state

        result = evaluation_state(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
