#!/usr/bin/env python3
"""
Apply optimized parameters safely to strategy configs.

Usage:
  python scripts/optimizer_apply.py --result artifacts/optimization/fibonacci_adx_optimization.json --yes --mcp

Behavior:
- Loads an optimization JSON or pickle with `best_params` and `strategy_name`.
- Shows a dry-run summary of changes and requires confirmation unless `--yes`.
- Validates params against the optimizer's ParameterSpec bounds.
- Persists changes to `configs/strategies/{strategy}.yaml` (creates file if missing).
- Optionally calls MCP `update_strategy_config` to apply changes via the server tool.
- Writes an audit log to `artifacts/optimization/apply-log.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.strategy_optimizer import STRATEGY_PARAMS, ParameterSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimizer_apply")


def load_result(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    if path.suffix in (".json", ".ndjson"):
        with open(path) as f:
            return json.load(f)
    else:
        # Try pickle
        try:
            import pickle

            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Unsupported result format or corrupted: {e}")


def validate_params(strategy: str, params: Dict[str, Any]) -> list[str]:
    """Return list of validation error messages (empty if ok)."""
    errors = []
    specs: list[ParameterSpec] = STRATEGY_PARAMS.get(strategy, [])

    # Build spec lookup by name
    spec_map: Dict[str, ParameterSpec] = {s.name: s for s in specs}

    for k, v in params.items():
        if k not in spec_map:
            logger.debug(f"Param {k} has no spec for strategy {strategy}; skipping bounds check")
            continue
        spec = spec_map[k]
        try:
            val = float(v)
        except Exception:
            errors.append(f"Param {k} value {v} is not numeric")
            continue

        if val < spec.min_val or val > spec.max_val:
            errors.append(f"Param {k}={val} out of bounds [{spec.min_val}, {spec.max_val}]")

    return errors


def load_strategy_config(strategy: str) -> Dict[str, Any]:
    path = Path("configs/strategies") / f"{strategy}.yaml"
    if not path.exists():
        logger.info(f"Strategy config for {strategy} not found; creating new one at {path}")
        # Basic template
        cfg = {
            "strategy": {"name": strategy, "version": "0.0.0", "type": "unknown", "description": "Auto-created by optimizer_apply"},
            "global_params": {},
            "risk_management": {},
            "symbols": {},
        }
        return cfg

    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_strategy_config(strategy: str, cfg: Dict[str, Any]) -> Path:
    path = Path("configs/strategies") / f"{strategy}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info(f"Saved strategy config to {path}")
    return path


async def call_mcp_update(strategy: str, key: str, value: Any, symbol: str | None = None) -> Dict[str, Any]:
    """Attempt to call the MCP tool `update_strategy_config` if available locally.

    This will only work when ordinis MCP server code is importable. We call the async tool
    directly and return the parsed JSON response.
    """
    try:
        from ordinis.mcp import server as mcp_server
    except Exception as e:
        logger.debug(f"MCP server module not importable: {e}")
        return {"error": "mcp-not-available"}

    # Ensure strategies are loaded in the server state
    try:
        # mcp_server._state._load_strategies is internal but safe to call
        mcp_server._state.strategies_config = mcp_server._state._load_strategies()
    except Exception:
        pass

    try:
        res_json = await mcp_server.update_strategy_config(strategy, key, value, symbol=symbol)
        try:
            return json.loads(res_json)
        except Exception:
            return {"raw": res_json}
    except Exception as e:
        logger.warning(f"MCP call failed for {strategy}/{key}: {e}")
        return {"error": str(e)}


def audit_log_entry(strategy: str, result_path: Path, changes: Dict[str, Any], user: str | None = None) -> None:
    out = Path("artifacts/optimization")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "apply-log.json"

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user": user or os.environ.get("USER", "unknown"),
        "strategy": strategy,
        "result_file": str(result_path),
        "changes": changes,
    }

    # Append to JSON-lines file
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote audit entry to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply optimized parameters safely to strategy configs")
    parser.add_argument("--result", type=str, required=True, help="Path to optimization result JSON or pickle")
    parser.add_argument("--yes", action="store_true", help="Auto-approve changes (non-interactive)")
    parser.add_argument("--mcp", action="store_true", help="Attempt to call MCP update_strategy_config for each param")
    parser.add_argument("--dry-run", action="store_true", help="Show changes but do not persist")
    args = parser.parse_args()

    result_path = Path(args.result)
    result = load_result(result_path)

    # Expect structure with best_params and strategy_name
    if "best_params" not in result or "strategy_name" not in result:
        raise RuntimeError("Result file missing best_params or strategy_name")

    strategy = result["strategy_name"]
    best_params: Dict[str, Any] = result["best_params"]

    logger.info(f"Loaded result for strategy {strategy}: {len(best_params)} params")

    # Validate
    errors = validate_params(strategy, best_params)
    if errors:
        logger.error("Validation errors:\n" + "\n".join(errors))
        raise RuntimeError("Parameter validation failed")

    # Load current config
    cfg = load_strategy_config(strategy)
    current_globals = cfg.get("global_params", {})

    # Build change set
    changes = {}
    for k, v in best_params.items():
        old = current_globals.get(k)
        changes[k] = {"old": old, "new": v}

    # Show summary
    print("\nProposed changes for strategy:", strategy)
    for k, ch in changes.items():
        print(f"  {k}: {ch['old']} -> {ch['new']}")

    # Edge markers and CI from result (if present)
    if "edge_zone_params" in result and result["edge_zone_params"]:
        print("\nEdge parameters detected:", ", ".join(result["edge_zone_params"]))
    if "bootstrap_cagr_ci" in result:
        print("Bootstrap CAGR CI:", result.get("bootstrap_cagr_ci"))

    if args.dry_run:
        print("\nDry-run mode: no changes applied.")
        return

    if not args.yes:
        ok = input("Apply changes? Type 'YES' to confirm: ")
        if ok.strip() != "YES":
            print("Aborted by user.")
            return

    # Apply to config in-memory
    cfg.setdefault("global_params", {})
    for k, ch in changes.items():
        cfg["global_params"][k] = ch["new"]

    # Persist
    cfg_path = save_strategy_config(strategy, cfg)

    # Optional MCP calls
    mcp_results = {}
    if args.mcp:
        print("Calling local MCP update_strategy_config for each param...")
        for k, ch in changes.items():
            try:
                res = asyncio.run(call_mcp_update(strategy, k, ch["new"]))
                mcp_results[k] = res
            except Exception as e:
                mcp_results[k] = {"error": str(e)}

        print("MCP results:")
        for k, r in mcp_results.items():
            print(f"  {k}: {r}")

    # Audit log
    audit_log_entry(strategy, result_path, changes)

    print(f"\nChanges applied and persisted to {cfg_path}")


if __name__ == "__main__":
    main()
