import pathlib

import yaml

from ordinis.engines.learning import LearningEngine  # existing class
from ordinis.services.helix import Helix  # LLM wrapper


def load_dev_cfg() -> dict:
    cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "dev.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_dev_cfg()
    if not cfg.get("learning", {}).get("enabled", False):
        print("[dev] LearningEngine disabled – set learning.enabled: true")
        return

    # Initialise Helix with the dev profile (Nemotron‑8B)
    helix = Helix(profile=cfg["learning"]["llm_profile"])

    # Initialise the LearningEngine – we pass the config dict so it can read
    # data_root, model_dir, epochs, etc.
    learner = LearningEngine(
        data_root=cfg["learning"]["data_root"],
        model_dir=cfg["learning"]["model_dir"],
        epochs=cfg["learning"]["epochs"],
        use_gpu=cfg["learning"]["use_gpu"],
        helix=helix,
        safety_filter=cfg["learning"]["safety_filter"],
        log_level=cfg["learning"]["log_level"],
    )

    # Record events from the running system (mocked here)
    learner.record_event({"type": "signal", "payload": {"symbol": "AAPL", "value": 0.12}})
    learner.record_event({"type": "execution", "payload": {"order_id": 42, "filled": True}})

    # Train / fine‑tune models
    learner.train(models=["signal_core", "cortex_prompt"])

    # Evaluate & optionally promote
    learner.evaluate(new_model="signal_core_v2", benchmark="dev_suite")
    print("[dev] Learning run complete – artefacts in", learner.model_dir)


if __name__ == "__main__":
    main()
