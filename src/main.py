from __future__ import annotations

import argparse

from fastapi import FastAPI

from src.runtime import VisionRuntime
from src.utils.config_loader import ServiceConfig, load_config
from src.utils.logging_utils import configure_logging


config = load_config()
configure_logging(config.debug)
runtime = VisionRuntime(config)

app = FastAPI(title="ArmVisionService", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    if config.auto_start:
        runtime.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    runtime.stop()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", **runtime.state()}


@app.get("/last-move")
def last_move() -> dict[str, str | None]:
    return {"move": runtime.last_move}


@app.post("/start")
def start() -> dict[str, object]:
    started = runtime.start()
    return {"started": started, **runtime.state()}


@app.post("/stop")
def stop() -> dict[str, object]:
    stopped = runtime.stop()
    return {"stopped": stopped, **runtime.state()}


@app.post("/human-done")
def human_done() -> dict[str, object]:
    ok = runtime.on_human_move_complete()
    return {"accepted": ok, **runtime.state()}


@app.post("/robot-done")
def robot_done() -> dict[str, object]:
    ok = runtime.on_robot_done()
    return {"accepted": ok, **runtime.state()}



def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="ArmVisionService launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server to")
    parser.add_argument("--test", action="store_true", help="Run the terminal test loop with the visualizer")

    args = parser.parse_args()

    if args.test:
        from src.tools.test_runner import run_test_mode

        run_test_mode(config)
        return


    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
