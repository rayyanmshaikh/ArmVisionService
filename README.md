# ArmVisionService

Python computer vision microservice for chess robot arm. Will be able to use a camera (laptop, external, etc...) to view a chessboard, recognize pieces and detect moves, outputting the recognized move on the board by a human.
Will communicate with ArmGameEngineService through POST for human moves.

## Quick start

- Install dependencies and run the API:

```bash
python -m pip install -r requirements.txt
python -m src.main
```

## Interactive API flow

1. `POST /start` — start camera and calibrate on an empty board.
2. Place pieces on the board.
3. `POST /capture-baseline` — capture the first board image with pieces.
4. Make your human move.
5. `POST /human-done` — server captures after image, detects the move, and (optionally) POSTs `{"move":"e2e4"}` to `MOVE_POST_URL`.
6. Robot moves.
7. `POST /robot-done` — server captures the board as the new baseline and returns to step 4.

## Key endpoints

- `GET /health` — runtime status and calibration state
- `GET /last-move` — last detected move
- `POST /start` — start vision loop
- `POST /capture-baseline` — capture baseline after placing pieces
- `POST /human-done` — tell the service the human finished a move
- `POST /robot-done` — tell the service the robot finished moving

## Test mode (terminal)

Run the simple terminal visualizer loop used for manual testing:

```bash
python -m src.main --test
```

## Configuration notes

- Environment and config live in `config/settings.yaml` and `ServiceConfig`.
- Important values:
  - `MOVE_POST_URL` — optional HTTP endpoint to receive `{"move":"e2e4"}` after a move is detected.
  - `CAMERA_INDEX`, `CAMERA_AUTO`, `BOARD_ORIENTATION`, and thresholds are configurable.

## Testing

- Unit and integration tests run with fixtures (no real camera required):

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## Docker

A minimal Docker setup is included. Build the image and run the service with camera passthrough where supported.

Build the image:

```bash
docker build -t arm-vision-service .
```

Run the container (Linux example with a local webcam):

```bash
docker run --rm -p 8000:8000 --device=/dev/video0:/dev/video0 arm-vision-service
```

The image runs `uvicorn src.main:app` by default. A `.dockerignore` file is included to keep the image small.

Using Docker Compose:

```bash
docker compose up --build
```

Notes:

- On Linux, map the camera device into the container with `--device` as shown above.
- On Windows, direct webcam passthrough varies by Docker/host setup; if passthrough is unavailable, run the service directly on the host (`python -m src.main`) or use WSL with access to the device.
