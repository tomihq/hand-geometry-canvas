# Hand Interaction

Camera → hand tracking → 3D pose → abstract interaction events.

The library does **not** know about your app, objects, or rendering. It only emits:

- `HandPose` (3D palm + quaternion) via `on_pose`
- `HandEvent` (2D/abstract) via `on_event`:
  - **Pinch*** — thumb–index (create / stretch cursor); same model as `hand_canvas`
  - **Grab*** — closed fist (select + move); same model as `hand_canvas`
  - **HandMove** — always-on palm tracking for other apps

Sweep-to-clear stays in `hand_canvas` only.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` registers the `hand-gesture` CLI. Place or auto-download `models/hand_landmarker.task` (MediaPipe).

## Quick start (fake / tests)

```python
from hand_interaction import create_hand_gesture, FakeHandSource, PinchStart

fake = FakeHandSource()
g = create_hand_gesture(source=fake)
g.on_event(print)
g.start()
fake.emit(PinchStart("Right", position=__import__("hand_interaction").Vector2(0.5, 0.5)))
g.stop()
```

## Live camera API

```python
from hand_interaction import create_hand_gesture

g = create_hand_gesture(live=True)
g.on_event(lambda e: print(e.type, getattr(e, "hand_id", "")))
g.on_pose(lambda p: print(p.hand_id, p.palm))
g.start()
# ...
g.stop()
```

## Hand Geometry Canvas

Interactive canvas app (gestures from `hand_interaction` + figures / trash / sweep):

```bash
python -m hand_canvas.main
```

## Demo (OpenCV HUD, no canvas)

```bash
python -m demo.hand_pose_demo
```

Shows: skeleton, PINCH, FIST, POSITION, orientation (Euler for display only), CONFIDENCE.

## WebSocket server (independent process)

Publish `HandEvent` JSON to localhost clients:

```bash
hand-gesture serve
# live camera window + WebSocket:
hand-gesture serve --preview
# or: python -m hand_interaction serve --preview
```

Listens on `ws://127.0.0.1:8766` by default. Verify with:

```bash
python examples/websocket_client.py
```

## Tests

```bash
python -m pytest tests/ -q
```

## Coordinate notes

- Camera is mirrored.
- Event `Vector2` uses image space: origin at the **top** (`y` matches OpenCV / `hand_canvas`).
- `HandPose.palm.y` is still hybrid (origin at the **bottom**) for 3D consumers; `HandMove` converts to image `y`.
- `HandPose.palm.z` is relative depth (`z_landmark / hand_scale`), not metres.
