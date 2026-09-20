# Hand Interaction

Camera → hand tracking → abstract interaction events.

The library does **not** know about your app, objects, or rendering. It emits `HandEvent` (2D / abstract) via `on_event`:

- **Pinch*** — thumb–index (create / stretch cursor); same model as `hand_canvas`
- **Grab*** — closed fist (select + move); same model as `hand_canvas`
- **Resize*** — second hand joins while owner holds (helper cursor for corner-follow)
- **HandMove** — always-on palm tracking for other apps

Optional `on_pose` (`HandPose`: palm + orientation) is for debugging / raw pose. The canvas and gesture events stay 2D — no 3D scene.

Sweep-to-clear stays in `hand_canvas` only.

## 1. Install

Needs Python ≥ 3.11 and a webcam.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

That installs deps (OpenCV, MediaPipe, ModernGL for the canvas, WebSockets, pytest) and registers the `hand-gesture` CLI.

First camera run downloads MediaPipe’s `hand_landmarker.task` into `models/` if it is missing.

## 2. Use the canvas

Interactive 2D figures (create / move / resize / trash / sweep):

```bash
python -m hand_canvas.main
```

## 3. Hand demo with HUD stats

Camera + skeleton + live stats (no canvas):

```bash
python -m demo.hand_pose_demo
```

HUD shows: skeleton, PINCH, FIST, POSITION, orientation (Euler), CONFIDENCE.

## 4. Emit events only (WebSocket, no camera window)

Publish `HandEvent` JSON on `ws://127.0.0.1:8766` (headless — tracking runs, no preview window):

```bash
hand-gesture serve
```

In another terminal, print events:

```bash
python examples/websocket_client.py
```

## 5. Emit events + see the camera

Same as above, with a live OpenCV preview (cursor / gesture overlay; press `q` to quit):

```bash
hand-gesture serve --preview
```

Then, in another terminal:

```bash
python examples/websocket_client.py
```

Equivalent module form: `python -m hand_interaction serve` / `serve --preview`.

## Programmatic API (optional)

Fake source (no camera):

```python
from hand_interaction import create_hand_gesture, FakeHandSource, PinchStart, Vector2

fake = FakeHandSource()
g = create_hand_gesture(source=fake)
g.on_event(print)
g.start()
fake.emit(PinchStart("Right", position=Vector2(0.5, 0.5)))
g.stop()
```

Live camera in-process:

```python
from hand_interaction import create_hand_gesture

g = create_hand_gesture(live=True)
g.on_event(lambda e: print(e.type, getattr(e, "hand_id", "")))
g.start()
# ...
g.stop()
```

## Tests

```bash
python -m pytest tests/ -q
```

## Coordinate notes

- Camera is mirrored (selfie view).
- `hand_id` / handedness are **anatomical**: `Right` is the user's right hand
  even though the frame is mirrored (MediaPipe's raw labels are corrected).
- Event `Vector2` uses image space: origin at the **top** (`y` matches OpenCV / `hand_canvas`).
- `HandPose.palm.y` (if you use `on_pose`) uses origin at the **bottom**; `HandMove` converts to image `y`.
- `HandPose.palm.z` is relative depth (`z_landmark / hand_scale`), not metres.
