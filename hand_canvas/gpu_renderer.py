"""GPU canvas compositor via OpenGL (moderngl).

Owns the visible window: the camera frame is uploaded as a texture and shapes
are drawn as GPU quads straight into the default framebuffer. Nothing is read
back to the CPU, which is what a ``cv2.imshow`` display path would have forced.
Falls back is handled by the caller if context creation fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from hand_canvas.geometry import (
    PointShape,
    Rectangle,
    is_visible_figure,
    rectangle_corners,
)

if TYPE_CHECKING:
    from hand_canvas.canvas import Canvas

_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG_TEX = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""

_VERT_COLOR = """
#version 330
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    v_color = in_color;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG_COLOR = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}
"""


def _to_ndc(x: float, y: float) -> tuple[float, float]:
    """Normalized [0,1] canvas coords → OpenGL NDC (Y flipped)."""
    return 2.0 * x - 1.0, 1.0 - 2.0 * y


def _bgr_to_rgba01(bgr: tuple[int, int, int], alpha: float) -> tuple[float, float, float, float]:
    b, g, r = bgr
    return r / 255.0, g / 255.0, b / 255.0, alpha


def _quad_ndc(
    x0: float, y0: float, x1: float, y1: float, rgba: tuple[float, float, float, float]
) -> list[float]:
    """Two triangles (6 verts) as interleaved pos.xy + color.rgba."""
    p00 = _to_ndc(x0, y0)
    p10 = _to_ndc(x1, y0)
    p01 = _to_ndc(x0, y1)
    p11 = _to_ndc(x1, y1)
    r, g, b, a = rgba
    verts = [
        p00[0], p00[1], r, g, b, a,
        p10[0], p10[1], r, g, b, a,
        p11[0], p11[1], r, g, b, a,
        p00[0], p00[1], r, g, b, a,
        p11[0], p11[1], r, g, b, a,
        p01[0], p01[1], r, g, b, a,
    ]
    return verts


class GpuCanvasRenderer:
    """Composites camera background + shapes on the GPU and owns the window."""

    def __init__(self, width: int, height: int, title: str = "Hand Geometry Canvas") -> None:
        import glfw
        import moderngl

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(max(width, 64), max(height, 64), title, None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("Could not create OpenGL window")
        glfw.make_context_current(window)
        # No vsync: capping to the monitor refresh would turn a 55fps pipeline
        # into a 30fps one every time a frame misses its slot.
        glfw.swap_interval(0)

        self._pressed: list[int] = []
        glfw.set_key_callback(window, self._on_key)

        self._glfw = glfw
        self._moderngl = moderngl
        self._window = window
        self._ctx = moderngl.create_context()
        self._width = width
        self._height = height

        self._tex_prog = self._ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG_TEX)
        self._color_prog = self._ctx.program(
            vertex_shader=_VERT_COLOR, fragment_shader=_FRAG_COLOR
        )

        # Fullscreen quad (NDC) with UVs matching OpenCV top-left origin
        fs = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 0.0,
                -1.0,  1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 1.0,
                 1.0,  1.0, 1.0, 0.0,
            ],
            dtype="f4",
        )
        self._fs_vbo = self._ctx.buffer(fs.tobytes())
        self._fs_vao = self._ctx.vertex_array(
            self._tex_prog, [(self._fs_vbo, "2f 2f", "in_pos", "in_uv")]
        )

        self._color_vbo = self._ctx.buffer(reserve=1024 * 1024)
        self._color_vao = self._ctx.vertex_array(
            self._color_prog, [(self._color_vbo, "2f 4f", "in_pos", "in_color")]
        )

        self._frame_tex = self._ctx.texture((width, height), 3)
        self._frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._overlay_tex = None
        self._overlay_version: int | None = None

    @classmethod
    def try_create(
        cls, width: int, height: int, title: str = "Hand Geometry Canvas"
    ) -> GpuCanvasRenderer | None:
        try:
            renderer = cls(width, height, title)
            print(f"Canvas GPU renderer ready (OpenGL {renderer._ctx.version_code})")
            return renderer
        except Exception as exc:  # noqa: BLE001
            print(f"Canvas GPU unavailable, using CPU: {exc}")
            return None

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        if action == self._glfw.PRESS:
            self._pressed.append(key)

    @property
    def should_close(self) -> bool:
        return bool(self._glfw.window_should_close(self._window))

    def poll_keys(self) -> list[str]:
        """Printable keys pressed since the last call, lowercased."""
        self._glfw.poll_events()
        keys = [chr(k).lower() for k in self._pressed if 32 <= k <= 126]
        self._pressed.clear()
        return keys

    def set_title(self, title: str) -> None:
        self._glfw.set_window_title(self._window, title)

    def _ensure_frame_tex(self, width: int, height: int) -> None:
        if width == self._width and height == self._height:
            return
        self._width = width
        self._height = height
        self._frame_tex.release()
        self._frame_tex = self._ctx.texture((width, height), 3)
        self._frame_tex.filter = (self._moderngl.LINEAR, self._moderngl.LINEAR)

    def _ensure_overlay_tex(self, width: int, height: int):
        tex = self._overlay_tex
        if tex is not None and tex.size == (width, height):
            return tex
        if tex is not None:
            tex.release()
        tex = self._ctx.texture((width, height), 4)
        tex.filter = (self._moderngl.LINEAR, self._moderngl.LINEAR)
        self._overlay_tex = tex
        self._overlay_version = None
        return tex

    def present(
        self,
        canvas: Canvas,
        background: np.ndarray,
        selected_ids: set[str] | None = None,
        overlay: np.ndarray | None = None,
        overlay_version: int = 0,
    ) -> None:
        """Draw one frame into the window. Nothing comes back to the CPU."""
        selected_ids = selected_ids or set()
        h, w = background.shape[:2]
        self._ensure_frame_tex(w, h)

        rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        self._frame_tex.write(rgb)

        fb_w, fb_h = self._glfw.get_framebuffer_size(self._window)
        self._ctx.screen.use()
        self._ctx.viewport = (0, 0, max(fb_w, 1), max(fb_h, 1))
        self._ctx.clear(0.0, 0.0, 0.0)
        self._ctx.disable(self._ctx.BLEND)

        self._frame_tex.use(0)
        self._tex_prog["u_tex"] = 0
        self._fs_vao.render()

        self._ctx.enable(self._ctx.BLEND)
        self._ctx.blend_func = self._ctx.SRC_ALPHA, self._ctx.ONE_MINUS_SRC_ALPHA

        verts: list[float] = []
        handle = 6.0 / w
        handle_y = 6.0 / h

        for shape in sorted(canvas.shapes, key=lambda s: s.z):
            # Only real figures get painted; a point or sliver is a live preview
            # of a gesture in progress, so it shows only while a hand holds it.
            if not is_visible_figure(shape) and shape.id not in selected_ids:
                continue

            if isinstance(shape, PointShape):
                cx, cy = shape.position.x, shape.position.y
                r = 8.0 / w
                ry = 8.0 / h
                # Outer ring then fill
                verts.extend(
                    _quad_ndc(
                        cx - r * 1.25,
                        cy - ry * 1.25,
                        cx + r * 1.25,
                        cy + ry * 1.25,
                        _bgr_to_rgba01((255, 255, 255), 0.9),
                    )
                )
                verts.extend(
                    _quad_ndc(cx - r, cy - ry, cx + r, cy + ry, _bgr_to_rgba01((0, 220, 255), 1.0))
                )
            elif isinstance(shape, Rectangle):
                color = (60, 180, 255) if shape.id in selected_ids else (80, 200, 120)
                x0, y0 = shape.x, shape.y
                x1, y1 = shape.x + shape.width, shape.y + shape.height
                verts.extend(_quad_ndc(x0, y0, x1, y1, _bgr_to_rgba01(color, 0.35)))

                # Border as thin quads
                t = 2.0 / w
                ty = 2.0 / h
                border = _bgr_to_rgba01(color, 1.0)
                verts.extend(_quad_ndc(x0, y0, x1, y0 + ty, border))
                verts.extend(_quad_ndc(x0, y1 - ty, x1, y1, border))
                verts.extend(_quad_ndc(x0, y0, x0 + t, y1, border))
                verts.extend(_quad_ndc(x1 - t, y0, x1, y1, border))

                handle_rgba = _bgr_to_rgba01(
                    (0, 255, 255) if shape.id in selected_ids else (220, 220, 220), 1.0
                )
                for corner_pt in rectangle_corners(shape).values():
                    hx, hy = corner_pt.x, corner_pt.y
                    verts.extend(
                        _quad_ndc(
                            hx - handle,
                            hy - handle_y,
                            hx + handle,
                            hy + handle_y,
                            handle_rgba,
                        )
                    )

        if verts:
            data = np.asarray(verts, dtype="f4")
            if data.nbytes > self._color_vbo.size:
                self._color_vbo.orphan(data.nbytes)
            self._color_vbo.write(data)
            self._color_vao.render(vertices=len(data) // 6)

        if overlay is not None:
            oh, ow = overlay.shape[:2]
            tex = self._ensure_overlay_tex(ow, oh)
            if overlay_version != self._overlay_version:
                tex.write(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA))
                self._overlay_version = overlay_version
            # cv2 antialiases into a transparent layer, which yields premultiplied
            # alpha, so the source factor is ONE rather than SRC_ALPHA.
            self._ctx.blend_func = self._ctx.ONE, self._ctx.ONE_MINUS_SRC_ALPHA
            tex.use(0)
            self._tex_prog["u_tex"] = 0
            self._fs_vao.render()

        self._glfw.swap_buffers(self._window)

    def close(self) -> None:
        try:
            self._fs_vao.release()
            self._color_vao.release()
            self._fs_vbo.release()
            self._color_vbo.release()
            self._frame_tex.release()
            if self._overlay_tex is not None:
                self._overlay_tex.release()
            self._tex_prog.release()
            self._color_prog.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._glfw.destroy_window(self._window)
            self._glfw.terminate()
        except Exception:  # noqa: BLE001
            pass
