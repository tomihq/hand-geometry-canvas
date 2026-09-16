"""GPU canvas compositor via OpenGL (moderngl).

Uploads the camera frame as a texture and draws shapes as GPU quads.
Falls back is handled by the caller if context creation fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hand_canvas.geometry import PointShape, Rectangle, rectangle_corners

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
    """Composites camera background + shapes on the GPU."""

    def __init__(self, width: int, height: int) -> None:
        import glfw
        import moderngl

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(max(width, 64), max(height, 64), "gpu", None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("Could not create hidden OpenGL window")
        glfw.make_context_current(window)

        self._glfw = glfw
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

        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((width, height), 3)]
        )
        self._frame_tex = self._ctx.texture((width, height), 3)
        self._frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    @classmethod
    def try_create(cls, width: int, height: int) -> GpuCanvasRenderer | None:
        try:
            renderer = cls(width, height)
            print(f"Canvas GPU renderer ready (OpenGL {renderer._ctx.version_code})")
            return renderer
        except Exception as exc:  # noqa: BLE001
            print(f"Canvas GPU unavailable, using CPU: {exc}")
            return None

    def resize(self, width: int, height: int) -> None:
        if width == self._width and height == self._height:
            return
        self._width = width
        self._height = height
        self._fbo.release()
        self._frame_tex.release()
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((width, height), 3)]
        )
        self._frame_tex = self._ctx.texture((width, height), 3)

    def render(
        self,
        canvas: Canvas,
        background: np.ndarray,
        selected_ids: set[str] | None = None,
    ) -> np.ndarray:
        selected_ids = selected_ids or set()
        h, w = background.shape[:2]
        self.resize(w, h)

        # BGR uint8 → RGB for GL texture
        rgb = np.ascontiguousarray(background[:, :, ::-1])
        self._frame_tex.write(rgb.tobytes())

        self._fbo.use()
        self._ctx.viewport = (0, 0, w, h)
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
            nbytes = data.nbytes
            if nbytes > self._color_vbo.size:
                self._color_vbo.orphan(nbytes)
            self._color_vbo.write(data.tobytes())
            self._color_vao.render(vertices=len(data) // 6)

        # Read back BGR for OpenCV display / debug overlay
        raw = self._fbo.read(components=3, alignment=1)
        rgb_out = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        # OpenGL origin is bottom-left
        rgb_out = np.flipud(rgb_out)
        return np.ascontiguousarray(rgb_out[:, :, ::-1])

    def close(self) -> None:
        try:
            self._fs_vao.release()
            self._color_vao.release()
            self._fs_vbo.release()
            self._color_vbo.release()
            self._frame_tex.release()
            self._fbo.release()
            self._tex_prog.release()
            self._color_prog.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._glfw.destroy_window(self._window)
            self._glfw.terminate()
        except Exception:  # noqa: BLE001
            pass
