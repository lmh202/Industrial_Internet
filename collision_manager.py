"""Lightweight shuttle occupancy and reservation checks."""

from contextlib import contextmanager

from scene_config import (
    LINE_A_CENTER_X,
    LINE_B_CENTER_X,
    LINE_WORKSPACE_Y,
    SHUTTLE_SAFE_MARGIN,
    SHUTTLE_SIZE_X,
    SHUTTLE_SIZE_Y,
)


class CollisionManager:
    def __init__(self, sim, shuttle_handles: dict[str, int],
                 width: float = SHUTTLE_SIZE_X,
                 height: float = SHUTTLE_SIZE_Y,
                 margin: float = SHUTTLE_SAFE_MARGIN):
        self.sim = sim
        self.shuttle_handles = dict(shuttle_handles)
        self.width = width
        self.height = height
        self.margin = margin
        self._reserved: dict[str, tuple[str, tuple[float, float, float, float]]] = {}

    def checker_for(self, shuttle_name: str):
        return lambda x, y: self.is_safe(shuttle_name, x, y)

    def is_safe(self, shuttle_name: str, x: float, y: float) -> bool:
        if not self._inside_workspace(x, y):
            return False

        box = self._box(x, y)
        for name, handle in self.shuttle_handles.items():
            if name == shuttle_name:
                continue
            pos = self.sim.getObjectPosition(handle, -1)
            if self._overlaps(box, self._box(pos[0], pos[1])):
                return False

        for owner, reserved in self._reserved.values():
            if owner != shuttle_name and self._overlaps(box, reserved):
                return False
        return True

    @contextmanager
    def reserve(self, owner: str, key: str, x: float, y: float):
        token = f"{owner}:{key}"
        self._reserved[token] = (owner, self._box(x, y))
        try:
            yield
        finally:
            self._reserved.pop(token, None)

    def _inside_workspace(self, x: float, y: float) -> bool:
        min_x = LINE_A_CENTER_X - self.width
        max_x = LINE_B_CENTER_X + self.width
        min_y = -LINE_WORKSPACE_Y / 2 - self.margin
        max_y = LINE_WORKSPACE_Y / 2 + self.margin
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _box(self, x: float, y: float) -> tuple[float, float, float, float]:
        half_w = self.width / 2 + self.margin
        half_h = self.height / 2 + self.margin
        return (x - half_w, x + half_w, y - half_h, y + half_h)

    def _overlaps(self, a, b) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])
