"""
Shared utilities for viewers: schema validation, visibility precompute,
and config parity with llm_bot_controller.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Union

from battle_types import BattleData, BattleFrame, BattleMetadata


def validate_battle_data(
    battle_data: Union[BattleData, Dict[str, Any]]
) -> BattleData:
    if isinstance(battle_data, BattleData):
        return battle_data
    return BattleData.model_validate(battle_data)


def get_visibility_config(controller: Any) -> Dict[str, Any]:
    # Mirror defaults used in controller and viewers
    sense_range = 15.0
    fov_deg = 120
    ray_count = int(getattr(controller, "visibility_ray_count", 24))
    return {"sense_range": sense_range, "fov_deg": fov_deg, "ray_count": ray_count}


def _build_mock_arena(
    current_state: Union[BattleFrame, Dict[str, Any]],
    metadata: Union[BattleMetadata, Dict[str, Any]],
):
    """Minimal arena+space mock compatible with PythonLLMController.generate_visible_objects()."""

    if isinstance(current_state, BattleFrame):
        current_state = current_state.model_dump(mode="json")
    if isinstance(metadata, BattleMetadata):
        metadata = metadata.model_dump(mode="json")

    class MockArena:
        COLLISION_TYPE_BOT = 1
        COLLISION_TYPE_PROJECTILE = 2
        COLLISION_TYPE_WALL = 3

        class _Vec2:
            __slots__ = ("x", "y")
            def __init__(self, x: float, y: float):
                self.x = float(x)
                self.y = float(y)
            def __iter__(self):
                yield self.x
                yield self.y
            def __add__(self, other):
                ox, oy = (other.x, other.y) if hasattr(other, "x") else other
                return MockArena._Vec2(self.x + ox, self.y + oy)
            def __sub__(self, other):
                ox, oy = (other.x, other.y) if hasattr(other, "x") else other
                return MockArena._Vec2(self.x - ox, self.y - oy)
            def __mul__(self, scalar: float):
                return MockArena._Vec2(self.x * scalar, self.y * scalar)
            __rmul__ = __mul__
            def dot(self, other) -> float:
                ox, oy = (other.x, other.y) if hasattr(other, "x") else other
                return self.x * ox + self.y * oy
            def cross(self, other) -> float:
                ox, oy = (other.x, other.y) if hasattr(other, "x") else other
                return self.x * oy - self.y * ox
            @property
            def length(self) -> float:
                return math.hypot(self.x, self.y)

        class _SegmentQueryInfo:
            __slots__ = ("shape", "alpha", "point")
            def __init__(self, shape, alpha: float, point):
                self.shape = shape
                self.alpha = alpha
                self.point = point

        class _MockBody:
            __slots__ = ("position", "angle", "velocity", "shapes")
            def __init__(self, position, angle: float, velocity):
                self.position = position
                self.angle = angle
                self.velocity = velocity
                self.shapes = ()

        class _MockCircleShape:
            __slots__ = ("body", "radius", "collision_type")
            def __init__(self, body, radius: float, collision_type: int):
                self.body = body
                self.radius = radius
                self.collision_type = collision_type
            def segment_intersections(self, start, end):
                center = MockArena._Vec2(*self.body.position)
                direction = end - start
                a = direction.dot(direction)
                if a <= 1e-9:
                    return []
                to_center = start - center
                b = 2.0 * to_center.dot(direction)
                c = to_center.dot(to_center) - self.radius**2
                disc = b * b - 4.0 * a * c
                if disc < 0.0:
                    return []
                sqrt_disc = math.sqrt(max(disc, 0.0))
                intersections = []
                for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
                    if 0.0 <= t <= 1.0:
                        point = start + direction * t
                        intersections.append((t, point))
                return intersections

        class _MockPolygonShape:
            __slots__ = ("body", "_vertices", "collision_type")
            def __init__(self, vertices, collision_type: int):
                self.body = None
                self._vertices = [MockArena._Vec2(*v) for v in vertices]
                self.collision_type = collision_type
            def get_vertices(self):
                return [(v.x, v.y) for v in self._vertices]
            def segment_intersections(self, start, end):
                results = []
                if len(self._vertices) < 2:
                    return results
                ray = end - start
                for idx in range(len(self._vertices)):
                    a = self._vertices[idx]
                    b = self._vertices[(idx + 1) % len(self._vertices)]
                    edge = b - a
                    denom = ray.cross(edge)
                    if abs(denom) < 1e-9:
                        continue
                    diff = a - start
                    t = diff.cross(edge) / denom
                    u = diff.cross(ray) / denom
                    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                        point = start + ray * t
                        results.append((t, point))
                return results

        class _MockSpace:
            def __init__(self):
                self._shapes = []
            def add_shape(self, shape):
                self._shapes.append(shape)
            def segment_query(self, start, end, _radius, _shape_filter):
                start_vec = MockArena._Vec2(*start)
                end_vec = MockArena._Vec2(*end)
                hits = []
                for shape in self._shapes:
                    if not hasattr(shape, "segment_intersections"):
                        continue
                    for alpha, point in shape.segment_intersections(start_vec, end_vec):
                        hits.append(MockArena._SegmentQueryInfo(shape, alpha, point))
                hits.sort(key=lambda info: info.alpha)
                return hits

        def __init__(self, state: Dict[str, Any], metadata: Dict[str, Any]):
            self.SENSE_RANGE = 15.0
            self.FOV_ANGLE = math.radians(120)
            self.BOT_RADIUS = 0.5
            self.bot_data = {}
            self.bot_bodies = {}
            self.projectile_data = {}
            self.projectile_bodies = {}
            self.wall_bodies = []
            self.space = MockArena._MockSpace()

            for bot in state.get("bots", []):
                if not bot.get("alive", True):
                    continue
                bot_id = bot["id"]

                class MockBotData:
                    def __init__(self, bot_info):
                        self.team = bot_info["team"]
                        self.hp = bot_info["hp"]
                        self.signal = bot_info.get("signal", "none")

                body = MockArena._MockBody(
                    (bot["x"], bot["y"]),
                    math.radians(bot["theta"]),
                    (bot.get("vx", 0), bot.get("vy", 0)),
                )
                shape = MockArena._MockCircleShape(body, self.BOT_RADIUS, MockArena.COLLISION_TYPE_BOT)
                body.shapes = (shape,)
                self.bot_data[bot_id] = MockBotData(bot)
                self.bot_bodies[bot_id] = body
                self.space.add_shape(shape)

            for proj in state.get("projectiles", []):
                proj_id = len(self.projectile_data)

                class MockProjData:
                    def __init__(self, proj_info):
                        self.team = proj_info.get("team", 0)
                        self.ttl = proj_info.get("ttl", 1.0)

                body = MockArena._MockBody(
                    (proj["x"], proj["y"]),
                    0.0,
                    (proj.get("vx", 0), proj.get("vy", 0)),
                )
                shape = MockArena._MockCircleShape(body, 0.1, MockArena.COLLISION_TYPE_PROJECTILE)
                body.shapes = (shape,)
                self.projectile_data[proj_id] = MockProjData(proj)
                self.projectile_bodies[proj_id] = body
                self.space.add_shape(shape)

            for wall_def in metadata.get("walls", []):
                if hasattr(wall_def, "center_x"):
                    cx = float(wall_def.get("center_x"))
                    cy = float(wall_def.get("center_y"))
                    w = float(wall_def.get("width"))
                    h = float(wall_def.get("height"))
                    angle_deg = float(wall_def.get("angle_deg"))
                else:
                    cx, cy, w, h, angle_deg = wall_def
                angle_rad = math.radians(angle_deg)
                c, s = math.cos(angle_rad), math.sin(angle_rad)
                hw, hh = w / 2.0, h / 2.0
                local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                rotated = [(px * c - py * s, px * s + py * c) for px, py in local_corners]
                world_corners = [(px + cx, py + cy) for px, py in rotated]
                wall_shape = MockArena._MockPolygonShape(world_corners, MockArena.COLLISION_TYPE_WALL)
                self.wall_bodies.append((None, wall_shape))
                self.space.add_shape(wall_shape)

        def _is_bot_alive(self, bot_id):
            return bot_id in self.bot_data

    return MockArena(current_state, metadata)


def precompute_visibility(
    battle_data: Union[BattleData, Dict[str, Any]], controller: Any
) -> BattleData:
    """Augment battle_data.timeline frames with precomputed visibility per bot.

    Adds a key per frame: precomp_visible_by_bot: {str(bot_id): [visible_obj, ...]}
    """
    validated = validate_battle_data(battle_data)
    timeline: List[BattleFrame] = list(validated.timeline)
    metadata = validated.metadata

    for frame in timeline:
        try:
            mock_arena = _build_mock_arena(frame, metadata)
        except Exception:
            # If we can't build the mock arena for this frame, skip augmentation
            continue

        vis_by_bot: Dict[str, List[Dict[str, Any]]] = {}

        for bot in frame.get("bots", []):
            if not bot.get("alive", True):
                continue
            bot_id = bot.get("id")
            if bot_id is None:
                continue
            try:
                visible = controller.generate_visible_objects(mock_arena, int(bot_id))
            except Exception:
                visible = []
            vis_by_bot[str(bot_id)] = visible

        if vis_by_bot:
            frame.precomp_visible_by_bot = vis_by_bot

    return validated
