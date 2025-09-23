"""
Python Function LLM Controller
Single hub for:
- Building concise prompts from bot observations
- Polling an OpenAI-compatible LLM endpoint
- Falling back to local example templates for testing and endpointless development
"""

import math
import random
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Tuple

import json
from urllib import error, request

import pymunk

import example_llm_gen_control_code


class PythonLLMController:
    """Generates Python functions for bot control and manages bot observations."""

    MOVE_HISTORY_LIMIT = 12

    def __init__(self, bot_count: int = 4, default_templates: List[str] | None = None):
        self.bot_count = bot_count
        self.bot_function_sources = {}
        self.function_templates = list(default_templates) if default_templates else []
        self.move_history: Dict[int, Deque[Dict[str, Any]]] = {
            bot_id: deque(maxlen=self.MOVE_HISTORY_LIMIT) for bot_id in range(bot_count)
        }
        self.visibility_ray_count = 24
        self._wall_segment_cache: Dict[pymunk.Shape, Tuple[Tuple[float, float], ...]] = {}
        self._regen_counter: int = 0
        self._mock_templates: List[str] | None = None
        self._mock_rng = random.Random()
        # Cache for visibility ray direction deltas (cos/sin tables)
        self._ray_key: Tuple[int, float] | None = None
        self._ray_cos: List[float] | None = None
        self._ray_sin: List[float] | None = None

        # Assign different function types to bots
        for bot_id in range(bot_count):
            if self.function_templates:
                template_idx = bot_id % len(self.function_templates)
                self.bot_function_sources[bot_id] = self.function_templates[template_idx]
            else:
                self.bot_function_sources[bot_id] = self._pick_mock_template(bot_id)
            
    # No internal templates; tests or callers can inject via constructor

    def _ensure_wall_cache(self, arena) -> Dict[pymunk.Shape, Tuple[Tuple[float, float], ...]]:
        if self._wall_segment_cache:
            return self._wall_segment_cache

        cache: Dict[pymunk.Shape, Tuple[Tuple[float, float], ...]] = {}
        for wall_body, wall_shape in getattr(arena, "wall_bodies", []):
            if not wall_shape or not hasattr(wall_shape, "get_vertices"):
                continue

            verts = wall_shape.get_vertices()
            if wall_body is not None:
                world_vertices = tuple(
                    (float(world_v.x), float(world_v.y))
                    for world_v in (wall_body.local_to_world(v) for v in verts)
                )
            else:
                world_vertices = tuple((float(v[0]), float(v[1])) for v in verts)

            cache[wall_shape] = world_vertices

        self._wall_segment_cache = cache
        return cache

    def _segment_query_first_excluding(
        self,
        space: pymunk.Space,
        start: Tuple[float, float],
        end: Tuple[float, float],
        skip_shapes: Tuple[pymunk.Shape, ...],
    ):
        hits = space.segment_query(start, end, 0.0, pymunk.ShapeFilter())
        if not hits:
            return None

        best_hit = None
        best_alpha = float("inf")
        skip = skip_shapes  # expected to be a set-like; caller passes frozenset

        for info in hits:
            if info.shape in skip:
                continue
            if info.alpha < best_alpha:
                best_alpha = info.alpha
                best_hit = info

        return best_hit

    def get_bot_function_source(self, bot_id: int) -> str:
        """Get Python function source code for a bot."""
        if bot_id in self.bot_function_sources:
            return self.bot_function_sources[bot_id]
        # Fallback to a balanced-like default from templates
        if self.function_templates:
            idx = 2 if len(self.function_templates) >= 3 else 0
            return self.function_templates[idx]
        return self._pick_mock_template(bot_id)

    def _is_in_fov(
        self,
        bot_heading: float,
        bot_x: float,
        bot_y: float,
        target_x: float,
        target_y: float,
        fov_angle: float = math.radians(120),
    ) -> bool:
        """Check if target is within bot's field of view."""
        # Calculate angle from bot to target
        target_angle = math.atan2(target_y - bot_y, target_x - bot_x)

        # Calculate relative angle from bot's heading
        angle_diff = target_angle - bot_heading

        # Normalize angle difference to [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Check if within FOV (half angle on each side)
        return abs(angle_diff) <= fov_angle / 2

    def _ray_intersects_segment(self, ray_start, ray_end, seg_start, seg_end):
        """Check if a ray intersects with a line segment. Returns (intersects, distance, point)."""
        # Convert to numpy arrays for easier math
        p1, p2 = ray_start, ray_end
        p3, p4 = seg_start, seg_end

        # Calculate intersection using line-line intersection formula
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return False, float("inf"), None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection point is on both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point and distance
            intersect_x = x1 + t * (x2 - x1)
            intersect_y = y1 + t * (y2 - y1)
            distance = math.sqrt((intersect_x - x1) ** 2 + (intersect_y - y1) ** 2)
            return True, distance, (intersect_x, intersect_y)

        return False, float("inf"), None

    def _ray_intersects_circle(self, ray_start, ray_end, circle_center, circle_radius):
        """Check if a ray intersects with a circle. Returns (intersects, distance, point)."""
        # Vector from ray start to circle center
        cx, cy = circle_center
        rx, ry = ray_start
        dx, dy = ray_end[0] - ray_start[0], ray_end[1] - ray_start[1]

        # Solve quadratic equation for ray-circle intersection
        a = dx * dx + dy * dy
        b = 2 * (dx * (rx - cx) + dy * (ry - cy))
        c = (rx - cx) ** 2 + (ry - cy) ** 2 - circle_radius**2

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False, float("inf"), None

        # Find closest intersection point (smallest t > 0)
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        # Use the closest intersection that's on the ray (0 <= t <= 1)
        t = None
        if 0 <= t1 <= 1:
            t = t1
        elif 0 <= t2 <= 1:
            t = t2

        if t is not None:
            intersect_x = rx + t * dx
            intersect_y = ry + t * dy
            distance = math.sqrt((intersect_x - rx) ** 2 + (intersect_y - ry) ** 2)
            return True, distance, (intersect_x, intersect_y)

        return False, float("inf"), None

    def generate_visible_objects(self, arena, bot_id: int) -> List[Dict]:
        """Gather objects within the bot's FOV using pymunk queries."""
        if not arena._is_bot_alive(bot_id):
            return []

        space = arena.space
        bot_body = arena.bot_bodies[bot_id]
        bot_data = arena.bot_data[bot_id]
        bot_pos = bot_body.position
        bot_pos_x = float(bot_pos.x)
        bot_pos_y = float(bot_pos.y)
        bot_heading = bot_body.angle
        sense_range = arena.SENSE_RANGE
        sense_range_sq = sense_range * sense_range
        fov_angle = arena.FOV_ANGLE
        half_fov = 0.5 * fov_angle
        skip_shapes = getattr(bot_body, "_llm_skip_shapes", None)
        if skip_shapes is None:
            skip_shapes = frozenset(bot_body.shapes)
            setattr(bot_body, "_llm_skip_shapes", skip_shapes)

        wall_cache = self._ensure_wall_cache(arena)

        entity_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
        wall_hits: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Use a consistent ray count regardless of bot count
        ray_count = int(self.visibility_ray_count)
        # Precompute direction table for current (ray_count, fov)
        ray_key = (ray_count, fov_angle)
        if self._ray_key != ray_key:
            n = max(1, ray_count)
            if n > 1:
                step = fov_angle / (n - 1)
            else:
                step = 0.0
            start = -half_fov
            cos_tab = []
            sin_tab = []
            for i in range(n):
                a = start + step * i
                cos_tab.append(math.cos(a))
                sin_tab.append(math.sin(a))
            self._ray_key = ray_key
            self._ray_cos = cos_tab
            self._ray_sin = sin_tab
        cos_tab = self._ray_cos or [1.0]
        sin_tab = self._ray_sin or [0.0]

        cosH = math.cos(bot_heading)
        sinH = math.sin(bot_heading)

        for ray_index in range(ray_count):
            cd = cos_tab[ray_index]
            sd = sin_tab[ray_index]
            cos_a = cosH * cd - sinH * sd
            sin_a = sinH * cd + cosH * sd
            end_x = bot_pos_x + cos_a * sense_range
            end_y = bot_pos_y + sin_a * sense_range
            hit = self._segment_query_first_excluding(
                space,
                (bot_pos_x, bot_pos_y),
                (end_x, end_y),
                skip_shapes,
            )
            if not hit:
                continue

            shape = hit.shape
            if shape.collision_type != arena.COLLISION_TYPE_WALL:
                continue

            point = hit.point
            dx = float(point.x) - bot_pos_x
            dy = float(point.y) - bot_pos_y
            distance_sq = dx * dx + dy * dy
            if distance_sq > sense_range_sq:
                continue
            distance = math.sqrt(distance_sq)
            if distance > sense_range:
                continue

            bearing = (
                math.degrees(math.atan2(dy, dx))
                + 360.0
            ) % 360.0

            cache_key = (id(shape), len(wall_cache.get(shape, ())))
            entry = {
                "type": "wall",
                "x": round(point.x, 2),
                "y": round(point.y, 2),
                "distance": distance,
                "angle": bearing,
            }

            current = wall_hits.get(cache_key)
            if not current or distance < current["distance"]:
                wall_hits[cache_key] = entry

        for other_id, other_data in arena.bot_data.items():
            if other_id == bot_id or not arena._is_bot_alive(other_id):
                continue

            other_body = arena.bot_bodies[other_id]
            other_pos = other_body.position
            dx = float(other_pos.x) - bot_pos_x
            dy = float(other_pos.y) - bot_pos_y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= 0.0 or dist_sq > sense_range_sq:
                continue

            bearing_rad = math.atan2(dy, dx)
            angle_diff = (bearing_rad - bot_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > half_fov:
                continue

            distance = math.sqrt(dist_sq)

            hit = self._segment_query_first_excluding(
                space,
                (bot_pos_x, bot_pos_y),
                (float(other_pos.x), float(other_pos.y)),
                skip_shapes,
            )
            if not hit or hit.shape.body is not other_body:
                continue

            bearing = (math.degrees(bearing_rad) + 360.0) % 360.0
            entry = {
                "type": "friend" if other_data.team == bot_data.team else "enemy",
                "x": float(other_pos.x),
                "y": float(other_pos.y),
                "distance": distance,
                "angle": bearing,
                "hp": int(other_data.hp),
                "team": f"team_{other_data.team}",
                "id": other_id,
                "velocity_x": float(other_body.velocity[0]),
                "velocity_y": float(other_body.velocity[1]),
                "signal": getattr(other_data, "signal", "none"),
            }

            entity_results[("bot", other_id)] = entry

        for proj_id, proj_data in arena.projectile_data.items():
            proj_body = arena.projectile_bodies.get(proj_id)
            if not proj_body:
                continue

            proj_pos = proj_body.position
            dx = float(proj_pos.x) - bot_pos_x
            dy = float(proj_pos.y) - bot_pos_y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= 0.0 or dist_sq > sense_range_sq:
                continue

            bearing_rad = math.atan2(dy, dx)
            angle_diff = (bearing_rad - bot_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > half_fov:
                continue

            distance = math.sqrt(dist_sq)

            hit = self._segment_query_first_excluding(
                space,
                (bot_pos_x, bot_pos_y),
                (float(proj_pos.x), float(proj_pos.y)),
                skip_shapes,
            )
            if not hit or hit.shape.body is not proj_body:
                continue

            bearing = (math.degrees(bearing_rad) + 360.0) % 360.0
            entry = {
                "type": "projectile",
                "x": float(proj_pos.x),
                "y": float(proj_pos.y),
                "distance": distance,
                "angle": bearing,
                "velocity_x": float(proj_body.velocity[0]),
                "velocity_y": float(proj_body.velocity[1]),
                "ttl": proj_data.ttl,
                "team": f"team_{proj_data.team}",
            }

            entity_results[("projectile", proj_id)] = entry

        combined = list(entity_results.values()) + list(wall_hits.values())
        combined.sort(key=lambda obj: obj.get("distance", float("inf")))
        return combined

    def build_observation(self, arena, bot_id: int) -> Dict[str, Any]:
        """Construct full observation payload for a bot."""
        self_state = self._get_self_state(arena, bot_id)
        visible = self.generate_visible_objects(arena, bot_id)
        params = self._get_world_params(arena)
        precomp = self._compute_precomputations(self_state, visible, params)

        return {
            "self": self_state,
            "visible_objects": visible,
            "move_history": self.generate_move_history(arena, bot_id),
            "params": params,
            "precomp": precomp,
        }

    def _lead_point(
        self,
        sx: float,
        sy: float,
        tx: float,
        ty: float,
        tvx: float,
        tvy: float,
        proj_speed: float,
    ) -> Tuple[float, float]:
        rx, ry = tx - sx, ty - sy
        a = tvx * tvx + tvy * tvy - proj_speed * proj_speed
        b = 2.0 * (rx * tvx + ry * tvy)
        c = rx * rx + ry * ry
        t = 0.0
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                t = max(0.0, -c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                rdisc = math.sqrt(disc)
                t1 = (-b - rdisc) / (2.0 * a)
                t2 = (-b + rdisc) / (2.0 * a)
                cand = [t for t in (t1, t2) if t >= 0.0]
                t = min(cand) if cand else 0.0
        return (tx + tvx * t, ty + tvy * t)

    def _friend_in_line_of_fire(
        self,
        sx: float,
        sy: float,
        ax: float,
        ay: float,
        friends: List[Dict[str, Any]],
        radius: float = 0.85,
        cone_deg: float = 11.0,
    ) -> bool:
        vx, vy = ax - sx, ay - sy
        L = math.hypot(vx, vy)
        if L < 1e-6:
            return True
        ux, uy = vx / L, vy / L
        cos_thresh = math.cos(math.radians(cone_deg))
        for f in friends:
            fx, fy = float(f.get("x", 0.0)), float(f.get("y", 0.0))
            rfx, rfy = fx - sx, fy - sy
            proj = rfx * ux + rfy * uy
            if proj < 0.0 or proj > L:
                continue
            mag = math.hypot(rfx, rfy)
            if mag > 1e-6 and ((rfx * ux + rfy * uy) / mag) < cos_thresh:
                continue
            px, py = ux * proj, uy * proj
            dx, dy = rfx - px, rfy - py
            if math.hypot(dx, dy) < radius:
                return True
        return False

    def _compute_precomputations(
        self,
        self_state: Dict[str, Any],
        visible: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        sx, sy = float(self_state.get("x", 0.0)), float(self_state.get("y", 0.0))
        proj_speed = float(params.get("proj_speed", 12.0))
        enemies = [o for o in visible if o.get("type") == "enemy"]
        friends = [o for o in visible if o.get("type") == "friend"]
        walls = [o for o in visible if o.get("type") == "wall"]
        projectiles = [o for o in visible if o.get("type") == "projectile"]

        nearest_enemy = None
        if enemies:
            e = min(enemies, key=lambda o: float(o.get("distance", 1e9)))
            tx, ty = float(e.get("x", 0.0)), float(e.get("y", 0.0))
            tvx, tvy = float(e.get("velocity_x", 0.0)), float(e.get("velocity_y", 0.0))
            aim_x, aim_y = self._lead_point(sx, sy, tx, ty, tvx, tvy, proj_speed)
            safe = not self._friend_in_line_of_fire(sx, sy, aim_x, aim_y, friends)
            nearest_enemy = {
                "id": e.get("id"),
                "x": tx,
                "y": ty,
                "distance": float(e.get("distance", 0.0)),
                "angle": float(e.get("angle", 0.0)),
                "hp": int(e.get("hp", 0)),
                "aim_x": aim_x,
                "aim_y": aim_y,
                "safe_to_fire": bool(safe),
            }

        nearest_friend = None
        if friends:
            f = min(friends, key=lambda o: float(o.get("distance", 1e9)))
            nearest_friend = {
                "id": f.get("id"),
                "x": float(f.get("x", 0.0)),
                "y": float(f.get("y", 0.0)),
                "distance": float(f.get("distance", 0.0)),
                "angle": float(f.get("angle", 0.0)),
                "signal": f.get("signal", "none"),
            }

        enemies_close = sum(1 for e in enemies if float(e.get("distance", 1e9)) < 5.0)
        enemies_mid = sum(
            1 for e in enemies if 5.0 <= float(e.get("distance", 1e9)) <= 10.0
        )
        enemies_far = sum(1 for e in enemies if float(e.get("distance", 0.0)) > 10.0)

        nearest_wall = None
        if walls:
            w = min(walls, key=lambda o: float(o.get("distance", 1e9)))
            nearest_wall = {
                "distance": float(w.get("distance", 0.0)),
                "angle": float(w.get("angle", 0.0)),
                "x": float(w.get("x", 0.0)),
                "y": float(w.get("y", 0.0)),
            }

        # Incoming projectile threat estimate (min time to closest approach)
        min_tta = None
        threat_dir = None
        approaching = 0
        for p in projectiles:
            px, py = float(p.get("x", 0.0)), float(p.get("y", 0.0))
            vx, vy = float(p.get("velocity_x", 0.0)), float(p.get("velocity_y", 0.0))
            rx, ry = sx - px, sy - py
            v2 = vx * vx + vy * vy
            if v2 <= 1e-8:
                continue
            tca = - (rx * vx + ry * vy) / v2
            if tca < 0.0:
                continue
            cx, cy = rx + vx * tca, ry + vy * tca
            sep = math.hypot(cx, cy)
            if sep < 1.5:  # within danger tube
                approaching += 1
                if min_tta is None or tca < min_tta:
                    min_tta = tca
                    threat_dir = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0

        # Nearby friend signals tally
        signal_counts: Dict[str, int] = {}
        for f in friends:
            sig = f.get("signal", "none")
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        return {
            "enemy_count": len(enemies),
            "friend_count": len(friends),
            "enemies_close": enemies_close,
            "enemies_mid": enemies_mid,
            "enemies_far": enemies_far,
            "nearest_enemy": nearest_enemy,
            "nearest_friend": nearest_friend,
            "nearest_wall": nearest_wall,
            "incoming_projectiles": {
                "count_threats": approaching,
                "min_time_to_approach": min_tta if min_tta is not None else float("inf"),
                "approach_dir_deg": threat_dir,
            },
            "friend_signals": signal_counts,
        }

    def build_llm_prompt(
        self,
        arena,
        bot_id: int,
        allowed_signals: List[str] | Tuple[str, ...] | None = None,
    ) -> str:
        """Canonical prompt builder for LLM-generated bot functions.

        Summarizes the observation and precomputed features to keep bot code short.
        """
        obs = self.build_observation(arena, bot_id)
        allow = list(allowed_signals) if allowed_signals is not None else []
        return self.format_llm_prompt_from_observation(obs, allow)

    @staticmethod
    def format_llm_prompt_from_observation(
        observation: Dict[str, Any],
        allowed_signals: Iterable[str] | None = None,
    ) -> str:
        """Format an LLM prompt using a pre-built observation payload."""
        pre = observation.get("precomp", {})
        allow = list(allowed_signals or [])

        lines: List[str] = []
        lines.append(
            "You are writing a short Python function bot_function(observation) for a 2D combat sim."
        )
        lines.append(
            "Rules: return a dict with action in {move, rotate, dodge, fire}; optionally set 'signal' from allowed_signals; keep code <= ~70 lines; avoid prints; be efficient."
        )
        lines.append(
            "Constraints: avoid friendly fire; dodge incoming projectiles; respect can_fire cooldown; use math only; no imports or I/O."
        )
        if allow:
            lines.append(f"Allowed signals: {allow}.")
        lines.append(
            "Observation keys: 'self', 'visible_objects', 'params', 'precomp', 'memory', 'allowed_signals'."
        )
        lines.append("Precomp fields simplify logic (use them if helpful):")
        lines.append(
            f"- counts: enemy_count={pre.get('enemy_count')}, friend_count={pre.get('friend_count')}, close/mid/far={pre.get('enemies_close')}/{pre.get('enemies_mid')}/{pre.get('enemies_far')}."
        )
        ne = pre.get("nearest_enemy") or {}
        lines.append(
            f"- nearest_enemy: id={ne.get('id')}, d={ne.get('distance')}, angle={ne.get('angle')}, aim=({ne.get('aim_x')},{ne.get('aim_y')}), safe_to_fire={ne.get('safe_to_fire')}."
        )
        nf = pre.get("nearest_friend") or {}
        lines.append(
            f"- nearest_friend: id={nf.get('id')}, d={nf.get('distance')}, signal={nf.get('signal')}."
        )
        inc = pre.get("incoming_projectiles") or {}
        lines.append(
            f"- incoming_projectiles: count_threats={inc.get('count_threats')}, min_time_to_approach={inc.get('min_time_to_approach')}."
        )
        nw = pre.get("nearest_wall") or {}
        lines.append(f"- nearest_wall: d={nw.get('distance')}, angle={nw.get('angle')}.")
        lines.append(
            "Strategy suggestions: if threats imminent -> dodge; if safe_to_fire and can_fire -> fire at aim; otherwise move to better position (flank/orbit/cover). Set helpful 'signal'."
        )
        lines.append("Return only one action dict. Use observation['memory'] if needed.")
        return "\n".join(lines)

    def generate_move_history(self, arena, bot_id: int) -> List[Dict[str, Any]]:
        """Return recent move history for a bot."""
        history = self.move_history.get(bot_id)
        if not history:
            return []
        return [dict(entry) for entry in history]

    def record_bot_action(self, arena, bot_id: int, action: Dict[str, Any], executed: bool):
        """Record the latest action outcome for move history tracking."""
        history = self.move_history.setdefault(
            bot_id, deque(maxlen=self.MOVE_HISTORY_LIMIT)
        )
        entry: Dict[str, Any] = {
            "tick": arena.tick,
            "time": round(arena.time, 3),
            "executed": bool(executed),
        }

        if action:
            entry["action"] = action.get("action", "none")
            entry["signal"] = action.get("signal", "none")
            for key in ("target_x", "target_y", "angle", "direction"):
                if key in action:
                    entry[key] = action[key]
        else:
            entry["action"] = "none"
            entry["signal"] = "none"

        history.append(entry)

    def _get_self_state(self, arena, bot_id: int) -> Dict[str, Any]:
        """Assemble the observing bot's own state."""
        bot_body = arena.bot_bodies.get(bot_id)
        bot_data = arena.bot_data.get(bot_id)

        if not bot_body or not bot_data:
            return {
                "id": bot_id,
                "team": None,
                "x": 0.0,
                "y": 0.0,
                "theta_deg": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "hp": 0,
                "cooldown_remaining": float("inf"),
                "can_fire": False,
                "time": round(arena.time, 3),
                "tick": arena.tick,
                "alive": False,
            }

        cooldown_remaining, can_fire = arena.get_fire_status(bot_id)
        theta_deg = (math.degrees(bot_body.angle) + 360.0) % 360.0

        return {
            "id": bot_id,
            "team": bot_data.team,
            "x": float(bot_body.position[0]),
            "y": float(bot_body.position[1]),
            "theta_deg": theta_deg,
            "vx": float(bot_body.velocity[0]),
            "vy": float(bot_body.velocity[1]),
            "hp": max(0, int(bot_data.hp)),
            "cooldown_remaining": round(cooldown_remaining, 3),
            "can_fire": bool(can_fire),
            "time": round(arena.time, 3),
            "tick": arena.tick,
            "alive": arena._is_bot_alive(bot_id),
            "signal": getattr(bot_data, "signal", "none"),
        }

    def _get_world_params(self, arena) -> Dict[str, Any]:
        """Expose relevant arena configuration parameters."""
        return {
            "dt_physics": arena.DT_PHYSICS,
            "dt_control": arena.DT_CONTROL,
            "v_max": arena.V_MAX,
            "v_rev_max": arena.V_REV_MAX,
            "a_max": arena.A_MAX,
            "omega_max_deg": math.degrees(arena.OMEGA_MAX),
            "proj_speed": arena.PROJ_SPEED,
            "proj_ttl": arena.PROJ_TTL,
            "proj_damage": arena.PROJ_DAMAGE,
            "fire_rate": arena.FIRE_RATE,
            "fire_cooldown": arena.FIRE_COOLDOWN,
            "fov_deg": math.degrees(arena.FOV_ANGLE),
            "sense_range": arena.SENSE_RANGE,
            "arena_width": arena.ARENA_SIZE[0],
            "arena_height": arena.ARENA_SIZE[1],
            "bots_per_side": arena.BOTS_PER_SIDE,
        }

    def update_bot_function(self, bot_id: int, new_function_source: str):
        """Update a bot's function source (simulating LLM rewriting)."""
        self.bot_function_sources[bot_id] = new_function_source

    # --- LLM polling helpers ---

    @staticmethod
    def call_openai_compatible_endpoint(
        endpoint_url: str,
        api_key: str,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        timeout: float = 10.0,
    ) -> str:
        """Call an OpenAI-compatible chat/completions endpoint and return assistant content."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a tactical programming assistant that outputs concise Python functions.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = request.Request(endpoint_url, data=data, headers=headers, method="POST")
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - guard
            raise ValueError("Unexpected response format from LLM endpoint") from exc

    def regenerate_function_from_observation(
        self,
        observation: Dict[str, Any],
        allowed_signals: Iterable[str] | None = None,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        timeout: float = 10.0,
    ) -> str:
        """Generate/refresh a bot function given an observation via remote or local fallback.

        If endpoint_url is None, returns a default template for rapid iteration.
        """
        prompt = self.format_llm_prompt_from_observation(observation, allowed_signals)
        bot_id = int((observation.get("self", {}) or {}).get("id", 0))
        if endpoint_url and endpoint_url.strip().lower() == "mock":
            return self._pick_mock_template(bot_id)
        if endpoint_url and api_key:
            try:
                return self.call_openai_compatible_endpoint(
                    endpoint_url, api_key, prompt, model=model, timeout=timeout
                )
            except error.URLError:
                # Fall back to local template if network fails
                pass
        # Fallback template selection
        templates = self.function_templates
        if templates:
            idx = (self._regen_counter + bot_id) % len(templates)
            self._regen_counter += 1
            return templates[idx]
        return self._pick_mock_template(bot_id)

    def _ensure_mock_templates(self) -> List[str]:
        if self._mock_templates is not None:
            return self._mock_templates

        try:
            templates = example_llm_gen_control_code._get_default_templates()
            self._mock_templates = [tpl.strip() for tpl in templates if tpl and tpl.strip()]
        except Exception:
            self._mock_templates = [
                (
                    "def bot_function(observation):\n"
                    "    return {'action': 'rotate', 'angle': 45.0, 'signal': 'ready'}\n"
                )
            ]

        return self._mock_templates

    def _pick_mock_template(self, bot_id: int | None = None) -> str:
        templates = self._ensure_mock_templates()
        if not templates:
            return (
                "def bot_function(observation):\n"
                "    return {'action': 'rotate', 'angle': 45.0, 'signal': 'ready'}\n"
            )
        pool = templates[:3] if len(templates) >= 3 else templates
        return self._mock_rng.choice(pool)

    def regenerate_bot_function(
        self,
        arena,
        bot_id: int,
        allowed_signals: Iterable[str] | None = None,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        timeout: float = 10.0,
    ) -> str:
        """Build prompt from arena state and fetch/choose the bot function source."""
        obs = self.build_observation(arena, bot_id)
        return self.regenerate_function_from_observation(
            obs,
            allowed_signals=allowed_signals,
            endpoint_url=endpoint_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
