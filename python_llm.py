"""
Python Function LLM Controller
Generates Python functions for bot control instead of DSL programs.
"""

import math
import random
from typing import Dict, List, Any, Tuple


class PythonLLMController:
    """Generates Python functions for bot control and manages bot observations."""

    def __init__(self, bot_count: int = 4):
        self.bot_count = bot_count
        self.bot_function_sources = {}
        self.function_templates = self._load_function_templates()

        # Assign different function types to bots
        for bot_id in range(bot_count):
            template_idx = bot_id % len(self.function_templates)
            self.bot_function_sources[bot_id] = self.function_templates[template_idx]

    def _load_function_templates(self) -> List[str]:
        """Load different bot function templates."""
        return [
            self._get_aggressive_function(),
            self._get_defensive_function(),
            self._get_balanced_function(),
            self._get_sniper_function(),
        ]

    def get_bot_function_source(self, bot_id: int) -> str:
        """Get Python function source code for a bot."""
        return self.bot_function_sources.get(bot_id, self._get_balanced_function())

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
        target_angle = math.atan2(target_x - bot_x, target_y - bot_y)

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
        """Ray march to find visible objects within FOV, accounting for occlusion."""
        if not arena._is_bot_alive(bot_id):
            return []

        visible_objects = []

        bot_body = arena.bot_bodies[bot_id]
        bot_data = arena.bot_data[bot_id]
        bot_x, bot_y = bot_body.position
        bot_heading = bot_body.angle  # Get bot's current heading

        # Ray marching parameters
        max_range = arena.SENSE_RANGE  # 15m
        fov_angle = arena.FOV_ANGLE  # 120 degrees
        ray_count = 24  # Cast 24 rays across the FOV (every 5 degrees)

        # Calculate ray directions
        start_angle = bot_heading - fov_angle / 2
        angle_step = fov_angle / (ray_count - 1)

        for i in range(ray_count):
            ray_angle = start_angle + i * angle_step
            ray_end_x = bot_x + math.sin(ray_angle) * max_range
            ray_end_y = bot_y + math.cos(ray_angle) * max_range

            # Find closest intersection along this ray
            closest_distance = float("inf")
            closest_object = None
            ray_objects = []  # Collect all objects on this ray for proper occlusion handling

            # Check walls first (they block everything behind them)
            if hasattr(arena, "wall_bodies"):
                for wall_body, wall_shape in arena.wall_bodies:
                    seg_start = wall_shape.a
                    seg_end = wall_shape.b

                    intersects, distance, point = self._ray_intersects_segment(
                        (bot_x, bot_y), (ray_end_x, ray_end_y), seg_start, seg_end
                    )

                    if intersects:
                        # Filter out arena boundary walls
                        wall_length = math.sqrt(
                            (seg_end[0] - seg_start[0]) ** 2
                            + (seg_end[1] - seg_start[1]) ** 2
                        )
                        if wall_length < 15.0:  # Only interior walls
                            bearing = (
                                math.degrees(
                                    math.atan2(point[0] - bot_x, point[1] - bot_y)
                                )
                                % 360
                            )
                            ray_objects.append(
                                {
                                    "type": "wall",
                                    "x": point[0],
                                    "y": point[1],
                                    "distance": distance,
                                    "angle": bearing,
                                    "wall_start_x": seg_start[0],
                                    "wall_start_y": seg_start[1],
                                    "wall_end_x": seg_end[0],
                                    "wall_end_y": seg_end[1],
                                }
                            )

            # Check bots
            for other_id, other_data in arena.bot_data.items():
                if other_id == bot_id or not arena._is_bot_alive(other_id):
                    continue

                other_body = arena.bot_bodies[other_id]
                other_x, other_y = other_body.position

                intersects, distance, point = self._ray_intersects_circle(
                    (bot_x, bot_y),
                    (ray_end_x, ray_end_y),
                    (other_x, other_y),
                    arena.BOT_RADIUS,
                )

                if intersects:
                    bearing = (
                        math.degrees(math.atan2(other_x - bot_x, other_y - bot_y)) % 360
                    )

                    bot_type = "friend" if other_data.team == bot_data.team else "enemy"
                    ray_objects.append(
                        {
                            "type": bot_type,
                            "x": other_x,
                            "y": other_y,
                            "distance": distance,
                            "angle": bearing,
                            "hp": int(other_data.hp),
                            "team": f"team_{other_data.team}",
                            "id": other_id,
                            "velocity_x": other_body.velocity[0],
                            "velocity_y": other_body.velocity[1],
                            "signal": other_data.signal
                            if hasattr(other_data, "signal")
                            else "none",
                        }
                    )

            # Check projectiles (always include unless blocked)
            for proj_id, proj_data in arena.projectile_data.items():
                proj_body = arena.projectile_bodies.get(proj_id)
                if not proj_body:
                    continue

                proj_x, proj_y = proj_body.position

                intersects, distance, point = self._ray_intersects_circle(
                    (bot_x, bot_y),
                    (ray_end_x, ray_end_y),
                    (proj_x, proj_y),
                    0.1,  # Small projectile radius
                )

                if intersects:
                    bearing = (
                        math.degrees(math.atan2(proj_x - bot_x, proj_y - bot_y)) % 360
                    )
                    ray_objects.append(
                        {
                            "type": "projectile",
                            "x": proj_x,
                            "y": proj_y,
                            "distance": distance,
                            "angle": bearing,
                            "velocity_x": proj_body.velocity[0],
                            "velocity_y": proj_body.velocity[1],
                            "ttl": proj_data.ttl,
                            "team": f"team_{proj_data.team}",
                        }
                    )

            # Sort objects by distance and add visible ones
            ray_objects.sort(key=lambda obj: obj["distance"])

            # Add all objects that aren't blocked by closer ones
            blocking_distance = float("inf")
            for obj in ray_objects:
                # Walls block everything behind them
                if obj["type"] == "wall":
                    blocking_distance = min(blocking_distance, obj["distance"])
                    visible_objects.append(obj)
                elif obj["distance"] < blocking_distance:
                    # Object is visible (not blocked by walls)
                    visible_objects.append(obj)
                    # Bots block other objects behind them, but projectiles don't block much
                    if obj["type"] in ["enemy", "friend"]:
                        # Only block objects very close behind (within bot radius)
                        blocking_distance = min(
                            blocking_distance, obj["distance"] + arena.BOT_RADIUS * 0.5
                        )

        # Remove duplicates (objects hit by multiple rays)
        unique_objects = []
        for obj in visible_objects:
            is_duplicate = False
            for existing in unique_objects:
                if (
                    existing["type"] == obj["type"]
                    and existing.get("id") == obj.get("id")
                    and abs(existing["x"] - obj["x"]) < 0.5
                    and abs(existing["y"] - obj["y"]) < 0.5
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_objects.append(obj)

        return unique_objects

    def generate_move_history(self, arena, bot_id: int) -> List[Dict]:
        """Generate move history for a bot (placeholder for now)."""
        # For now, return empty history - could be enhanced to track actual moves
        return []

    def _get_aggressive_function(self) -> str:
        """Aggressive bot function that charges at enemies."""
        return '''
def bot_function(visible_objects, move_history, allowed_signals):
    """Aggressive bot that charges at nearest enemy and fires aggressively."""
    
    # Check for immediate threats (projectiles)
    for obj in visible_objects:
        if obj.get('type') == 'projectile' and obj.get('distance', float('inf')) < 1.5:
            # Emergency dodge
            dodge_angle = (obj['angle'] + 90) % 360
            return {'action': 'dodge', 'direction': dodge_angle, 'signal': 'retreating'}
    
    # Find nearest enemy
    enemies = [obj for obj in visible_objects if obj.get('type') == 'enemy']
    if not enemies:
        # No enemies, search by rotating
        return {'action': 'rotate', 'angle': 45.0, 'signal': 'ready'}
    
    # Sort enemies by distance
    enemies.sort(key=lambda e: e.get('distance', float('inf')))
    target = enemies[0]
    
    distance = target['distance']
    
    if distance < 2.0:
        # Too close, dodge away
        retreat_angle = (target['angle'] + 180) % 360
        return {'action': 'dodge', 'direction': retreat_angle, 'signal': 'retreating'}
    elif distance < 8.0:
        # Good range for aggressive fire
        return {'action': 'fire', 'target_x': target['x'], 'target_y': target['y'], 'signal': 'firing'}
    else:
        # Charge towards enemy
        return {'action': 'move', 'target_x': target['x'], 'target_y': target['y'], 'signal': 'attacking'}
'''

    def _get_defensive_function(self) -> str:
        """Defensive bot function that maintains distance."""
        return '''
def bot_function(visible_objects, move_history, allowed_signals):
    """Defensive bot that keeps distance and fires carefully."""
    import math
    
    # Priority 1: Avoid projectiles
    projectiles = [obj for obj in visible_objects if obj.get('type') == 'projectile']
    for proj in projectiles:
        if proj.get('distance', float('inf')) < 2.5:
            # Calculate dodge direction perpendicular to projectile trajectory
            proj_angle = math.radians(proj['angle'])
            dodge_angle = math.degrees(proj_angle + math.pi/2)
            return {'action': 'dodge', 'direction': dodge_angle, 'signal': 'moving_to_cover'}
    
    # Priority 2: Engage enemies at safe distance
    enemies = [obj for obj in visible_objects if obj.get('type') == 'enemy']
    if not enemies:
        return {'action': 'rotate', 'angle': 90.0, 'signal': 'watching_flank'}
    
    # Find closest enemy
    enemies.sort(key=lambda e: e.get('distance', float('inf')))
    closest_enemy = enemies[0]
    distance = closest_enemy['distance']
    
    if distance < 6.0:
        # Too close, retreat while firing
        retreat_x = closest_enemy['x'] + 8 * math.cos(math.radians(closest_enemy['angle'] + 180))
        retreat_y = closest_enemy['y'] + 8 * math.sin(math.radians(closest_enemy['angle'] + 180))
        return {'action': 'move', 'target_x': retreat_x, 'target_y': retreat_y, 'signal': 'retreating'}
    elif distance < 12.0:
        # Perfect range, fire
        return {'action': 'fire', 'target_x': closest_enemy['x'], 'target_y': closest_enemy['y'], 'signal': 'cover_fire'}
    else:
        # Move closer but cautiously
        approach_distance = distance - 10.0  # Stay 10 units away
        approach_angle = math.radians(closest_enemy['angle'])
        approach_x = closest_enemy['x'] - approach_distance * math.cos(approach_angle)
        approach_y = closest_enemy['y'] - approach_distance * math.sin(approach_angle)
        return {'action': 'move', 'target_x': approach_x, 'target_y': approach_y, 'signal': 'advancing'}
'''

    def _get_balanced_function(self) -> str:
        """Balanced bot function with moderate aggression."""
        return '''
def bot_function(visible_objects, move_history, allowed_signals):
    """Balanced bot with adaptive tactics and team coordination."""
    
    # Count threats, allies, and cover
    enemies = [obj for obj in visible_objects if obj.get('type') == 'enemy']
    projectiles = [obj for obj in visible_objects if obj.get('type') == 'projectile']
    friends = [obj for obj in visible_objects if obj.get('type') == 'friend']
    walls = [obj for obj in visible_objects if obj.get('type') == 'wall']
    
    # Check friend signals for coordination
    backup_needed = any(friend.get('signal') == 'need_backup' for friend in friends)
    
    # Emergency projectile avoidance
    for proj in projectiles:
        if proj.get('distance', float('inf')) < 2.0:
            dodge_angle = (proj['angle'] + 90) % 360
            return {'action': 'dodge', 'direction': dodge_angle, 'signal': 'moving_to_cover'}
    
    if not enemies:
        # No enemies - use walls for tactical positioning
        if walls and not backup_needed:
            nearest_wall = min(walls, key=lambda w: w.get('distance', float('inf')))
            if nearest_wall['distance'] > 3.0:
                # Move towards wall for cover
                return {'action': 'move', 'target_x': nearest_wall['x'], 'target_y': nearest_wall['y'], 'signal': 'positioning'}
        
        signal = 'regrouping' if backup_needed else 'ready'
        return {'action': 'rotate', 'angle': 135.0, 'signal': signal}
    
    # Find best target (closest enemy)
    enemies.sort(key=lambda e: e.get('distance', float('inf')))
    primary_target = enemies[0]
    distance = primary_target['distance']
    
    # Adaptive behavior based on enemy count and distance
    enemy_count = len(enemies)
    
    if enemy_count >= 2 and distance < 5.0:
        # Multiple enemies close - retreat and call for backup
        retreat_angle = (primary_target['angle'] + 180) % 360
        return {'action': 'dodge', 'direction': retreat_angle, 'signal': 'need_backup'}
    elif distance < 3.0:
        # Single enemy too close - dodge
        dodge_angle = (primary_target['angle'] + 90) % 360
        return {'action': 'dodge', 'direction': dodge_angle, 'signal': 'retreating'}
    elif distance < 10.0:
        # Good firing range
        signal = 'focus_fire' if backup_needed else 'firing'
        return {'action': 'fire', 'target_x': primary_target['x'], 'target_y': primary_target['y'], 'signal': signal}
    else:
        # Move closer
        signal = 'advancing' if backup_needed else 'attacking'
        return {'action': 'move', 'target_x': primary_target['x'], 'target_y': primary_target['y'], 'signal': signal}
'''

    def _get_sniper_function(self) -> str:
        """Sniper bot function that prioritizes long-range combat."""
        return '''
def bot_function(visible_objects, move_history, allowed_signals):
    """Sniper bot that prefers long-range engagement and provides overwatch."""
    import math
    
    # Immediate threat assessment
    projectiles = [obj for obj in visible_objects if obj.get('type') == 'projectile']
    friends = [obj for obj in visible_objects if obj.get('type') == 'friend']
    
    for proj in projectiles:
        if proj.get('distance', float('inf')) < 1.8:
            # Quick dodge
            return {'action': 'dodge', 'direction': (proj['angle'] + 90) % 360, 'signal': 'moving_to_cover'}
    
    enemies = [obj for obj in visible_objects if obj.get('type') == 'enemy']
    if not enemies:
        # Check if friends need support
        friends_in_combat = any(friend.get('signal') in ['need_backup', 'retreating'] for friend in friends)
        signal = 'watching_flank' if friends_in_combat else 'holding_position'
        return {'action': 'rotate', 'angle': 180.0, 'signal': signal}
    
    # Sort enemies by distance (prefer longer range targets)
    enemies.sort(key=lambda e: e.get('distance', float('inf')))
    
    # Look for enemies at preferred range (8-15 units)
    preferred_targets = [e for e in enemies if 8.0 <= e.get('distance', 0) <= 15.0]
    
    if preferred_targets:
        # Fire at target in preferred range
        target = preferred_targets[0]
        return {'action': 'fire', 'target_x': target['x'], 'target_y': target['y'], 'signal': 'cover_fire'}
    
    closest_enemy = enemies[0]
    distance = closest_enemy['distance']
    
    if distance < 8.0:
        # Too close, back away to preferred range
        retreat_distance = 12.0 - distance
        retreat_angle = math.radians(closest_enemy['angle'] + 180)
        retreat_x = closest_enemy['x'] + retreat_distance * math.cos(retreat_angle)
        retreat_y = closest_enemy['y'] + retreat_distance * math.sin(retreat_angle)
        return {'action': 'move', 'target_x': retreat_x, 'target_y': retreat_y, 'signal': 'retreating'}
    else:
        # Enemy too far, move to optimal range
        optimal_distance = 10.0
        approach_angle = math.radians(closest_enemy['angle'])
        optimal_x = closest_enemy['x'] - optimal_distance * math.cos(approach_angle)
        optimal_y = closest_enemy['y'] - optimal_distance * math.sin(approach_angle)
        return {'action': 'move', 'target_x': optimal_x, 'target_y': optimal_y, 'signal': 'advancing'}
'''

    def update_bot_function(self, bot_id: int, new_function_source: str):
        """Update a bot's function source (simulating LLM rewriting)."""
        self.bot_function_sources[bot_id] = new_function_source

    def get_bot_personality(self, bot_id: int) -> str:
        """Get bot personality description."""
        personalities = ["aggressive", "defensive", "balanced", "sniper"]
        return personalities[bot_id % len(personalities)]
