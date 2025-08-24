"""
LLM Battle Simulator - Core Arena and Physics with pymunk
Handles 2v2 battles with robust physics and deterministic simulation.
"""

import math
import time
import json
import pymunk
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BotData:
    """Bot metadata not stored in pymunk body."""

    team: int
    hp: float
    fire_command: bool
    last_fire_time: float
    shots_fired: int
    shots_hit: int
    damage_dealt: float
    damage_taken: float
    kills: int
    deaths: int
    signal: str = "none"  # Bot's current signal/status for team communication


@dataclass
class ProjectileData:
    """Projectile metadata not stored in pymunk body."""

    shooter_id: int
    team: int
    birth_tick: int
    ttl: float


class Arena:
    """2v2 Battle Arena with pymunk physics simulation."""

    # Default arena configuration (can be overridden in __init__)
    DEFAULT_ARENA_SIZE = (20.0, 20.0)  # 20x20m for tight 2v2 action
    DEFAULT_BOTS_PER_SIDE = 2
    BOT_RADIUS = 0.4  # meters
    WALL_THICKNESS = 0.2  # meters - used for all walls (perimeter and interior)
    MAX_HP = 100

    # Physics parameters
    DT_PHYSICS = 1.0 / 240.0  # 240Hz physics
    DT_CONTROL = 1.0 / 120.0  # 120Hz control
    V_MAX = 2.0  # max speed m/s
    V_REV_MAX = 1.0  # max reverse speed m/s
    A_MAX = 8.0  # max acceleration m/s²
    OMEGA_MAX = math.radians(260)  # max turn rate rad/s
    LINEAR_DAMPING = 0.1  # velocity damping

    # Projectile configuration
    PROJ_SPEED = 6.0  # m/s
    PROJ_TTL = 5.0  # seconds
    PROJ_DAMAGE = 25  # HP
    FIRE_RATE = 8.0  # Hz (shots per second)

    # Perception configuration
    FOV_ANGLE = math.radians(120)  # 120° field of view
    SENSE_RANGE = 15.0  # meters

    # Collision types for pymunk
    COLLISION_TYPE_BOT = 1
    COLLISION_TYPE_PROJECTILE = 2
    COLLISION_TYPE_WALL = 3

    def __init__(
        self,
        seed: int = 42,
        spawn_config: Dict = None,
        arena_size: Optional[Tuple[float, float]] = None,
        bots_per_side: Optional[int] = None,
    ):
        """Initialize arena with deterministic seed and optional spawn configuration."""
        import random

        self.rng = random.Random(seed)

        # Set arena configuration
        self.ARENA_SIZE = arena_size if arena_size else self.DEFAULT_ARENA_SIZE
        self.BOTS_PER_SIDE = (
            bots_per_side if bots_per_side else self.DEFAULT_BOTS_PER_SIDE
        )
        self.BOT_COUNT = self.BOTS_PER_SIDE * 2

        # Create pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity in top-down view

        # Bot data storage (metadata not in pymunk bodies)
        self.bot_data: Dict[int, BotData] = {}
        self.projectile_data: Dict[int, ProjectileData] = {}

        # Pymunk bodies storage
        self.bot_bodies: Dict[int, pymunk.Body] = {}
        self.projectile_bodies: Dict[int, pymunk.Body] = {}

        # Simulation state
        self.tick = 0
        self.physics_tick = 0
        self.time = 0.0
        self.battle_log = []
        self.events = []
        self.walls = []
        self.next_projectile_id = 1000  # Start projectile IDs at 1000

        # Setup collision handlers
        self._setup_collision_handlers()

        # Create arena walls
        self._create_arena_walls()

        # Initialize bots
        self._spawn_bots(spawn_config or {})

    def _spawn_bots(self, spawn_config: Dict):
        """Spawn bots in starting positions with optional randomization."""
        team0_quadrant = spawn_config.get("team0_quadrant", "left")
        team1_quadrant = spawn_config.get("team1_quadrant", "right")
        randomize = spawn_config.get("randomize_within_quadrants", True)

        margin = self.BOT_RADIUS + 0.5
        W, H = self.ARENA_SIZE
        quadrants = {
            "left": (-W / 2 + margin, -1, -H / 2 + margin, H / 2 - margin),
            "right": (1, W / 2 - margin, -H / 2 + margin, H / 2 - margin),
            "top": (-W / 2 + margin, W / 2 - margin, 1, H / 2 - margin),
            "bottom": (-W / 2 + margin, W / 2 - margin, -H / 2 + margin, -1),
        }

        team0_bounds = quadrants[team0_quadrant]
        team1_bounds = quadrants[team1_quadrant]

        # Spawn team 0
        for i in range(self.BOTS_PER_SIDE):
            if randomize:
                x = self.rng.uniform(team0_bounds[0], team0_bounds[1])
                y = self.rng.uniform(team0_bounds[2], team0_bounds[3])
                dx, dy = -x, -y  # Vector to center (0,0)
                center_angle = math.atan2(dy, dx)
                angle_variation = self.rng.uniform(-math.pi / 6, math.pi / 6)
                theta = center_angle + angle_variation
            else:
                # Fixed positions (simplified for centered arena)
                if team0_quadrant == "left":
                    x, y = -W / 2 + margin + i * 2, 0
                    theta = 0  # Face East
                else:  # Default to right
                    x, y = W / 2 - margin - i * 2, 0
                    theta = math.pi  # Face West
            self._create_bot(i, x, y, theta, team=0)

        # Spawn team 1
        for i in range(self.BOTS_PER_SIDE):
            bot_id = i + self.BOTS_PER_SIDE
            if randomize:
                x = self.rng.uniform(team1_bounds[0], team1_bounds[1])
                y = self.rng.uniform(team1_bounds[2], team1_bounds[3])
                dx, dy = -x, -y
                center_angle = math.atan2(dy, dx)
                angle_variation = self.rng.uniform(-math.pi / 6, math.pi / 6)
                theta = center_angle + angle_variation
            else:
                if team1_quadrant == "right":
                    x, y = W / 2 - margin - i * 2, 0
                    theta = math.pi  # Face West
                else:  # Default to left
                    x, y = -W / 2 + margin + i * 2, 0
                    theta = 0  # Face East
            self._create_bot(bot_id, x, y, theta, team=1)

    def step_physics(self):
        """Single physics step at 240Hz using pymunk."""
        dt = self.DT_PHYSICS

        # Update bot control forces
        for bot_id in range(self.BOT_COUNT):
            if bot_id not in self.bot_data or not self._is_bot_alive(bot_id):
                continue

            body = self.bot_bodies[bot_id]
            bot_data = self.bot_data[bot_id]

            # Apply acceleration toward target velocity
            current_vel = body.velocity
            target_vel = (
                getattr(self, "target_vx", [0] * self.BOT_COUNT)[bot_id],
                getattr(self, "target_vy", [0] * self.BOT_COUNT)[bot_id],
            )

            dvx = target_vel[0] - current_vel[0]
            dvy = target_vel[1] - current_vel[1]

            # Clamp acceleration
            ax = max(-self.A_MAX, min(self.A_MAX, dvx / dt))
            ay = max(-self.A_MAX, min(self.A_MAX, dvy / dt))

            # Apply force (F = ma, assuming unit mass)
            force = (ax, ay)
            body.force = force

            # Update heading with turn rate limit
            target_theta = getattr(self, "target_theta", [0] * self.BOT_COUNT)[bot_id]
            current_angle = body.angle
            theta_diff = self._normalize_angle(target_theta - current_angle)
            max_dtheta = self.OMEGA_MAX * dt
            dtheta = max(-max_dtheta, min(max_dtheta, theta_diff))
            body.angular_velocity = dtheta / dt

        # Update projectile TTL and remove expired ones
        self._update_projectile_ttl(dt)

        # Step pymunk physics
        self.space.step(dt)

        # Apply velocity damping and limits
        self._apply_velocity_constraints()

        self.physics_tick += 1
        self.time += dt

    def step_control(self):
        """Single control step at 120Hz. This is where DSL gets executed."""
        # Initialize control arrays if not present for legacy compatibility
        if not hasattr(self, "target_vx"):
            self.target_vx = [0.0] * self.BOT_COUNT
            self.target_vy = [0.0] * self.BOT_COUNT
            self.target_theta = [body.angle for body in self.bot_bodies.values()]
            self.fire_command = [False] * self.BOT_COUNT

        self.tick += 1

    def set_bot_commands(
        self, bot_id: int, move_votes: Dict, rotate_votes: Dict, fire_votes: Dict
    ):
        """Apply DSL voting results to bot commands."""
        if bot_id not in self.bot_data:
            return

        body = self.bot_bodies[bot_id]

        # Initialize control arrays if not present
        if not hasattr(self, "target_vx"):
            self.target_vx = [0.0] * self.BOT_COUNT
            self.target_vy = [0.0] * self.BOT_COUNT
            self.target_theta = [body.angle for body in self.bot_bodies.values()]
            self.fire_command = [False] * self.BOT_COUNT

        # Movement commands
        if move_votes:
            winning_move = max(move_votes.items(), key=lambda x: x[1])
            direction, speed, _ = winning_move[0]  # (direction, speed, weight)

            # Convert direction and speed to target velocity using current heading
            current_angle = body.angle
            if direction == "FWD":
                dx, dy = math.cos(current_angle), math.sin(current_angle)
            elif direction == "BACK":
                dx, dy = -math.cos(current_angle), -math.sin(current_angle)
                speed = min(speed, self.V_REV_MAX / self.V_MAX)  # limit reverse speed
            elif direction == "LEFT":
                dx, dy = -math.sin(current_angle), math.cos(current_angle)
            elif direction == "RIGHT":
                dx, dy = math.sin(current_angle), -math.cos(current_angle)
            else:
                dx, dy = 0, 0

            self.target_vx[bot_id] = dx * speed * self.V_MAX
            self.target_vy[bot_id] = dy * speed * self.V_MAX

        # Rotation commands
        if rotate_votes:
            winning_rotate = max(rotate_votes.items(), key=lambda x: x[1])
            target_type, target_value, _ = winning_rotate[0]

            if target_type == "HEADING":
                self.target_theta[bot_id] = math.radians(target_value)
            elif target_type == "TARGET":
                # Resolve target bearing to absolute heading
                target_heading = self._resolve_target_bearing(bot_id, target_value)
                if target_heading is not None:
                    self.target_theta[bot_id] = target_heading

        # Fire commands - now handled through voting system
        if fire_votes:
            winning_fire = max(fire_votes.items(), key=lambda x: x[1])
            fire_state, _ = winning_fire[0]  # (fire_state, reason)
            self.fire_command[bot_id] = fire_state == "ON"

    def set_single_bot_action(self, bot_id: int, action: Optional[Dict]):
        """Apply single winning action to bot (supports both DSL and Python function actions)."""
        if bot_id not in self.bot_data or not action:
            return

        body = self.bot_bodies[bot_id]

        # Initialize control arrays if not present
        if not hasattr(self, "target_vx"):
            self.target_vx = [0.0] * self.BOT_COUNT
            self.target_vy = [0.0] * self.BOT_COUNT
            self.target_theta = [body.angle for body in self.bot_bodies.values()]
            self.fire_command = [False] * self.BOT_COUNT

        action_type = action.get(
            "type", action.get("action")
        )  # Support both 'type' and 'action' keys

        if action_type == "move":
            # Handle movement action - support both DSL and Python function formats
            if "target_x" in action and "target_y" in action:
                # Python function format: move to specific coordinates
                target_x = action["target_x"]
                target_y = action["target_y"]

                # Convert target position to velocity vector
                current_x, current_y = body.position
                dx = target_x - current_x
                dy = target_y - current_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > 0:
                    # Normalize and scale to max velocity
                    scale = (
                        min(self.V_MAX, distance * 2.0) / distance
                    )  # Scale based on distance
                    self.target_vx[bot_id] = dx * scale
                    self.target_vy[bot_id] = dy * scale

            else:
                # DSL format: directional movement
                direction = action.get("direction")
                speed = action.get("speed", 1.0)

                # Convert direction and speed to target velocity using current heading
                current_angle = body.angle
                if direction == "FWD":
                    dx, dy = math.cos(current_angle), math.sin(current_angle)
                elif direction == "BACK":
                    dx, dy = -math.cos(current_angle), -math.sin(current_angle)
                    speed = min(
                        speed, self.V_REV_MAX / self.V_MAX
                    )  # limit reverse speed
                elif direction == "LEFT":
                    dx, dy = -math.sin(current_angle), math.cos(current_angle)
                elif direction == "RIGHT":
                    dx, dy = math.sin(current_angle), -math.cos(current_angle)
                else:
                    dx, dy = 0, 0

                self.target_vx[bot_id] = dx * speed * self.V_MAX
                self.target_vy[bot_id] = dy * speed * self.V_MAX

        elif action_type == "rotate":
            # Handle rotation action
            if "angle" in action:
                # Python function format: absolute angle
                self.target_theta[bot_id] = math.radians(action["angle"])
            elif "target_value" in action:
                # DSL format: target value
                self.target_theta[bot_id] = math.radians(action["target_value"])

        elif action_type == "dodge":
            # Handle dodge action (Python function format)
            dodge_angle = math.radians(action.get("direction", 0))
            dodge_speed = 1.5  # Dodge at higher speed

            # Move in the dodge direction
            dx = math.sin(dodge_angle) * dodge_speed
            dy = math.cos(dodge_angle) * dodge_speed

            self.target_vx[bot_id] = dx * self.V_MAX
            self.target_vy[bot_id] = dy * self.V_MAX

        elif action_type == "fire":
            # Handle fire action
            if "target_x" in action and "target_y" in action:
                # Python function format: aim at specific coordinates
                target_x = action["target_x"]
                target_y = action["target_y"]

                # Calculate heading to target
                current_x, current_y = body.position
                dx = target_x - current_x
                dy = target_y - current_y
                target_heading = math.atan2(dy, dx)

                # Set heading and enable firing
                self.target_theta[bot_id] = target_heading
                self.fire_command[bot_id] = True
            else:
                # DSL format: fire state
                self.fire_command[bot_id] = action.get("fire", False)

    def try_fire_projectile(self, bot_id: int) -> bool:
        """Attempt to fire projectile from bot with 1-second rate limit."""
        if bot_id not in self.bot_data or not self._is_bot_alive(bot_id):
            return False

        bot_data = self.bot_data[bot_id]
        body = self.bot_bodies[bot_id]

        # Check 1-second fire rate limit
        time_since_last = self.time - bot_data.last_fire_time
        min_time_between = 1.0  # 1 second minimum between shots

        if time_since_last < min_time_between:
            return False

        # Check friendly fire gating (simplified - 1° cone check)
        if self._check_friendly_fire_risk(bot_id):
            return False

        # Create projectile at bot position with bot heading
        px, py = body.position
        heading = body.angle
        pvx = math.cos(heading) * self.PROJ_SPEED
        pvy = math.sin(heading) * self.PROJ_SPEED

        proj_id = self.next_projectile_id
        self.next_projectile_id += 1

        self._create_projectile(proj_id, px, py, pvx, pvy, bot_id, bot_data.team)

        # Track shot statistics
        bot_data.shots_fired += 1
        bot_data.last_fire_time = self.time

        self.events.append(
            {
                "type": "shot",
                "tick": self.tick,
                "bot_id": bot_id,
                "pos": [px, py],
                "heading": math.degrees(heading),
                "cooldown_remaining": 0.0,
                "total_shots": bot_data.shots_fired,
            }
        )
        return True

    def _update_projectile_ttl(self, dt: float):
        """Update projectile TTL and remove expired ones."""
        expired_projectiles = []

        for proj_id, proj_data in self.projectile_data.items():
            proj_data.ttl -= dt

            # Check if projectile expired or left arena
            body = self.projectile_bodies.get(proj_id)
            if body and (proj_data.ttl <= 0 or self._is_projectile_out_of_bounds(body)):
                expired_projectiles.append(proj_id)

        # Remove expired projectiles
        for proj_id in expired_projectiles:
            self._remove_projectile(proj_id)

    def _setup_collision_handlers(self):
        """Setup pymunk collision handlers."""

        # Projectile-Bot collision handler
        def projectile_bot_collision(arbiter, space, data):
            projectile_shape, bot_shape = arbiter.shapes
            proj_id = projectile_shape.body.user_data
            bot_id = bot_shape.body.user_data

            # Skip collision with shooter
            if proj_id in self.projectile_data and bot_id in self.bot_data:
                proj_data = self.projectile_data[proj_id]
                if proj_data.shooter_id == bot_id:
                    return False  # Ignore collision

                # Process hit
                self._process_projectile_hit(proj_id, bot_id)

            return True  # Allow collision to proceed

        # Register collision handler using correct pymunk API
        self.space.on_collision(
            self.COLLISION_TYPE_PROJECTILE,
            self.COLLISION_TYPE_BOT,
            begin=projectile_bot_collision,
        )

        # Projectile-Wall collision (removes projectile)
        def projectile_wall_collision(arbiter, space, data):
            projectile_shape, wall_shape = arbiter.shapes
            proj_id = projectile_shape.body.user_data
            if proj_id in self.projectile_data:
                self._remove_projectile(proj_id)
            return False  # Don't actually collide, just remove

        self.space.on_collision(
            self.COLLISION_TYPE_PROJECTILE,
            self.COLLISION_TYPE_WALL,
            begin=projectile_wall_collision,
        )

    def _check_friendly_fire_risk(self, bot_id: int) -> bool:
        """Check if firing would risk friendly fire (simplified)."""
        if bot_id not in self.bot_data:
            return True

        bot_body = self.bot_bodies[bot_id]
        bot_data = self.bot_data[bot_id]
        fire_angle = bot_body.angle
        cone_half_angle = math.radians(0.5)

        for other_id, other_data in self.bot_data.items():
            if (
                other_id == bot_id
                or not self._is_bot_alive(other_id)
                or other_data.team != bot_data.team
            ):
                continue

            # Check if teammate is in firing cone
            other_body = self.bot_bodies[other_id]
            dx = other_body.position[0] - bot_body.position[0]
            dy = other_body.position[1] - bot_body.position[1]
            angle_to_teammate = math.atan2(dy, dx)

            angle_diff = abs(self._normalize_angle(angle_to_teammate - fire_angle))
            if angle_diff <= cone_half_angle:
                return True

        return False

    def _resolve_target_bearing(self, bot_id: int, target_bearing: float) -> float:
        """Convert absolute bearing to target heading with predictive aiming."""
        # target_bearing is already in absolute degrees from observation system
        # This now includes predictive aiming from the LLM observation system
        return math.radians(target_bearing)

    def _calculate_intercept_heading(
        self,
        bot_id: int,
        target_x: float,
        target_y: float,
        target_vx: float,
        target_vy: float,
    ) -> float:
        """Calculate heading to intercept a moving target with projectile."""
        # Get bot position and projectile speed
        bx, by = self.bot_bodies[bot_id].position
        proj_speed = self.PROJ_SPEED

        # Relative position and velocity
        dx = target_x - bx
        dy = target_y - by
        dvx = target_vx  # projectile has no initial velocity relative to target
        dvy = target_vy

        # Solve intercept equation: |proj_pos + proj_vel*t| = |target_pos + target_vel*t|
        # This gives us the time when projectile and target meet

        # Quadratic equation coefficients for intercept time
        a = dvx * dvx + dvy * dvy - proj_speed * proj_speed
        b = 2 * (dx * dvx + dy * dvy)
        c = dx * dx + dy * dy

        # If target is stationary or we can't intercept, aim at current position
        if abs(a) < 1e-6:
            # Linear case or stationary target
            if abs(b) < 1e-6:
                # Stationary target
                intercept_time = 0
            else:
                intercept_time = -c / b
        else:
            # Quadratic case
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                # No intercept possible, aim at current position
                intercept_time = 0
            else:
                # Choose the positive root (future intercept)
                t1 = (-b + math.sqrt(discriminant)) / (2 * a)
                t2 = (-b - math.sqrt(discriminant)) / (2 * a)
                intercept_time = t1 if t1 > 0 else (t2 if t2 > 0 else 0)

        # Limit intercept time to reasonable values
        intercept_time = max(0, min(intercept_time, 3.0))  # Max 3 seconds prediction

        # Calculate intercept position
        intercept_x = target_x + target_vx * intercept_time
        intercept_y = target_y + target_vy * intercept_time

        # Calculate heading to intercept point
        aim_dx = intercept_x - bx
        aim_dy = intercept_y - by

        # Convert to our coordinate system (0° = North)
        return math.atan2(aim_dy, aim_dx)

    def _create_arena_walls(self):
        """Create static walls around the arena and interior obstacles."""
        width, height = self.ARENA_SIZE
        thickness = self.WALL_THICKNESS

        # Wall definitions: (center_x, center_y, width, height, angle_deg)
        # Angle is 0 for horizontal walls aligned with X-axis.
        self.walls = [
            # Perimeter walls
            (0, height / 2, width + thickness, thickness, 0),  # Top
            (0, -height / 2, width + thickness, thickness, 0),  # Bottom
            (-width / 2, 0, height, thickness, 90),  # Left
            (width / 2, 0, height, thickness, 90),  # Right
            # Interior walls
            (-width * 0.25, height * 0.2, 10, thickness, 0),
            (0, -height * 0.15, 8, thickness, 90),
            (-width * 0.25, -height * 0.3, 9, thickness, 20),
            (width * 0.3, height * 0.1, 6, thickness, 90),
        ]

        self.wall_bodies = []
        print("Creating arena walls:")
        for center_x, center_y, w, h, angle_deg in self.walls:
            print(
                f"  Creating wall: center=({center_x:.1f}, {center_y:.1f}), size=({w:.1f}, {h:.1f}), angle={angle_deg}°"
            )
            wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            wall_body.position = center_x, center_y
            wall_body.angle = math.radians(angle_deg)

            # Pymunk polys are defined relative to body center of gravity
            half_w, half_h = w / 2, h / 2
            verts = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]

            wall_shape = pymunk.Poly(wall_body, verts)
            wall_shape.friction = 0.1
            wall_shape.collision_type = self.COLLISION_TYPE_WALL
            self.space.add(wall_body, wall_shape)
            self.wall_bodies.append((wall_body, wall_shape))

    def _create_bot(self, bot_id: int, x: float, y: float, theta: float, team: int):
        """Create a bot with pymunk physics body."""
        # Create bot data
        self.bot_data[bot_id] = BotData(
            team=team,
            hp=float(self.MAX_HP),
            fire_command=False,
            last_fire_time=-999.0,
            shots_fired=0,
            shots_hit=0,
            damage_dealt=0.0,
            damage_taken=0.0,
            kills=0,
            deaths=0,
        )

        # Create pymunk body
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, self.BOT_RADIUS)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.angle = theta
        body.user_data = bot_id

        # Create shape
        shape = pymunk.Circle(body, self.BOT_RADIUS)
        shape.friction = 0.1
        shape.collision_type = self.COLLISION_TYPE_BOT

        # Add to space
        self.space.add(body, shape)
        self.bot_bodies[bot_id] = body

    def _create_projectile(
        self,
        proj_id: int,
        x: float,
        y: float,
        vx: float,
        vy: float,
        shooter_id: int,
        team: int,
    ):
        """Create a projectile with pymunk physics body."""
        # Create projectile data
        self.projectile_data[proj_id] = ProjectileData(
            shooter_id=shooter_id, team=team, birth_tick=self.tick, ttl=self.PROJ_TTL
        )

        # Create pymunk body
        mass = 0.01  # Very light projectile
        radius = 0.05  # Small projectile
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.velocity = vx, vy
        body.user_data = proj_id

        # Create shape
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.0
        shape.collision_type = self.COLLISION_TYPE_PROJECTILE

        # Add to space
        self.space.add(body, shape)
        self.projectile_bodies[proj_id] = body

    def _remove_projectile(self, proj_id: int):
        """Remove projectile from simulation."""
        if proj_id in self.projectile_bodies:
            body = self.projectile_bodies[proj_id]
            self.space.remove(body, *body.shapes)
            del self.projectile_bodies[proj_id]

        if proj_id in self.projectile_data:
            del self.projectile_data[proj_id]

    def _is_bot_alive(self, bot_id: int) -> bool:
        """Check if bot is alive."""
        return bot_id in self.bot_data and self.bot_data[bot_id].hp > 0

    def _is_projectile_out_of_bounds(self, body: pymunk.Body) -> bool:
        """Check if projectile is outside arena bounds."""
        x, y = body.position
        W, H = self.ARENA_SIZE
        return not (-W / 2 <= x <= W / 2 and -H / 2 <= y <= H / 2)

    def _process_projectile_hit(self, proj_id: int, bot_id: int):
        """Process projectile hitting a bot."""
        if proj_id not in self.projectile_data or bot_id not in self.bot_data:
            return

        proj_data = self.projectile_data[proj_id]
        bot_data = self.bot_data[bot_id]
        shooter_data = self.bot_data.get(proj_data.shooter_id)

        if not shooter_data:
            return

        # Apply damage
        damage = self.PROJ_DAMAGE
        bot_data.hp -= damage
        bot_data.damage_taken += damage
        shooter_data.damage_dealt += damage
        shooter_data.shots_hit += 1

        # Get positions for event logging
        proj_body = self.projectile_bodies[proj_id]
        bot_body = self.bot_bodies[bot_id]

        self.events.append(
            {
                "type": "hit",
                "tick": self.tick,
                "projectile_shooter": proj_data.shooter_id,
                "target": bot_id,
                "damage": damage,
                "pos": list(proj_body.position),
                "shooter_accuracy": shooter_data.shots_hit
                / max(1, shooter_data.shots_fired),
            }
        )

        # Check for death
        if bot_data.hp <= 0:
            bot_data.deaths += 1
            shooter_data.kills += 1

            self.events.append(
                {
                    "type": "death",
                    "tick": self.tick,
                    "bot_id": bot_id,
                    "killer_id": proj_data.shooter_id,
                    "pos": list(bot_body.position),
                    "killer_kills": shooter_data.kills,
                }
            )

        # Remove projectile
        self._remove_projectile(proj_id)

    def _apply_velocity_constraints(self):
        """Apply velocity damping and speed limits to bots."""
        for bot_id, body in self.bot_bodies.items():
            if not self._is_bot_alive(bot_id):
                body.velocity = (0, 0)
                body.angular_velocity = 0
                continue

            # Apply linear damping
            vx, vy = body.velocity
            vx *= 1.0 - self.LINEAR_DAMPING * self.DT_PHYSICS
            vy *= 1.0 - self.LINEAR_DAMPING * self.DT_PHYSICS

            # Clamp to max speed
            speed = math.sqrt(vx * vx + vy * vy)
            if speed > self.V_MAX:
                scale = self.V_MAX / speed
                vx *= scale
                vy *= scale

            body.velocity = (vx, vy)

    def get_battle_state(self) -> Dict:
        """Get current battle state for logging/visualization."""
        bots = []
        for bot_id, bot_data in self.bot_data.items():
            if self._is_bot_alive(bot_id):
                body = self.bot_bodies[bot_id]
                bots.append(
                    {
                        "id": bot_id,
                        "x": round(body.position[0], 2),
                        "y": round(body.position[1], 2),
                        "theta": round(math.degrees(body.angle), 1),
                        "vx": round(body.velocity[0], 2),
                        "vy": round(body.velocity[1], 2),
                        "hp": int(bot_data.hp),
                        "alive": self._is_bot_alive(bot_id),
                        "team": bot_data.team,
                    }
                )

        projectiles = []
        for proj_id, proj_data in self.projectile_data.items():
            if proj_id in self.projectile_bodies:
                body = self.projectile_bodies[proj_id]
                projectiles.append(
                    {
                        "id": proj_id,
                        "x": round(body.position[0], 2),
                        "y": round(body.position[1], 2),
                        "vx": round(body.velocity[0], 2),
                        "vy": round(body.velocity[1], 2),
                        "team": proj_data.team,
                        "age": self.tick - proj_data.birth_tick,
                    }
                )

        return {
            "tick": self.tick,
            "time": round(self.time, 3),
            "bots": bots,
            "projectiles": projectiles,
            "events": self.events.copy(),
        }

    def log_state(self):
        """Log current state to battle log."""
        self.battle_log.append(self.get_battle_state())
        self.events.clear()  # Clear events after logging

    def get_walls(self) -> List[Tuple]:
        """Get wall data for logging/visualization."""
        return self.walls

    def is_battle_over(
        self, max_duration: float = 60.0
    ) -> Tuple[bool, Optional[int], str]:
        """Check if battle is over. Returns (is_over, winning_team, reason)."""
        team0_alive = sum(
            1
            for bot_id, bot_data in self.bot_data.items()
            if bot_data.team == 0 and self._is_bot_alive(bot_id)
        )
        team1_alive = sum(
            1
            for bot_id, bot_data in self.bot_data.items()
            if bot_data.team == 1 and self._is_bot_alive(bot_id)
        )

        if team0_alive == 0:
            return True, 1, "elimination"
        elif team1_alive == 0:
            return True, 0, "elimination"
        elif self.time > max_duration:
            # Determine winner by total HP
            team0_hp = sum(
                bot_data.hp for bot_data in self.bot_data.values() if bot_data.team == 0
            )
            team1_hp = sum(
                bot_data.hp for bot_data in self.bot_data.values() if bot_data.team == 1
            )
            winner = 0 if team0_hp > team1_hp else 1
            return True, winner, "timeout"

        return False, None, ""

    def calculate_bot_scores(self) -> List[Dict]:
        """Calculate comprehensive scores for each bot."""
        scores = []

        for bot_id, bot_data in self.bot_data.items():
            # Basic stats
            shots = bot_data.shots_fired
            hits = bot_data.shots_hit
            damage_dealt = bot_data.damage_dealt
            damage_taken = bot_data.damage_taken
            kills = bot_data.kills
            deaths = bot_data.deaths

            # Calculate metrics
            hit_rate = hits / max(1, shots)  # Accuracy percentage
            survival_rate = 1.0 if deaths == 0 else 0.0
            damage_efficiency = damage_dealt / max(1, damage_taken)  # Damage ratio

            # Scoring formula
            # Base scores
            accuracy_score = hit_rate * 100  # 0-100 points for perfect accuracy
            damage_score = damage_dealt * 0.4  # 0.4 points per HP dealt
            kill_score = kills * 50  # 50 points per kill
            survival_bonus = survival_rate * 25  # 25 bonus for staying alive
            damage_penalty = damage_taken * -0.2  # -0.2 points per HP taken

            # Overall score
            total_score = (
                accuracy_score
                + damage_score
                + kill_score
                + survival_bonus
                + damage_penalty
            )

            scores.append(
                {
                    "bot_id": bot_id,
                    "team": self.bot_data[bot_id].team,
                    "total_score": round(total_score, 1),
                    # Performance metrics
                    "hit_rate": round(hit_rate, 3),
                    "damage_efficiency": round(damage_efficiency, 2),
                    "survival_rate": survival_rate,
                    # Raw stats
                    "shots_fired": shots,
                    "shots_hit": hits,
                    "damage_dealt": round(damage_dealt, 1),
                    "damage_taken": round(damage_taken, 1),
                    "kills": kills,
                    "deaths": deaths,
                    # Score breakdown
                    "score_breakdown": {
                        "accuracy": round(accuracy_score, 1),
                        "damage": round(damage_score, 1),
                        "kills": round(kill_score, 1),
                        "survival": round(survival_bonus, 1),
                        "damage_penalty": round(damage_penalty, 1),
                    },
                }
            )

        return scores

    def calculate_team_scores(self) -> Dict:
        """Calculate team-level scores and rankings."""
        bot_scores = self.calculate_bot_scores()

        team_stats = {
            0: {"bots": [], "total_score": 0},
            1: {"bots": [], "total_score": 0},
        }

        for score in bot_scores:
            team = score["team"]
            team_stats[team]["bots"].append(score)
            team_stats[team]["total_score"] += score["total_score"]

        # Calculate team aggregates
        for team in [0, 1]:
            bots = team_stats[team]["bots"]
            if bots:
                team_stats[team].update(
                    {
                        "avg_hit_rate": sum(b["hit_rate"] for b in bots) / len(bots),
                        "total_kills": sum(b["kills"] for b in bots),
                        "total_deaths": sum(b["deaths"] for b in bots),
                        "total_damage_dealt": sum(b["damage_dealt"] for b in bots),
                        "total_damage_taken": sum(b["damage_taken"] for b in bots),
                        "bots_alive": sum(1 for b in bots if b["deaths"] == 0),
                    }
                )

        return team_stats

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def cleanup(self):
        """Clean up pymunk space to avoid CFFI callback issues."""
        try:
            # Remove all projectiles first
            for proj_id in list(self.projectile_bodies.keys()):
                self._remove_projectile(proj_id)

            # Remove all bot bodies
            for bot_id, body in list(self.bot_bodies.items()):
                try:
                    self.space.remove(body, *body.shapes)
                except:
                    pass  # Ignore removal errors

            # Remove wall bodies
            if hasattr(self, "wall_bodies"):
                for wall_body, wall_shape in self.wall_bodies:
                    try:
                        self.space.remove(wall_body, wall_shape)
                    except:
                        pass  # Ignore removal errors

            # Clear the space without using iteration callbacks
            self.space = None
        except:
            pass  # Ignore all cleanup errors when profiling


def run_battle_simulation(
    seed: int = 42,
    max_duration: float = 60.0,
    spawn_config: Dict = None,
    bot_programs: Dict = None,
    verbose: bool = True,
    arena_size: Optional[Tuple[float, float]] = None,
    bots_per_side: Optional[int] = None,
) -> Dict:
    """Run a complete battle simulation with configurable parameters.

    Args:
        seed: Random seed for deterministic simulation
        max_duration: Maximum battle duration in seconds
        spawn_config: Bot spawning configuration
        bot_programs: Custom bot program assignments
        verbose: Whether to print progress
    """
    from dsl_parser import DSLParser, DSLExecutor
    from llm import LLMController

    arena = Arena(seed, spawn_config, arena_size, bots_per_side)
    dsl_parser = DSLParser()
    dsl_executor = DSLExecutor()
    llm = LLMController(arena.BOT_COUNT)

    # Apply custom bot programs if provided
    if bot_programs:
        llm.bot_programs.update(bot_programs)

    if verbose:
        bots_per_side_actual = arena.BOTS_PER_SIDE
        print(
            f"Starting {bots_per_side_actual}v{bots_per_side_actual} battle simulation (seed={seed})"
        )
    start_time = time.time()

    # Main simulation loop
    while True:
        # Physics step (240Hz)
        arena.step_physics()

        # Control step (120Hz)
        if arena.physics_tick % 2 == 0:  # 240Hz -> 120Hz
            arena.step_control()

            # Execute DSL for each bot
            for bot_id in range(arena.BOT_COUNT):
                if not arena._is_bot_alive(bot_id):
                    continue

                # Get observation for this bot
                obs = llm.get_observation(arena, bot_id)

                # Get DSL program text from LLM (mock for now)
                program_text = llm.get_program_text(bot_id)

                # Parse and execute DSL program
                rules = dsl_parser.parse_program(program_text)
                winning_action = dsl_executor.execute_rules(rules, obs, bot_id)

                # Apply single winning action to bot
                arena.set_single_bot_action(bot_id, winning_action)

            # Handle firing for all bots (after all commands are processed)
            for bot_id in range(arena.BOT_COUNT):
                if (
                    arena._is_bot_alive(bot_id)
                    and hasattr(arena, "fire_command")
                    and arena.fire_command[bot_id]
                ):
                    arena.try_fire_projectile(bot_id)

            # Log state every few ticks for visualization
            if arena.tick % 12 == 0:  # ~10Hz logging
                arena.log_state()

        # Check for battle end
        is_over, winner, reason = arena.is_battle_over(max_duration)
        if is_over:
            arena.log_state()  # Final state
            break

    elapsed = time.time() - start_time
    if verbose:
        print(
            f"Battle complete: Team {winner} wins by {reason} ({arena.time:.1f}s simulated, {elapsed:.2f}s real)"
        )

    # Generate battle data before cleanup
    battle_data = {
        "metadata": {
            "seed": seed,
            "duration": round(arena.time, 2),
            "winner": f"team_{winner}",
            "reason": reason,
            "arena_size": arena.ARENA_SIZE,
            "total_ticks": arena.tick,
            "real_time": round(elapsed, 2),
        },
        "timeline": arena.battle_log,
        "summary": _generate_battle_summary(arena),
    }

    # Clean up pymunk space to avoid CFFI callback errors
    arena.cleanup()

    return battle_data


def _generate_battle_summary(arena: Arena) -> Dict:
    """Generate comprehensive battle summary with detailed scoring."""
    # Get detailed scoring data
    bot_scores = arena.calculate_bot_scores()
    team_scores = arena.calculate_team_scores()

    # Count events by type
    all_events = []
    for state in arena.battle_log:
        all_events.extend(state.get("events", []))

    total_shots = sum(bot_data.shots_fired for bot_data in arena.bot_data.values())
    total_hits = sum(bot_data.shots_hit for bot_data in arena.bot_data.values())
    total_deaths = len([e for e in all_events if e["type"] == "death"])

    # Determine MVP (highest scoring bot)
    mvp = max(bot_scores, key=lambda x: x["total_score"]) if bot_scores else None

    # Determine best and worst performers by category
    best_accuracy = max(bot_scores, key=lambda x: x["hit_rate"]) if bot_scores else None
    most_damage = (
        max(bot_scores, key=lambda x: x["damage_dealt"]) if bot_scores else None
    )
    most_kills = max(bot_scores, key=lambda x: x["kills"]) if bot_scores else None

    # Get final positions and HP
    final_hp = []
    final_positions = []
    for bot_id in range(arena.BOT_COUNT):
        if bot_id in arena.bot_data:
            bot_data = arena.bot_data[bot_id]
            body = arena.bot_bodies[bot_id]
            is_alive = arena._is_bot_alive(bot_id)
            final_hp.append(int(bot_data.hp) if is_alive else 0)
            final_positions.append(
                [round(body.position[0], 1), round(body.position[1], 1)]
                if is_alive
                else None
            )
        else:
            final_hp.append(0)
            final_positions.append(None)

    return {
        # Legacy stats for compatibility
        "total_shots": total_shots,
        "total_hits": total_hits,
        "total_deaths": total_deaths,
        "hit_rate": round(total_hits / max(total_shots, 1), 3),
        "final_hp": final_hp,
        "final_positions": final_positions,
        # Detailed scoring system
        "bot_scores": bot_scores,
        "team_scores": team_scores,
        # Performance highlights
        "mvp": {
            "bot_id": mvp["bot_id"] if mvp else None,
            "score": mvp["total_score"] if mvp else 0,
            "team": mvp["team"] if mvp else None,
        },
        "best_accuracy": {
            "bot_id": best_accuracy["bot_id"] if best_accuracy else None,
            "hit_rate": best_accuracy["hit_rate"] if best_accuracy else 0,
            "shots": f"{best_accuracy['shots_hit']}/{best_accuracy['shots_fired']}"
            if best_accuracy
            else "0/0",
        },
        "most_damage": {
            "bot_id": most_damage["bot_id"] if most_damage else None,
            "damage": most_damage["damage_dealt"] if most_damage else 0,
        },
        "most_kills": {
            "bot_id": most_kills["bot_id"] if most_kills else None,
            "kills": most_kills["kills"] if most_kills else 0,
        },
        # Battle statistics
        "battle_intensity": round(
            total_shots / max(arena.time, 1), 1
        ),  # shots per second
        "lethality": round(total_deaths / max(arena.BOT_COUNT, 1), 2),  # death rate
        "overall_accuracy": round(total_hits / max(total_shots, 1), 3),
    }
