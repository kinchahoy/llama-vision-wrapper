"""
pygfx Battle Viewer
Interactive 3D visualization of battle simulations using the pygfx engine.
"""

import asyncio
import json
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

# Import visibility system to use same logic as 2D viewer
try:
    from python_llm import PythonLLMController
    from battle_sim.arena import Arena
except ImportError:
    PythonLLMController = None
    Arena = None


class Battle3DViewer:
    """Interactive 3D viewer for battle simulations using pygfx."""

    def __init__(self, battle_data: Dict, canvas: WgpuCanvas):
        """Initialize the pygfx viewer."""
        self.canvas = canvas
        self.renderer = gfx.WgpuRenderer(canvas)
        self.scene = gfx.Scene()

        self.battle_data = battle_data
        self.timeline = battle_data["timeline"]
        self.metadata = battle_data["metadata"]

        # Playback state
        self.current_frame = 0.0
        self.playing = False
        self.playback_speed = 1.0
        self.last_update_time = time.time()

        # Arena dimensions
        arena_size = self.metadata.get("arena_size", [20, 20])
        self.arena_width, self.arena_height = arena_size

        # UI state
        self.show_fov = False
        self.selected_bot = None
        self.selected_bot_info: Optional[Dict] = (
            None  # Store more details about selected bot
        )

        # Initialize visibility system (same as 2D viewer)
        if PythonLLMController is not None:
            self.llm_controller = PythonLLMController()
        else:
            self.llm_controller = None

        # pygfx object management
        self.bot_objects = {}  # {bot_id: {'body': world_object, 'healthbar':...}}
        self.projectile_objects = {}
        self.fov_object = None

        # Signal descriptions for UI
        self.signal_descriptions = {
            "none": "No specific status",
            "ready": "Ready for action",
            "low_hp": "Health critical (<30 HP)",
            "reloading": "Weapon cooling down",
            "attacking": "Engaging enemy target",
            "firing": "Currently shooting",
            "flanking": "Moving to flank enemy",
            "retreating": "Falling back from combat",
            "advancing": "Moving forward to engage",
            "cover_fire": "Providing suppressive fire",
            "need_backup": "Requesting immediate assistance",
            "enemy_spotted": "Enemy contact established",
            "holding_position": "Maintaining current location",
            "moving_to_cover": "Relocating for protection",
            "watching_flank": "Covering team's side/rear",
            "regrouping": "Moving to rally point",
            "follow_me": "Request team to follow",
            "wait": "Hold current position",
            "go_go_go": "Execute coordinated advance",
            "spread_out": "Increase team dispersion",
            "focus_fire": "Concentrate fire on target",
            "disengage": "Break contact and withdraw",
        }

        # Setup
        self._setup_scene()
        self._setup_ui()
        self._setup_controls()

    def _setup_scene(self):
        """Set up the 3D scene, camera, and lighting."""
        self.scene.add(gfx.Background(None, gfx.BackgroundMaterial("#222", "#444")))

        # Camera
        self.camera = gfx.PerspectiveCamera(70, 16 / 9)
        arena_diagonal = math.sqrt(self.arena_width**2 + self.arena_height**2)
        self.camera.position.set(0, -arena_diagonal * 0.9, arena_diagonal * 0.9)
        self.camera.look_at(gfx.linalg.Vector3(0, 0, 0))

        # Lighting
        self.scene.add(gfx.AmbientLight(intensity=0.4))
        sun = gfx.DirectionalLight(intensity=0.8)
        sun.position.set(-1, -2, 3)
        self.scene.add(sun)

        fill_light = gfx.DirectionalLight(intensity=0.4)
        fill_light.position.set(1, 2, 1)
        self.scene.add(fill_light)

        # Arena floor
        floor_geom = gfx.plane_geometry(self.arena_width, self.arena_height)
        floor_mat = gfx.MeshStandardMaterial(
            color=(0.2, 0.2, 0.25), roughness=0.05, metalness=1.0
        )
        self.floor = gfx.Mesh(floor_geom, floor_mat)
        self.scene.add(self.floor)

        self._create_walls()

    def _create_walls(self):
        """Create interior and perimeter walls from metadata."""
        wall_3d_height = 2.0
        walls_data = self.metadata.get("walls", [])

        for i, wall_def in enumerate(walls_data):
            center_x, center_y, width, height, angle_deg = wall_def

            wall_geom = gfx.box_geometry(width, height, wall_3d_height)

            if i < 4:  # Perimeter walls
                wall_mat = gfx.MeshStandardMaterial(
                    color=(0.4, 0.4, 0.45), roughness=0.2, metalness=0.8
                )
            else:  # Interior walls
                wall_mat = gfx.MeshStandardMaterial(
                    color=(0.6, 0.8, 0.9), roughness=0.1, metalness=1.0
                )

            wall = gfx.Mesh(wall_geom, wall_mat)
            wall.position.set(center_x, center_y, wall_3d_height / 2)
            wall.rotation.set_from_euler(
                gfx.linalg.Euler(0, 0, math.radians(angle_deg))
            )
            self.scene.add(wall)

    def _setup_ui(self):
        """Set up text elements for UI."""
        self.ui_scene = gfx.Scene()
        self.ui_camera = gfx.ScreenCoordsCamera()

        self.info_text = self._create_text_object("", (10, 10))
        self.bot_info_text = self._create_text_object(
            "Click on a bot to select it", (10, 150)
        )
        self.tactical_info_text = self._create_text_object("", (10, 350))
        self.events_text = self._create_text_object(
            "", (0, 10)
        )  # Positioned dynamically
        self.ui_scene.add(
            self.info_text,
            self.bot_info_text,
            self.tactical_info_text,
            self.events_text,
        )

    def _create_text_object(
        self, text, position, font_size=14, color="#FFF", anchor="top-left"
    ):
        text_geom = gfx.TextGeometry(text, font_size=font_size, anchor=anchor)
        text_mat = gfx.TextMaterial(color=color)
        text_obj = gfx.Text(text_geom, text_mat)
        text_obj.local.position = position
        return text_obj

    def _setup_controls(self):
        """Set up camera controls and keyboard handlers."""
        self.camera_controls = gfx.OrbitController(
            self.camera, register_events=self.renderer
        )
        self.canvas.add_event_handler(self.handle_event, "key_down", "pointer_down")

    def handle_event(self, event):
        """Handle keyboard and mouse events."""
        if event.type == "key_down":
            self._handle_keypress(event.key)
        elif event.type == "pointer_down" and event.button == 1:  # Left click
            self._handle_mouse_click(event)

    def _handle_keypress(self, key: str):
        if key.lower() == "q" or key == "Escape":
            sys.exit()
        elif key == " ":
            self._toggle_play()
        elif key == "r":
            self._reset_sim()
        elif key == "ArrowLeft":
            self._step_frame(-1)
        elif key == "ArrowRight":
            self._step_frame(1)
        elif key == "c":
            self._reset_camera_view()
        elif key == "f":
            self._toggle_fov()
        elif key == "=" or key == "+":
            self._change_playback_speed(1)
        elif key == "-":
            self._change_playback_speed(-1)

    def _handle_mouse_click(self, event):
        pick_info = self.scene.get_pick_info((event.x, event.y))
        if pick_info and hasattr(pick_info["world_object"], "bot_id"):
            bot_id = pick_info["world_object"].bot_id
            current_state = self._get_current_state()
            for bot in current_state.get("bots", []):
                if bot["id"] == bot_id:
                    self.selected_bot = bot
                    self.selected_bot_info = self._get_detailed_bot_info(bot_id)
                    self.playing = False
                    self._update_fov_display()
                    break

    def _get_current_state(self) -> Dict:
        """Get the state for the current frame, with interpolation for smooth animation."""
        if not self.timeline:
            return {}

        frame_idx = self.current_frame
        if frame_idx < 0:
            return self.timeline[0]
        if frame_idx >= len(self.timeline) - 1:
            return self.timeline[-1]

        frame_before = int(frame_idx)
        frame_after = frame_before + 1
        alpha = frame_idx - frame_before

        state_before = self.timeline[frame_before]
        state_after = self.timeline[frame_after]

        interpolated_state = {
            "time": state_before.get("time", 0) * (1 - alpha)
            + state_after.get("time", 0) * alpha,
            "bots": [],
            "projectiles": [],
            "events": state_before.get("events", [])
            + state_after.get("events", []),  # Simplified event handling
        }

        # Interpolate bot positions and orientations
        for bot_before in state_before.get("bots", []):
            bot_id = bot_before["id"]
            bot_after = next(
                (b for b in state_after.get("bots", []) if b["id"] == bot_id), None
            )

            if bot_after:
                # Interpolate position
                pos_before = (bot_before["x"], bot_before["y"])
                pos_after = (bot_after["x"], bot_after["y"])
                interpolated_pos = (
                    pos_before[0] * (1 - alpha) + pos_after[0] * alpha,
                    pos_before[1] * (1 - alpha) + pos_after[1] * alpha,
                )

                # Interpolate theta (angle) - handle wrapping correctly
                theta_before = bot_before.get("theta", 0.0)
                theta_after = bot_after.get("theta", 0.0)
                delta_theta = (
                    theta_after - theta_before + 180
                ) % 360 - 180  # Handle angle wrapping
                interpolated_theta = theta_before + delta_theta * alpha
                interpolated_theta = (interpolated_theta + 360) % 360  # Ensure positive

                interpolated_state["bots"].append(
                    {
                        "id": bot_id,
                        "team": bot_before["team"],
                        "x": interpolated_pos[0],
                        "y": interpolated_pos[1],
                        "theta": interpolated_theta,
                        "hp": bot_before.get("hp", 100) * (1 - alpha)
                        + bot_after.get("hp", 100) * alpha,  # Interpolate health
                        "alive": bot_before.get("alive", True)
                        or bot_after.get("alive", True),  # Simplified alive
                        "signal": bot_before.get(
                            "signal", "none"
                        ),  # No interpolation for signals for now
                    }
                )
            else:
                # If bot exists in before but not after, use before state (could be a death)
                interpolated_state["bots"].append(bot_before)

        # Interpolate projectiles
        for proj_before in state_before.get("projectiles", []):
            proj_id = proj_before["id"]
            proj_after = next(
                (p for p in state_after.get("projectiles", []) if p["id"] == proj_id),
                None,
            )

            if proj_after:
                pos_before = (proj_before["x"], proj_before["y"])
                pos_after = (proj_after["x"], proj_after["y"])
                interpolated_pos = (
                    pos_before[0] * (1 - alpha) + pos_after[0] * alpha,
                    pos_before[1] * (1 - alpha) + pos_after[1] * alpha,
                )
                interpolated_state["projectiles"].append(
                    {
                        "id": proj_id,
                        "x": interpolated_pos[0],
                        "y": interpolated_pos[1],
                        "team": proj_before.get("team", 0),
                    }
                )
            else:
                interpolated_state["projectiles"].append(proj_before)

        return interpolated_state

    def _get_detailed_bot_info(self, bot_id: int) -> Optional[Dict]:
        """Retrieves detailed information about a specific bot,
        including visibility data."""
        current_state = self._get_current_state()
        bot = next(
            (b for b in current_state.get("bots", []) if b["id"] == bot_id), None
        )
        if not bot:
            return None

        bot_x, bot_y = bot["x"], bot["y"]
        arena_size = self.metadata.get("arena_size", [20, 20])
        arena = Arena(arena_size[0], arena_size[1], self.metadata.get("walls", []))

        visible_objects = self._get_visible_objects_for_bot(
            bot_x, bot_y, bot.get("theta", 0.0), bot["team"], current_state, arena
        )
        return {
            "bot": bot,
            "visible_objects": visible_objects,
        }

    def _get_visible_objects_for_bot(
        self,
        x: float,
        y: float,
        theta: float,
        team: int,
        state: Dict,
        arena: Arena,
    ) -> Dict:
        """
        Determine which bots, projectiles, and walls are visible to a given bot.
        """

        visible_bots = self._get_visible_bots(x, y, theta, team, state, arena)
        visible_projectiles = self._get_nearby_projectiles(x, y, theta, state, arena)
        visible_walls = self._get_visible_walls(x, y, theta, arena)
        return {
            "bots": visible_bots,
            "projectiles": visible_projectiles,
            "walls": visible_walls,
        }

    def _get_visible_bots(
        self,
        x: float,
        y: float,
        theta: float,
        team: int,
        state: Dict,
        arena: Arena,
    ) -> List[Dict]:
        """
        Determine which bots are visible to a given bot.
        """
        visible_bots = []
        for bot in state.get("bots", []):
            if bot["team"] == team or not bot["alive"]:
                continue
            if self._is_in_fov(x, y, theta, bot["x"], bot["y"], arena):
                visible_bots.append(bot)
        return visible_bots

    def _get_nearby_projectiles(
        self,
        x: float,
        y: float,
        theta: float,
        state: Dict,
        arena: Arena,
    ) -> List[Dict]:
        """
        Determine which projectiles are nearby a given bot.
        """
        nearby_projectiles = []
        for projectile in state.get("projectiles", []):
            if self._is_in_range(x, y, projectile["x"], projectile["y"], 5.0):
                nearby_projectiles.append(projectile)
        return nearby_projectiles

    def _get_visible_walls(
        self,
        x: float,
        y: float,
        theta: float,
        arena: Arena,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Determine which walls are visible to a given bot.
        """
        visible_walls = []
        for wall in arena.walls:
            if self._is_in_fov(x, y, theta, wall[0], wall[1], arena):
                visible_walls.append(wall)
        return visible_walls

    def _is_in_fov(
        self,
        x: float,
        y: float,
        theta: float,
        target_x: float,
        target_y: float,
        arena: Arena,
        fov_angle: float = 120.0,
        max_range: float = 15.0,
    ) -> bool:
        """
        Check if a target is within the field of view (FOV) of a bot.
        """
        dx = target_x - x
        dy = target_y - y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > max_range:
            return False

        # Convert theta to radians and normalize to 0-360
        theta_rad = math.radians(theta)
        theta_norm = (theta + 360) % 360

        # Calculate angle to target in radians
        angle_to_target_rad = math.atan2(dy, dx)
        angle_to_target_deg = math.degrees(angle_to_target_rad)
        angle_to_target_norm = (angle_to_target_deg + 360) % 360

        # Calculate FOV bounds
        fov_half_angle = fov_angle / 2
        fov_min = (theta_norm - fov_half_angle + 360) % 360
        fov_max = (theta_norm + fov_half_angle) % 360

        # Check if target is within FOV, handling angle wrapping
        if fov_min < fov_max:
            return fov_min <= angle_to_target_norm <= fov_max
        else:
            return angle_to_target_norm >= fov_min or angle_to_target_norm <= fov_max

    def _is_in_range(
        self, x1: float, y1: float, x2: float, y2: float, max_range: float
    ) -> bool:
        """Check if a point (x2, y2) is within a certain range of (x1, y1)."""
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance <= max_range

    def _update_bots(self, state: Dict):
        """Update bot models in the scene."""
        current_bot_ids = {bot["id"] for bot in state.get("bots", []) if bot["alive"]}

        # Remove dead bots
        for bot_id in list(self.bot_objects.keys()):
            if bot_id not in current_bot_ids:
                for obj in self.bot_objects[bot_id].values():
                    self.scene.remove(obj)
                del self.bot_objects[bot_id]

        # Update or create bots
        for bot in state.get("bots", []):
            if not bot["alive"]:
                continue

            bot_id = bot["id"]
            pos = (bot["x"], bot["y"], 0.5)

            if bot_id not in self.bot_objects:
                # Create bot body
                bot_geom = gfx.sphere_geometry(0.4)
                color = (0, 0.5, 1) if bot["team"] == 0 else (1, 0.3, 0.3)
                bot_mat = gfx.MeshStandardMaterial(color=color)
                bot_body = gfx.Mesh(bot_geom, bot_mat)
                bot_body.bot_id = bot_id  # For picking
                self.scene.add(bot_body)

                # Heading indicator
                heading_geom = gfx.cone_geometry(0.2, 0.4)
                heading_mat = gfx.MeshStandardMaterial(
                    color=tuple(c + 0.4 for c in color)
                )
                heading_cone = gfx.Mesh(heading_geom, heading_mat)
                heading_cone.position.set(0, 0.25, 0)  # In front of body
                heading_cone.rotation.set_from_euler(
                    gfx.linalg.Euler(math.pi / 2, 0, 0)
                )
                bot_body.add(heading_cone)

                # Health bar
                hb_bg_geom = gfx.plane_geometry(0.6, 0.06)
                hb_bg_mat = gfx.MeshBasicMaterial(color=(0.2, 0.2, 0.2))
                hb_bg = gfx.Mesh(hb_bg_geom, hb_bg_mat)
                hb_bg.position.set(0, 0, 0.6)
                bot_body.add(hb_bg)

                hb_fill_geom = gfx.plane_geometry(0.6, 0.06)
                hb_fill_mat = gfx.MeshBasicMaterial(color=(0, 1, 0))
                hb_fill = gfx.Mesh(hb_fill_geom, hb_fill_mat)
                hb_fill.position.set(0, 0, 0.001)  # Slightly above bg
                hb_bg.add(hb_fill)

                # Bot ID label
                id_text = gfx.Text(
                    gfx.TextGeometry(str(bot_id), font_size=10, anchor="center"),
                    gfx.TextMaterial(color="#FFF"),
                )
                id_text.position.set(0, 0, 1.0)
                bot_body.add(id_text)

                self.bot_objects[bot_id] = {
                    "body": bot_body,
                    "heading": heading_cone,
                    "hb_bg": hb_bg,
                    "hb_fill": hb_fill,
                    "id_label": id_text,
                }

            # Update bot state
            bot_obj_group = self.bot_objects[bot_id]
            bot_body = bot_obj_group["body"]
            bot_body.position.set(*pos)
            # Sim angle: 0 is +X. pygfx Z-up: rotation is around Z axis.
            bot_body.rotation.set_from_euler(
                gfx.linalg.Euler(0, 0, math.radians(bot["theta"]))
            )

            # Update health bar
            hp_ratio = max(0.0, min(1.0, bot["hp"] / 100.0))
            hb_fill = bot_obj_group["hb_fill"]
            hb_fill.scale.x = hp_ratio
            hb_fill.position.x = -0.3 * (1 - hp_ratio)
            if hp_ratio > 0.6:
                hp_color = (0, 1, 0)
            elif hp_ratio > 0.3:
                hp_color = (1, 1, 0)
            else:
                hp_color = (1, 0, 0)
            hb_fill.material.color = hp_color

    def _update_projectiles(self, state: Dict):
        """Update projectile models in the scene."""
        projectiles_in_state = {
            p["id"]: p for p in state.get("projectiles", []) if "id" in p
        }
        current_proj_ids = set(projectiles_in_state.keys())

        # Remove old projectiles
        for proj_id in list(self.projectile_objects.keys()):
            if proj_id not in current_proj_ids:
                self.scene.remove(self.projectile_objects[proj_id])
                del self.projectile_objects[proj_id]

        # Update/create projectiles
        for proj_id, proj in projectiles_in_state.items():
            pos = (proj["x"], proj["y"], 0.5)
            if proj_id not in self.projectile_objects:
                proj_geom = gfx.sphere_geometry(0.1)
                color = (0.2, 1, 1) if proj.get("team") == 0 else (1, 0.5, 1)
                proj_mat = gfx.MeshStandardMaterial(
                    color=color, emissive=color, emissive_intensity=0.8
                )
                proj_model = gfx.Mesh(proj_geom, proj_mat)
                self.projectile_objects[proj_id] = proj_model
                self.scene.add(proj_model)

            np = self.projectile_objects[proj_id]
            np.position.set(*pos)

    def _update_ui(self, state: Dict):
        """Update the text in the UI panels."""
        # General info
        time_info = f"Time: {state.get('time', 0):.1f}s"
        frame_info = f"Frame: {int(self.current_frame)}/{len(self.timeline) - 1}"
        speed_info = f"Speed: {self.playback_speed:.1f}x"
        winner_info = f"Winner: {self.metadata.get('winner', 'N/A')}"
        self.info_text.geometry.set_text(
            f"{time_info}\n{frame_info}\n{speed_info}\n{winner_info}"
        )

        # Selected bot info
        bot_info_lines = []
        if self.selected_bot_info:
            bot = self.selected_bot_info["bot"]
            visible_objects = self.selected_bot_info["visible_objects"]

            bot_info_lines.append(f"Bot {bot['id']} (Team {bot['team']})")
            bot_info_lines.append(f"HP: {bot['hp']:.0f}")
            bot_info_lines.append(f"Pos: ({bot['x']:.1f}, {bot['y']:.1f})")
            bot_info_lines.append(f"Heading: {bot.get('theta', 0.0):.0f}°")
            bot_info_lines.append(
                f"Signal: {self.signal_descriptions.get(bot.get('signal', 'none'), 'Unknown')}"
            )

            # Visible bots
            visible_bots = visible_objects["bots"]
            if visible_bots:
                bot_info_lines.append("Visible Bots:")
                for vb in visible_bots:
                    bot_info_lines.append(
                        f"  - Bot {vb['id']} (Team {vb['team']}, HP: {vb['hp']:.0f})"
                    )

            # Nearby projectiles
            nearby_projectiles = visible_objects["projectiles"]
            if nearby_projectiles:
                bot_info_lines.append("Nearby Projectiles:")
                for proj in nearby_projectiles:
                    bot_info_lines.append(
                        f"  - Team {proj.get('team', 'Unknown')} at ({proj['x']:.1f}, {proj['y']:.1f})"
                    )

            # Visible walls
            visible_walls = visible_objects["walls"]
            if visible_walls:
                bot_info_lines.append("Visible Walls:")
                for wall in visible_walls:
                    bot_info_lines.append(f"  - Wall at ({wall[0]:.1f}, {wall[1]:.1f})")
        else:
            bot_info_lines.append("Click on a bot to select it")

        self.bot_info_text.geometry.set_text("\n".join(bot_info_lines))

        # Events
        events = state.get("events", [])
        if events:
            lines = ["Recent Events:"]
            for event in events[-5:]:
                et = event.get("type", "unknown")
                if et == "shot":
                    text = f"Bot {event['bot_id']} fired"
                elif et == "hit":
                    text = (
                        f"Bot {event['projectile_shooter']} hit Bot {event['target']}"
                    )
                elif et == "death":
                    text = f"Bot {event['bot_id']} destroyed"
                else:
                    text = f"{et}: {event}"
                lines.append(text)
            self.events_text.geometry.set_text("\n".join(lines))
            w, h = self.canvas.get_logical_size()
            self.events_text.local.position = (w - 250, 10, 0)
        else:
            self.events_text.geometry.set_text("")

    def _toggle_play(self):
        self.playing = not self.playing

    def _reset_sim(self):
        self.playing = False
        self.current_frame = 0.0

    def _step_frame(self, direction: int):
        self.playing = False
        self.current_frame += direction
        self.current_frame = max(0.0, min(len(self.timeline) - 1, self.current_frame))

    def _change_playback_speed(self, direction: float):
        if direction > 0:
            self.playback_speed = min(16.0, self.playback_speed * 1.5)
        else:
            self.playback_speed = max(0.1, self.playback_speed / 1.5)

    def _reset_camera_view(self):
        self.camera.position.set(0, -self.arena_width * 1.2, self.arena_height * 1.1)
        self.camera.look_at(gfx.linalg.Vector3(0, 0, 0))
        self.camera_controls.target.set(0, 0, 0)

    def _toggle_fov(self):
        self.show_fov = not self.show_fov
        self._update_fov_display()

    def _update_fov_display(self):
        """Update the FOV indicator's visibility and position."""
        should_show = self.show_fov and self.selected_bot

        if should_show:
            if not self.fov_object:
                fov_geom = self._create_fov_geom(120, 15.0)
                fov_mat = gfx.MeshBasicMaterial(
                    color=(1, 1, 0, 0.3), side="both", transparent=True
                )
                self.fov_object = gfx.Mesh(fov_geom, fov_mat)
                self.scene.add(self.fov_object)

            bot = self.selected_bot
            self.fov_object.position.set(bot["x"], bot["y"], 0.1)
            self.fov_object.rotation.set_from_euler(
                gfx.linalg.Euler(0, 0, math.radians(bot["theta"]))
            )
            color = (0, 0.5, 1, 0.3) if bot["team"] == 0 else (1, 0.3, 0.3, 0.3)
            self.fov_object.material.color = color
            self.fov_object.visible = True
        elif self.fov_object:
            self.fov_object.visible = False

    def _create_fov_geom(self, fov_angle_deg, fov_range, num_segments=20):
        """Create a Geometry for a 2D FOV fan."""
        fov_angle_rad = math.radians(fov_angle_deg)

        positions = np.zeros((num_segments + 2, 3), dtype=np.float32)
        indices = np.zeros((num_segments, 3), dtype=np.int32)

        # Center vertex
        positions[0] = (0, 0, 0)

        # Arc vertices
        angle_step = fov_angle_rad / num_segments
        start_angle = -fov_angle_rad / 2

        for i in range(num_segments + 1):
            angle = start_angle + i * angle_step
            # Create fan along +X axis and rotate later
            x = fov_range * math.cos(angle)
            y = fov_range * math.sin(angle)
            positions[i + 1] = (x, y, 0)

        for i in range(num_segments):
            indices[i] = (0, i + 1, i + 2)

        return gfx.Geometry(positions=positions, indices=indices)

    def animate(self):
        """Main animation loop."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if self.playing:
            frame_advance = (
                self.playback_speed * dt * 10
            )  # Adjust for the desired speed.
            self.current_frame += frame_advance

            if self.current_frame >= len(self.timeline) - 1:
                self.current_frame = len(self.timeline) - 1
                self.playing = False

        # Ensure current_frame is within bounds
        self.current_frame = max(0.0, min(self.current_frame, len(self.timeline) - 1))

        current_state = self._get_current_state()

        # Update selected bot with data from the current frame
        if self.selected_bot:
            bot_id = self.selected_bot["id"]
            found_bot = next(
                (b for b in current_state.get("bots", []) if b["id"] == bot_id), None
            )
            self.selected_bot = found_bot
            if self.selected_bot:
                self.selected_bot_info = self._get_detailed_bot_info(bot_id)

        self._update_bots(current_state)
        self._update_projectiles(current_state)
        self._update_ui(current_state)
        self._update_fov_display()

        self.renderer.render(self.scene, self.camera)
        self.renderer.render(self.ui_scene, self.ui_camera)
        self.canvas.request_draw()


def run_3d_viewer(battle_file: str):
    """Launch 3D viewer with a saved battle JSON file."""
    print(f"\n=== pygfx Battle Viewer: {battle_file} ===")

    try:
        with open(battle_file, "r") as f:
            battle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Battle file '{battle_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{battle_file}'")
        return

    print("Launching pygfx viewer...")
    print("Controls:")
    print("  SPACE = Play/Pause")
    print("  ←/→ = Step frame by frame")
    print("  R = Reset to start")
    print("  C = Reset camera view")
    print("  F = Toggle FOV display")
    print("  Q/ESC = Quit")
    print("  Click bots to select them")
    print("\n  Camera Controls:")
    print("    - Pan: Right-click + Drag Mouse")
    print("    - Orbit: Left-click + Drag Mouse")
    print("    - Zoom: Use Mouse Wheel")

    canvas = WgpuCanvas(size=(1280, 960), title="pygfx Battle Viewer")
    app = Battle3DViewer(battle_data, canvas)

    # Using asyncio to run the animation loop
    async def mainloop():
        while True:
            app.animate()
            await asyncio.sleep(
                1 / 60
            )  # Adjust sleep time for consistent frame rate, if needed

    # This is a bit of a hack to integrate with wgpu's run()
    # which expects to own the loop.
    canvas.request_draw(app.animate)

    run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graphics-3d-pygfx.py <battle_log.json>")
        sys.exit(1)

    battle_file = sys.argv[1]
    run_3d_viewer(battle_file)
