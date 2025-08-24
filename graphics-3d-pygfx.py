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
import pygfx.text as gfx_text
import pylinalg as la
from wgpu.gui.auto import WgpuCanvas, run

# Import visibility system to use same logic as 2D viewer
try:
    from python_llm import PythonLLMController
except ImportError:
    PythonLLMController = None


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
        self.camera.local.position = (0, -arena_diagonal * 0.9, arena_diagonal * 0.9)
        self.camera.look_at((0, 0, 0))

        # Lighting
        self.scene.add(gfx.AmbientLight(intensity=0.4))
        sun = gfx.DirectionalLight(intensity=0.8)
        sun.local.position = (-1, -2, 3)
        self.scene.add(sun)

        fill_light = gfx.DirectionalLight(intensity=0.4)
        fill_light.local.position = (1, 2, 1)
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
            wall.local.position = (center_x, center_y, wall_3d_height / 2)
            wall.local.rotation = la.quat_from_axis_angle(
                (0, 0, 1), math.radians(angle_deg)
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
        text_geom = gfx_text.TextGeometry(text, font_size=font_size, anchor=anchor)
        text_mat = gfx_text.TextMaterial(color=color)
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
                    self.playing = False
                    self._update_fov_display()
                    break

    def _get_current_state(self) -> Dict:
        """Get the state for the current frame, with interpolation for smooth animation."""
        if not self.timeline:
            return {}

        frame_idx_float = self.current_frame
        frame_idx = int(frame_idx_float)
        interp = frame_idx_float - frame_idx

        # Clamp frame indices
        frame_idx = max(0, min(len(self.timeline) - 1, frame_idx))
        next_frame_idx = min(frame_idx + 1, len(self.timeline) - 1)

        state1 = self.timeline[frame_idx]
        if frame_idx == next_frame_idx:
            return state1  # No interpolation at the end of the timeline

        state2 = self.timeline[next_frame_idx]

        # Create a new state, starting with data from the first frame
        interp_state = state1.copy()

        # Interpolate time
        time1 = state1.get("time", 0)
        time2 = state2.get("time", 0)
        interp_state["time"] = time1 + (time2 - time1) * interp

        # Interpolate bots
        bots1 = {bot["id"]: bot for bot in state1.get("bots", [])}
        bots2 = {bot["id"]: bot for bot in state2.get("bots", [])}
        interp_bots = []

        for bot_id, bot1 in bots1.items():
            if bot_id in bots2:
                bot2 = bots2[bot_id]
                # Always interpolate if bot exists in both frames.
                # The `alive` flag from state1 is carried over. The bot will be
                # removed on the next discrete frame if it's dead in that frame.
                interp_bot = bot1.copy()

                # Interpolate position
                interp_bot["x"] = bot1["x"] + (bot2["x"] - bot1["x"]) * interp
                interp_bot["y"] = bot1["y"] + (bot2["y"] - bot1["y"]) * interp

                # Interpolate angle (heading), handling wraparound from 0-360 degrees
                theta1 = bot1["theta"]
                theta2 = bot2["theta"]
                d_theta = theta2 - theta1
                if d_theta > 180:
                    d_theta -= 360
                elif d_theta < -180:
                    d_theta += 360
                interp_bot["theta"] = theta1 + d_theta * interp

                interp_bots.append(interp_bot)
            else:
                # Bot not in next frame, just use its last known state
                interp_bots.append(bot1)

        interp_state["bots"] = interp_bots

        # Interpolate projectiles using velocities for smoother constant-speed motion.
        projectiles1 = {p["id"]: p for p in state1.get("projectiles", []) if "id" in p}
        projectiles2 = {p["id"]: p for p in state2.get("projectiles", []) if "id" in p}
        interp_projectiles = []

        dt = max(0.0, time2 - time1)
        for proj_id, proj1 in projectiles1.items():
            if proj_id in projectiles2:
                interp_proj = proj1.copy()
                if dt > 0:
                    t = interp * dt
                    vx = proj1.get("vx", 0.0)
                    vy = proj1.get("vy", 0.0)
                    interp_proj["x"] = proj1["x"] + vx * t
                    interp_proj["y"] = proj1["y"] + vy * t
                else:
                    # Fallback to position lerp if no time delta
                    proj2 = projectiles2[proj_id]
                    interp_proj["x"] = proj1["x"] + (proj2["x"] - proj1["x"]) * interp
                    interp_proj["y"] = proj1["y"] + (proj2["y"] - proj1["y"]) * interp
                interp_projectiles.append(interp_proj)
            # If a projectile from state1 is not in state2, it has been removed
            # (e.g., hit a wall or expired), so we don't add it to the interpolated state.

        interp_state["projectiles"] = interp_projectiles

        return interp_state

    def _get_visible_objects_for_bot(self, selected_bot, current_state):
        """Get visible objects for a bot using the same system as bot programs."""
        if self.llm_controller is None:
            return self._get_visible_objects_fallback(selected_bot, current_state)

        # Create a mock arena compatible with the LLM visibility API
        from battle_sim import Arena

        class MockArena:
            def __init__(self, viewer, current_state):
                self.SENSE_RANGE = 15.0
                self.FOV_ANGLE = math.radians(120)
                self.BOT_RADIUS = 0.5
                self.bot_data = {}
                self.bot_bodies = {}
                self.projectile_data = {}
                self.projectile_bodies = {}
                self.wall_bodies = []

                for bot in current_state.get("bots", []):
                    if bot["alive"]:
                        bot_id = bot["id"]

                        class MockBotData:
                            def __init__(self, bot_info):
                                self.team = bot_info["team"]
                                self.hp = bot_info["hp"]
                                self.signal = bot_info.get("signal", "none")

                        class MockBotBody:
                            def __init__(self, bot_info):
                                self.position = (bot_info["x"], bot_info["y"])
                                self.angle = math.radians(bot_info["theta"])
                                self.velocity = (
                                    bot_info.get("vx", 0),
                                    bot_info.get("vy", 0),
                                )

                        self.bot_data[bot_id] = MockBotData(bot)
                        self.bot_bodies[bot_id] = MockBotBody(bot)

                for proj in current_state.get("projectiles", []):
                    proj_id = len(self.projectile_data)

                    class MockProjData:
                        def __init__(self, proj_info):
                            self.team = proj_info.get("team", 0)
                            self.ttl = proj_info.get("ttl", 1.0)

                    class MockProjBody:
                        def __init__(self, proj_info):
                            self.position = (proj_info["x"], proj_info["y"])
                            self.velocity = (
                                proj_info.get("vx", 0),
                                proj_info.get("vy", 0),
                            )

                    self.projectile_data[proj_id] = MockProjData(proj)
                    self.projectile_bodies[proj_id] = MockProjBody(proj)

                class MockWallShape:
                    def __init__(self, vertices):
                        self._vertices = vertices

                    def get_vertices(self):
                        return self._vertices

                walls_data = viewer.metadata.get("walls", [])
                for wall_def in walls_data:
                    cx, cy, w, h, angle_deg = wall_def
                    angle_rad = math.radians(angle_deg)
                    c, s = math.cos(angle_rad), math.sin(angle_rad)
                    hw, hh = w / 2, h / 2
                    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                    rotated_corners = [
                        (p[0] * c - p[1] * s, p[0] * s + p[1] * c) for p in corners
                    ]
                    abs_corners = [(p[0] + cx, p[1] + cy) for p in rotated_corners]
                    wall_shape = MockWallShape(abs_corners)
                    self.wall_bodies.append((None, wall_shape))

            def _is_bot_alive(self, bot_id):
                return bot_id in self.bot_data

        mock_arena = MockArena(self, current_state)
        bot_id = selected_bot["id"]

        try:
            visible_objects = self.llm_controller.generate_visible_objects(
                mock_arena, bot_id
            )
            return visible_objects
        except Exception as e:
            print(f"Warning: Visibility system failed, using fallback: {e}")
            return self._get_visible_objects_fallback(selected_bot, current_state)

    def _get_visible_objects_fallback(self, selected_bot, current_state):
        """Fallback visibility system using simple distance checks."""
        visible_objects = []
        max_range = 15.0

        for bot in current_state.get("bots", []):
            if bot["id"] == selected_bot["id"] or not bot["alive"]:
                continue
            dx = bot["x"] - selected_bot["x"]
            dy = bot["y"] - selected_bot["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= max_range:
                bearing = math.degrees(math.atan2(dy, dx))
                if bearing < 0:
                    bearing += 360
                bot_type = "friend" if bot["team"] == selected_bot["team"] else "enemy"
                visible_objects.append(
                    {
                        "type": bot_type,
                        "x": bot["x"],
                        "y": bot["y"],
                        "distance": distance,
                        "angle": bearing,
                        "hp": bot["hp"],
                        "team": f"team_{bot['team']}",
                        "id": bot["id"],
                        "velocity_x": bot.get("vx", 0),
                        "velocity_y": bot.get("vy", 0),
                        "signal": bot.get("signal", "none"),
                    }
                )

        for proj in current_state.get("projectiles", []):
            if proj.get("team") == selected_bot["team"]:
                continue
            dx = proj["x"] - selected_bot["x"]
            dy = proj["y"] - selected_bot["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= max_range:
                bearing = math.degrees(math.atan2(dy, dx))
                if bearing < 0:
                    bearing += 360
                visible_objects.append(
                    {
                        "type": "projectile",
                        "x": proj["x"],
                        "y": proj["y"],
                        "distance": distance,
                        "angle": bearing,
                        "velocity_x": proj.get("vx", 0),
                        "velocity_y": proj.get("vy", 0),
                        "ttl": proj.get("ttl", 1.0),
                        "team": f"team_{proj.get('team', 0)}",
                    }
                )

        return visible_objects

    def _get_visible_bots(self, selected_bot, current_state):
        """Return list of (bot_data, distance, angle) for visible bots."""
        visible_objects = self._get_visible_objects_for_bot(selected_bot, current_state)
        visible_bots = []
        for obj in visible_objects:
            if obj["type"] in ["enemy", "friend"]:
                bot_data = None
                for bot in current_state.get("bots", []):
                    if bot["id"] == obj.get("id"):
                        bot_data = bot
                        break
                if bot_data:
                    visible_bots.append((bot_data, obj["distance"], obj["angle"]))
        visible_bots.sort(key=lambda x: x[1])
        return visible_bots

    def _get_nearby_projectiles(self, selected_bot, current_state):
        """Return list of (proj_data, distance, angle) for nearby projectiles."""
        visible_objects = self._get_visible_objects_for_bot(selected_bot, current_state)
        nearby = []
        for obj in visible_objects:
            if obj["type"] == "projectile":
                proj_data = {
                    "x": obj["x"],
                    "y": obj["y"],
                    "velocity_x": obj.get("velocity_x", 0),
                    "velocity_y": obj.get("velocity_y", 0),
                    "team": obj.get("team", "unknown"),
                    "ttl": obj.get("ttl", 0),
                }
                nearby.append((proj_data, obj["distance"], obj["angle"]))
        nearby.sort(key=lambda x: x[1])
        return nearby

    def _get_visible_walls(self, selected_bot, current_state=None):
        """Return list of (wall_obj, distance, angle) for visible walls."""
        if current_state is None:
            current_state = self.timeline[int(self.current_frame)]
        visible_objects = self._get_visible_objects_for_bot(selected_bot, current_state)
        walls = []
        for obj in visible_objects:
            if obj["type"] == "wall":
                walls.append((obj, obj["distance"], obj["angle"]))
        walls.sort(key=lambda x: x[1])
        return walls

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
                heading_cone.local.position = (0, 0.25, 0)  # In front of body
                heading_cone.local.rotation = la.quaternion_from_axis_angle(
                    (1, 0, 0), math.pi / 2
                )
                bot_body.add(heading_cone)

                # Health bar
                hb_bg_geom = gfx.plane_geometry(0.6, 0.06)
                hb_bg_mat = gfx.MeshBasicMaterial(color=(0.2, 0.2, 0.2))
                hb_bg = gfx.Mesh(hb_bg_geom, hb_bg_mat)
                hb_bg.local.position = (0, 0, 0.6)
                bot_body.add(hb_bg)

                hb_fill_geom = gfx.plane_geometry(0.6, 0.06)
                hb_fill_mat = gfx.MeshBasicMaterial(color=(0, 1, 0))
                hb_fill = gfx.Mesh(hb_fill_geom, hb_fill_mat)
                hb_fill.local.position = (0, 0, 0.001)  # Slightly above bg
                hb_bg.add(hb_fill)

                # Bot ID label
                id_text = gfx.Text(
                    gfx_text.TextGeometry(str(bot_id), font_size=10, anchor="center"),
                    gfx_text.TextMaterial(color="#FFF"),
                )
                id_text.local.position = (0, 0, 1.0)
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
            bot_body.local.position = pos
            # Sim angle: 0 is +X. pygfx Z-up: rotation is around Z axis.
            bot_body.local.rotation = la.quaternion_from_axis_angle(
                (0, 0, 1), math.radians(bot["theta"])
            )

            # Update health bar
            hp_ratio = max(0.0, min(1.0, bot["hp"] / 100.0))
            hb_fill = bot_obj_group["hb_fill"]
            hb_fill.local.scale = (hp_ratio, 1, 1)
            hb_fill.local.position = (-0.3 * (1 - hp_ratio), 0, 0.001)
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
            np.local.position = pos

    def _update_ui(self, state: Dict):
        """Update the text in the UI panels."""
        # General info + metadata + summary
        time_info = f"Time: {state.get('time', 0):.1f}s"
        frame_info = f"Frame: {int(self.current_frame)}/{len(self.timeline) - 1}"
        speed_info = f"Speed: {self.playback_speed:.1f}x"

        meta = self.metadata or {}
        winner = meta.get("winner", "unknown")
        reason = meta.get("reason", "unknown")

        summary = self.battle_data.get("summary", {})
        mvp = summary.get("mvp", {}) or {}
        intensity = summary.get("battle_intensity", 0)
        accuracy = summary.get("overall_accuracy", 0)

        info_lines = [
            time_info,
            frame_info,
            speed_info,
            f"Winner: {winner} ({reason})",
        ]
        if mvp.get("bot_id") is not None:
            info_lines.append(
                f"MVP: Bot {mvp.get('bot_id')} (Team {mvp.get('team')}) - {mvp.get('score', 0):.1f} pts"
            )
        info_lines.append(f"Intensity: {float(intensity):.1f} shots/sec")
        info_lines.append(f"Overall Accuracy: {float(accuracy):.1%}")
        self.info_text.geometry.set_text("\n".join(info_lines))

        # Selected bot info with more details
        if self.selected_bot:
            bot = self.selected_bot
            heading = bot.get("theta", 0.0)
            speed = math.sqrt(bot.get("vx", 0) ** 2 + bot.get("vy", 0) ** 2)
            signal = bot.get("signal", "none")
            signal_desc = self.signal_descriptions.get(signal, "Unknown signal")

            # Function info from summary (supports int or str keys)
            bot_functions = summary.get("bot_functions", {}) or {}
            bot_func_data = (
                bot_functions.get(bot["id"]) or bot_functions.get(str(bot["id"])) or {}
            )
            personality = bot_func_data.get("personality", "unknown")
            version = bot_func_data.get("version", "N/A")

            # Visible objects summary
            visible_bots = self._get_visible_bots(bot, state)
            friends_count = len(
                [b for b, _, _ in visible_bots if b["team"] == bot["team"]]
            )
            enemies_count = len(visible_bots) - friends_count
            nearby_projectiles = self._get_nearby_projectiles(bot, state)
            visible_walls = self._get_visible_walls(bot, state)

            info = [
                f"Bot {bot['id']} (Team {bot['team']})",
                f"Function: {personality}_combat_v{version}",
                f"HP: {bot['hp']}",
                f"Pos: ({bot['x']:.1f}, {bot['y']:.1f})",
                f"Heading: {heading:.0f}°  Speed: {speed:.1f} m/s",
                f"Signal: {signal}",
                f"{signal_desc}",
                f"Tactical: {friends_count}F, {enemies_count}E, {len(nearby_projectiles)}P, {len(visible_walls)}W",
            ]
            self.bot_info_text.geometry.set_text("\n".join(info))

            # Detailed tactical info
            tactical_lines = ["--- Tactical Situation ---"]
            if visible_bots:
                tactical_lines.append("Units:")
                for vis_bot, distance, bearing in visible_bots:
                    if vis_bot["team"] == bot["team"]:
                        unit_type = "F"
                        signal = vis_bot.get("signal", "none")
                        signal_part = f" [{signal}]" if signal != "none" else ""
                        vis_text = f"  {unit_type}{vis_bot['id']}: {distance:.1f}m @ {bearing:.0f}°{signal_part}"
                    else:
                        unit_type = "E"
                        vis_text = f"  {unit_type}{vis_bot['id']}: {distance:.1f}m @ {bearing:.0f}°"
                    tactical_lines.append(vis_text)

            if nearby_projectiles:
                tactical_lines.append("Projectiles:")
                for proj, distance, bearing in nearby_projectiles:
                    proj_text = f"  P: {distance:.1f}m @ {bearing:.0f}°"
                    tactical_lines.append(proj_text)

            if visible_walls:
                tactical_lines.append("Walls:")
                for wall, distance, bearing in visible_walls:
                    wall_text = f"  W: {distance:.1f}m @ {bearing:.0f}°"
                    tactical_lines.append(wall_text)

            if len(tactical_lines) > 1:
                self.tactical_info_text.geometry.set_text("\n".join(tactical_lines))
            else:
                self.tactical_info_text.geometry.set_text("")
        else:
            self.bot_info_text.geometry.set_text("Click on a bot to select it")
            self.tactical_info_text.geometry.set_text("")

        # Recent events (last 5)
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
        self.camera.local.position = (
            0,
            -self.arena_width * 1.2,
            self.arena_height * 1.1,
        )
        self.camera.look_at((0, 0, 0))
        self.camera_controls.target = (0, 0, 0)

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
            self.fov_object.local.position = (bot["x"], bot["y"], 0.1)
            self.fov_object.local.rotation = la.quaternion_from_axis_angle(
                (0, 0, 1), math.radians(bot["theta"])
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
