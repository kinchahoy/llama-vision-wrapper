"""
Ursina 3D Battle Viewer
Interactive 3D visualization of battle simulations with glorious 3D rendering.
"""

import math
import json
from typing import Dict, List, Tuple, Optional
from ursina import *
from ursina import Vec2, Vec3

# Import visibility system to use same logic as bot programs
try:
    from python_llm import PythonLLMController
except ImportError:
    PythonLLMController = None


class Battle3DViewer:
    """Interactive 3D viewer for battle simulations using Ursina engine."""

    def __init__(self, battle_data: Dict):
        """Initialize 3D viewer with battle data."""
        self.battle_data = battle_data
        self.timeline = battle_data["timeline"]
        self.metadata = battle_data["metadata"]

        # Playback state
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self.target_fps = 60

        # Arena scaling
        arena_size = self.metadata.get("arena_size", [20, 20])
        self.arena_width, self.arena_height = arena_size

        # UI state (matching graphics.py)
        self.show_fov = False
        self.show_trails = False
        self.selected_bot = None
        self.dragging_scrubber = False
        self.camera_mode = "overview"  # Keep for compatibility but always use bird's eye

        # Initialize visibility system (same as bot programs use)
        if PythonLLMController is not None:
            self.llm_controller = PythonLLMController()
        else:
            self.llm_controller = None

        # Initialize Ursina with matching window size
        self.app = Ursina()
        window.title = "LLM Battle Sim 3D Viewer"
        window.borderless = False
        window.fullscreen = False
        window.size = (1000, 700)  # Match graphics.py window size
        window.exit_button.visible = False
        window.fps_counter.enabled = True

        # Initialize scene
        self._setup_scene()
        self._setup_ui()

        # Assign input/update handlers
        self.app.input = self.input
        self.app.update = self.update

        # Bot and projectile entities
        self.bot_entities = {}
        self.projectile_entities = []
        self.trail_entities = {}
        self.wall_entities = []
        self.fov_entities = []

        # Create initial objects
        self._create_arena()
        self._create_walls()

    def _setup_scene(self):
        """Set up the 3D scene with lighting and bird's eye camera."""
        # Lighting optimized for top-down view
        DirectionalLight().look_at(Vec3(0, -1, 0))  # Light from directly above
        AmbientLight(color=color.rgba(150, 150, 150, 0.3))  # Brighter ambient light

        # Dark sky for better contrast
        Sky(color=color.dark_gray)

        # Camera setup to show full arena with proper zoom
        # Position camera high enough to see the entire arena
        arena_diagonal = math.sqrt(self.arena_width**2 + self.arena_height**2)
        camera_height = arena_diagonal * 1.2  # Height to see full arena
        camera.position = Vec3(0, camera_height, -arena_diagonal * 0.3)
        camera.rotation_x = 70  # Slight angle for better 3D perspective
        camera.rotation_y = 0
        camera.rotation_z = 0
        camera.fov = 50  # Wider field of view to see entire arena

        # Remove ground reference as we'll see the arena floor directly

    def _setup_ui(self):
        """Set up UI elements matching graphics.py layout."""
        # Timeline scrubber (bottom of screen)
        self.timeline_panel = Entity(
            parent=camera.ui,
            model="cube",
            color=color.dark_gray,
            scale=(0.8, 0.03, 1),
            position=(0, -0.47, 0),
        )

        self.timeline_progress = Entity(
            parent=camera.ui,
            model="cube",
            color=color.blue,
            scale=(0.01, 0.025, 1),
            position=(-0.4, -0.47, -0.1),
        )

        # Timeline position indicator
        self.timeline_indicator = Entity(
            parent=camera.ui,
            model="cube",
            color=color.white,
            scale=(0.005, 0.04, 1),
            position=(-0.4, -0.47, -0.15),
        )

        # Control info text (bottom)
        self.control_text = Text(
            "Controls: SPACE=Play/Pause, ←→=Step, R=Reset, +/-=Speed, F=FOV, Q=Quit, Click bots to select",
            parent=camera.ui,
            position=(-0.85, -0.4),
            scale=0.6,
            color=color.white,
        )

        # Battle info text (right side, outside 3D area)
        self.info_text = Text(
            "", 
            parent=camera.ui, 
            position=(0.4, 0.4), 
            scale=0.7, 
            color=color.white
        )

        # Selected bot info panel (left side, overlaid on 3D)
        self.bot_info_panel = Entity(
            parent=camera.ui,
            model="quad",
            color=color.rgba(20, 20, 20, 200),
            scale=(0.4, 0.6),
            position=(-0.7, 0.1),
            visible=False,
        )
        self.bot_info_text = Text(
            "",
            parent=self.bot_info_panel,
            origin=(-0.5, 0.5),
            position=(-0.45, 0.45),
            scale=1.2, # scale is relative to parent, so needs to be adjusted
            color=color.white,
        )

    def input(self, key):
        """Handle keyboard and mouse input."""
        if key == "space":
            self.playing = not self.playing
        elif key == "r":
            self.current_frame = 0
            self.playing = False
        elif key == "left arrow":
            self.current_frame = max(0, self.current_frame - 1)
            self.playing = False
        elif key == "right arrow":
            self.current_frame = min(len(self.timeline) - 1, self.current_frame + 1)
            self.playing = False
        elif key == "plus" or key == "equal":
            self.playback_speed = min(5.0, self.playback_speed * 1.5)
        elif key == "minus":
            self.playback_speed = max(0.1, self.playback_speed / 1.5)
        elif key == "f":
            self.show_fov = not self.show_fov
            self._update_fov_display()
        elif key == "q" or key == "escape":
            application.quit()
        elif key == "left mouse down":
            self._handle_mouse_input()
            self._handle_bot_selection()
        elif key == "left mouse up":
            self.dragging_scrubber = False

    def _create_arena(self):
        """Create the arena floor and boundaries."""
        # Arena floor
        self.arena_floor = Entity(
            model="cube",
            texture="white_cube",
            color=color.gray,
            scale=(self.arena_width, 0.1, self.arena_height),
            position=(0, 0, 0),
        )

        # Arena boundaries
        wall_height = 2
        wall_thickness = 0.5

        # North wall
        Entity(
            model="cube",
            color=color.white,
            scale=(self.arena_width + wall_thickness, wall_height, wall_thickness),
            position=(0, wall_height / 2, self.arena_height / 2 + wall_thickness / 2),
        )

        # South wall
        Entity(
            model="cube",
            color=color.white,
            scale=(self.arena_width + wall_thickness, wall_height, wall_thickness),
            position=(0, wall_height / 2, -self.arena_height / 2 - wall_thickness / 2),
        )

        # East wall
        Entity(
            model="cube",
            color=color.white,
            scale=(wall_thickness, wall_height, self.arena_height),
            position=(self.arena_width / 2 + wall_thickness / 2, wall_height / 2, 0),
        )

        # West wall
        Entity(
            model="cube",
            color=color.white,
            scale=(wall_thickness, wall_height, self.arena_height),
            position=(-self.arena_width / 2 - wall_thickness / 2, wall_height / 2, 0),
        )

    def _create_walls(self):
        """Create interior walls matching the 2D version."""
        wall_height = 1.5
        wall_thickness = 0.3
        wall_color = color.gray

        width = self.arena_width
        height = self.arena_height

        # Interior walls (matching the ones in graphics.py)
        walls = [
            # Horizontal wall in upper area
            {
                "start": (width * 0.2, height * 0.7),
                "end": (width * 0.2 + 10, height * 0.7),
            },
            # Vertical wall in middle-left
            {
                "start": (width * 0.4, height * 0.3),
                "end": (width * 0.4, height * 0.3 + 8),
            },
            # Horizontal wall in lower-right
            {
                "start": (width * 0.6, height * 0.2),
                "end": (width * 0.6 + 9, height * 0.2),
            },
            # Short vertical wall in upper-right
            {
                "start": (width * 0.8, height * 0.6),
                "end": (width * 0.8, height * 0.6 + 6),
            },
        ]

        # Convert 2D walls to 3D (Y becomes Z, flip Z coordinate system)
        for wall in walls:
            start_x, start_z = wall["start"]
            end_x, end_z = wall["end"]

            # Convert from arena coordinates (Y up) to 3D coordinates (Z forward)
            start_z = start_z - height / 2  # Center at origin
            end_z = end_z - height / 2
            start_x = start_x - width / 2
            end_x = end_x - width / 2

            # Calculate wall center and dimensions
            center_x = (start_x + end_x) / 2
            center_z = (start_z + end_z) / 2
            wall_length = math.sqrt((end_x - start_x) ** 2 + (end_z - start_z) ** 2)

            # Calculate rotation
            wall_angle = math.atan2(end_z - start_z, end_x - start_x)

            wall_entity = Entity(
                model="cube",
                color=wall_color,
                scale=(wall_length, wall_height, wall_thickness),
                position=(center_x, wall_height / 2, center_z),
                rotation_y=math.degrees(wall_angle),
            )

            self.wall_entities.append(wall_entity)

    def _handle_mouse_input(self):
        """Handle mouse input for timeline scrubbing."""
        # This is called on mouse down and on motion while dragging
        norm_x = (mouse.x + 1) / 2  # Convert from [-1,1] to [0,1]
        norm_y = (mouse.y + 1) / 2  # Convert from [-1,1] to [0,1]

        # Check if clicking on timeline (bottom area)
        if norm_y < 0.15:  # Bottom 15% of screen
            if 0.1 < norm_x < 0.9:  # Timeline area
                self.dragging_scrubber = True
                # Calculate target frame
                progress = (norm_x - 0.1) / 0.8  # Normalize to timeline area
                target_frame = int(progress * (len(self.timeline) - 1))
                self.current_frame = max(0, min(len(self.timeline) - 1, target_frame))
                self.playing = False

    def _handle_bot_selection(self):
        """Handle bot selection with mouse clicks."""
        if not self.dragging_scrubber and mouse.hovered_entity and hasattr(mouse.hovered_entity, 'bot_id'):
            # Find bot data from current state
            current_state = self._get_current_state()
            for bot in current_state.get("bots", []):
                if bot["id"] == mouse.hovered_entity.bot_id and bot["alive"]:
                    self.selected_bot = bot
                    self.playing = False
                    self._update_fov_display()
                    break

    def _update_camera(self, current_state: Dict):
        """Keep camera in bird's eye view position."""
        # Always maintain bird's eye view
        # Keep the camera at the calculated position that shows full arena
        arena_diagonal = math.sqrt(self.arena_width**2 + self.arena_height**2)
        camera_height = arena_diagonal * 1.2
        camera.position = Vec3(0, camera_height, -arena_diagonal * 0.3)
        camera.rotation_x = 70
        camera.rotation_y = 0
        camera.rotation_z = 0

    def _update_fov_display(self):
        """Update FOV visualization for selected bot."""
        # Remove existing FOV entities
        for entity in self.fov_entities:
            if entity:
                destroy(entity)
        self.fov_entities.clear()

        if self.show_fov and self.selected_bot:
            self._create_fov_indicator(self.selected_bot)

    def _create_fov_indicator(self, bot: Dict):
        """Create 3D FOV indicator for a bot."""
        bot_x = bot["x"] - self.arena_width / 2
        bot_z = -(bot["y"] - self.arena_height / 2)
        bot_heading = math.radians(bot["theta"])

        fov_range = 15.0
        fov_angle = math.radians(120)

        # Create FOV cone as a flat sector visible from above
        vertices = [(0, 0, 0)]  # Center point at bot position

        # Generate arc points
        for i in range(25):  # 24 segments for smooth arc
            angle = bot_heading - fov_angle / 2 + (fov_angle * i / 24)
            x = math.sin(angle) * fov_range
            z = math.cos(angle) * fov_range
            vertices.append((x, 0.1, z))  # Slightly above ground

        # Create FOV cone entity
        team_color = color.blue if bot["team"] == 0 else color.red
        fov_entity = Entity(
            model=Mesh(vertices=vertices, mode="triangle_fan"),
            color=color.rgba(*[int(c*255) for c in team_color[:3]], 80),
            position=(bot_x, 1, bot_z),
        )
        self.fov_entities.append(fov_entity)

    def _update_trail_display(self):
        """Update projectile trail visualization."""
        if not self.show_trails:
            # Remove all trail entities
            for trail_list in self.trail_entities.values():
                for trail_entity in trail_list:
                    destroy(trail_entity)
            self.trail_entities.clear()

    def update(self):
        """Main update loop, called every frame."""
        if self.dragging_scrubber:
            self._handle_mouse_input()

        if not self.timeline:
            return

        # Update playback
        if self.playing:
            frames_per_second = 10  # 10Hz logging rate
            frame_advance = frames_per_second * self.playback_speed * time.dt
            self.current_frame += frame_advance

            if self.current_frame >= len(self.timeline):
                self.current_frame = len(self.timeline) - 1
                self.playing = False

        # Get current state
        current_state = self._get_current_state()

        # Update 3D objects
        self._update_bots(current_state)
        self._update_projectiles(current_state)
        self._update_camera(current_state)
        self._update_ui(current_state)

    def run(self):
        """Main viewer loop."""
        self.app.run()

    def _get_current_state(self) -> Dict:
        """Get current frame state."""
        if not self.timeline:
            return {}

        frame_idx = int(self.current_frame)
        frame_idx = max(0, min(len(self.timeline) - 1, frame_idx))
        return self.timeline[frame_idx]

    def _update_bots(self, state: Dict):
        """Update bot 3D entities."""
        current_bots = state.get("bots", [])

        # Remove entities for dead bots
        for bot_id in list(self.bot_entities.keys()):
            bot_alive = any(
                bot["id"] == bot_id and bot["alive"] for bot in current_bots
            )
            if not bot_alive:
                destroy(self.bot_entities[bot_id])
                del self.bot_entities[bot_id]

        # Update or create bot entities
        for bot in current_bots:
            if not bot["alive"]:
                continue

            bot_id = bot["id"]

            # Convert coordinates (flip Z and center)
            x = bot["x"] - self.arena_width / 2
            z = -(bot["y"] - self.arena_height / 2)
            y = 1.0  # Height above ground

            if bot_id not in self.bot_entities:
                # Create new bot entity
                bot_color = color.blue if bot["team"] == 0 else color.red
                bot_entity = Entity(
                    model="sphere", color=bot_color, scale=0.8, position=(x, y, z)
                )
                bot_entity.bot_id = bot_id
                self.bot_entities[bot_id] = bot_entity

                # Add heading indicator (small cylinder)
                heading_indicator = Entity(
                    model="cube",
                    color=color.white,
                    scale=(0.1, 0.1, 1),
                    parent=bot_entity,
                    position=(0, 0.3, 0),
                )
            else:
                # Update existing bot
                bot_entity = self.bot_entities[bot_id]
                bot_entity.position = (x, y, z)

            # Update rotation based on heading
            bot_entity.rotation_y = -bot[
                "theta"
            ]  # Negative because of coordinate system

            # Update health indicator (scale based on HP)
            health_ratio = max(0.3, bot["hp"] / 100)  # Minimum 30% size
            bot_entity.scale = 0.8 * health_ratio

            # Update color based on health
            if bot["hp"] > 60:
                base_color = color.blue if bot["team"] == 0 else color.red
            elif bot["hp"] > 30:
                base_color = color.yellow
            else:
                base_color = color.orange

            bot_entity.color = base_color

    def _update_projectiles(self, state: Dict):
        """Update projectile 3D entities."""
        # Remove all existing projectile entities
        for proj_entity in self.projectile_entities:
            destroy(proj_entity)
        self.projectile_entities.clear()

        # Create new projectile entities
        for proj in state.get("projectiles", []):
            # Convert coordinates
            x = proj["x"] - self.arena_width / 2
            z = -(proj["y"] - self.arena_height / 2)
            y = 1.5  # Height above ground

            # Projectile color based on team
            proj_color = color.cyan if proj.get("team") == 0 else color.magenta

            proj_entity = Entity(
                model="sphere", 
                color=proj_color, 
                scale=Vec3(0.3, 0.3, 0.3), 
                position=Vec3(x, y, z)
            )
            self.projectile_entities.append(proj_entity)

            # Add trail effect if enabled
            if self.show_trails:
                trail_entity = Entity(
                    model="cube",
                    color=color.rgba(int(proj_color.r*255), int(proj_color.g*255), int(proj_color.b*255), 100),
                    scale=Vec3(0.1, 0.1, 2),
                    position=Vec3(x, y, z),
                )
                self.projectile_entities.append(trail_entity)

    def _update_ui(self, state: Dict):
        """Update UI elements."""
        # Update timeline progress
        if len(self.timeline) > 1:
            progress = self.current_frame / (len(self.timeline) - 1)
            self.timeline_progress.x = -0.4 + (0.8 * progress)

        # Update info text
        info_lines = [
            f"Time: {state.get('time', 0):.1f}s",
            f"Frame: {int(self.current_frame)}/{len(self.timeline) - 1}",
            f"Speed: {self.playback_speed:.1f}x",
            f"Camera: {self.camera_mode}",
        ]

        if self.metadata:
            winner = self.metadata.get("winner", "unknown")
            reason = self.metadata.get("reason", "unknown")
            info_lines.append(f"Winner: {winner} ({reason})")

        self.info_text.text = "\n".join(info_lines)

        # Update selected bot info
        if self.selected_bot:
            self.bot_info_panel.visible = True
            self._update_selected_bot_info(state)
        else:
            self.bot_info_panel.visible = False
            self.bot_info_text.text = ""

    def _update_selected_bot_info(self, state: Dict):
        """Update comprehensive bot info panel."""
        bot = self.selected_bot
        bot_id = bot["id"]

        # Build comprehensive bot info text
        info_lines = [f"<bold>Bot {bot_id} (Team {bot['team']})</bold>"]

        # Function info
        bot_functions = self.battle_data.get("summary", {}).get("bot_functions", {})
        bot_func_data = bot_functions.get(str(bot_id), {})
        personality = bot_func_data.get("personality", "unknown")
        info_lines.append(f"Personality: {personality}")

        # State
        info_lines.append("\n<bold>State</bold>")
        info_lines.append(f"HP: {bot['hp']} | Signal: \"{bot.get('signal', 'none')}\"")
        info_lines.append(f"Pos: ({bot['x']:.1f}, {bot['y']:.1f}) | H: {bot['theta']:.0f}°")

        # Tactical
        visible_bots = self._get_visible_bots(bot, state)
        enemies = [b for b,_,_ in visible_bots if b['team'] != bot['team']]
        info_lines.append(f"\n<bold>Tactical ({len(enemies)} enemies visible)</bold>")
        for i, (vis_bot, dist, bear) in enumerate(visible_bots[:4]):
            utype = "F" if vis_bot["team"] == bot["team"] else "E"
            info_lines.append(f" {utype}{vis_bot['id']}: {dist:.1f}m @ {bear:.0f}°")

        # Performance
        summary = self.battle_data.get("summary", {})
        bot_scores = summary.get("bot_scores", [])
        bot_score = next((s for s in bot_scores if s["bot_id"] == bot_id), None)

        if bot_score:
            info_lines.append("\n<bold>Performance</bold>")
            info_lines.append(f"Score: {bot_score['total_score']:.1f} | K/D: {bot_score['kills']}/{bot_score['deaths']}")
            info_lines.append(f"Accuracy: {bot_score['hit_rate']:.0%}")

        self.bot_info_text.text = "\n".join(info_lines)

    def _get_visible_bots(self, selected_bot, current_state):
        """Get visible bots using simplified distance-based visibility."""
        visible_bots = []
        max_range = 15.0
        
        for bot in current_state.get("bots", []):
            if bot["id"] == selected_bot["id"] or not bot["alive"]:
                continue
                
            dx = bot["x"] - selected_bot["x"]
            dy = bot["y"] - selected_bot["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= max_range:
                bearing = math.degrees(math.atan2(dx, dy))
                if bearing < 0:
                    bearing += 360
                visible_bots.append((bot, distance, bearing))
        
        # Sort by distance (closest first)
        visible_bots.sort(key=lambda x: x[1])
        return visible_bots

    def _get_nearby_projectiles(self, selected_bot, current_state):
        """Get nearby projectiles using simplified distance-based detection."""
        nearby_projectiles = []
        max_range = 15.0
        
        for proj in current_state.get("projectiles", []):
            dx = proj["x"] - selected_bot["x"]
            dy = proj["y"] - selected_bot["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= max_range:
                bearing = math.degrees(math.atan2(dx, dy))
                if bearing < 0:
                    bearing += 360
                nearby_projectiles.append((proj, distance, bearing))
        
        # Sort by distance (closest first)
        nearby_projectiles.sort(key=lambda x: x[1])
        return nearby_projectiles

    def _get_visible_walls(self, selected_bot, current_state):
        """Get visible walls (simplified - just return empty for now)."""
        # For simplicity, return empty list - walls are static anyway
        return []


def run_3d_viewer(battle_file: str):
    """Launch 3D viewer with a saved battle JSON file."""
    print(f"\n=== 3D Battle Viewer: {battle_file} ===")

    try:
        with open(battle_file, "r") as f:
            battle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Battle file '{battle_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{battle_file}'")
        return

    print("Launching 3D viewer...")
    print("Controls:")
    print("  SPACE = Play/Pause")
    print("  ←/→ = Step frame by frame")
    print("  +/- = Adjust speed")
    print("  R = Reset to start")
    print("  F = Toggle FOV display")
    print("  T = Toggle projectile trails")
    print("  C = Cycle camera modes (Overview/Follow/FPS)")
    print("  Q/ESC = Quit")
    print("  Click bots to select them")

    # Launch 3D viewer
    viewer = Battle3DViewer(battle_data)
    viewer.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 3dgraphics.py <battle_log.json>")
        sys.exit(1)

    battle_file = sys.argv[1]
    run_3d_viewer(battle_file)
