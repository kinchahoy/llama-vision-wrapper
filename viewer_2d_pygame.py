"""
Pygame Battle Viewer with Timeline Scrubbing
Interactive visualization of battle simulations with playback controls.
"""

import pygame
import math
import json
from typing import Dict, List, Tuple, Optional

# Import visibility system to use same logic as bot programs
try:
    from llm_bot_controller import PythonLLMController
except ImportError:
    try:
        from python_llm import PythonLLMController  # Backward compatibility
    except ImportError:
        PythonLLMController = None


class BattleViewer:
    """Interactive pygame viewer for battle simulations."""

    # Display configuration
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700
    ARENA_DISPLAY_SIZE = 600
    CONTROL_PANEL_HEIGHT = 100

    # Colors (RGB)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (64, 64, 64)

    # Team colors
    TEAM_0_COLOR = (64, 128, 255)  # Blue
    TEAM_1_COLOR = (255, 64, 64)  # Red
    PROJECTILE_COLOR = (255, 255, 0)  # Yellow

    # Health bar colors
    HEALTH_HIGH = (0, 255, 0)  # Green
    HEALTH_MID = (255, 255, 0)  # Yellow
    HEALTH_LOW = (255, 0, 0)  # Red
    HEALTH_BG = (50, 50, 50)  # Dark gray

    def __init__(self, battle_data: Dict):
        """Initialize viewer with battle data."""
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
        self.scale_x = self.ARENA_DISPLAY_SIZE / self.arena_width
        self.scale_y = self.ARENA_DISPLAY_SIZE / self.arena_height

        # Display offsets
        self.arena_offset_x = 50
        self.arena_offset_y = 50

        # UI state
        self.dragging_scrubber = False
        self.show_fov = False
        self.selected_bot = None

        # Initialize visibility system (same as bot programs use)
        if PythonLLMController is not None:
            self.llm_controller = PythonLLMController()
        else:
            self.llm_controller = None

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("LLM Battle Sim Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)

    def run(self):
        """Main viewer loop."""
        running = True

        while running:
            dt = self.clock.tick(self.target_fps) / 1000.0

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keypress(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouse_up(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)

            # Update playback
            if self.playing:
                self._update_playback(dt)

            # Render
            self._render()
            pygame.display.flip()

        pygame.quit()

    def _handle_keypress(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == pygame.K_q:
            return False
        elif key == pygame.K_SPACE:
            self.playing = not self.playing
        elif key == pygame.K_r:
            self.current_frame = 0
            self.playing = False
        elif key == pygame.K_LEFT:
            self.current_frame = max(0, self.current_frame - 1)
            self.playing = False
        elif key == pygame.K_RIGHT:
            self.current_frame = min(len(self.timeline) - 1, self.current_frame + 1)
            self.playing = False
        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:  # + key
            self.playback_speed = min(5.0, self.playback_speed * 1.5)
        elif key == pygame.K_MINUS:
            self.playback_speed = max(0.1, self.playback_speed / 1.5)
        elif key == pygame.K_f:
            self.show_fov = not self.show_fov

        return True

    def _handle_mouse_down(self, pos: Tuple[int, int]):
        """Handle mouse button press."""
        x, y = pos

        # Check if clicking on timeline scrubber
        scrubber_y = self.WINDOW_HEIGHT - 50
        if scrubber_y - 10 <= y <= scrubber_y + 10:
            scrubber_x_start = 50
            scrubber_width = self.WINDOW_WIDTH - 100
            if scrubber_x_start <= x <= scrubber_x_start + scrubber_width:
                self.dragging_scrubber = True
                self._scrub_to_position(x, scrubber_x_start, scrubber_width)
                return

        # Check if clicking on a bot
        if self._is_in_arena(x, y):
            self._handle_bot_click(x, y)

    def _handle_mouse_up(self, pos: Tuple[int, int]):
        """Handle mouse button release."""
        self.dragging_scrubber = False

    def _handle_mouse_motion(self, pos: Tuple[int, int]):
        """Handle mouse movement."""
        if self.dragging_scrubber:
            x, y = pos
            scrubber_x_start = 50
            scrubber_width = self.WINDOW_WIDTH - 100
            self._scrub_to_position(x, scrubber_x_start, scrubber_width)

    def _scrub_to_position(
        self, mouse_x: int, scrubber_start: int, scrubber_width: int
    ):
        """Scrub timeline to mouse position."""
        relative_x = mouse_x - scrubber_start
        progress = max(0, min(1, relative_x / scrubber_width))
        target_frame = int(progress * (len(self.timeline) - 1))
        self.current_frame = target_frame
        self.playing = False

    def _handle_bot_click(self, mouse_x: int, mouse_y: int):
        """Handle clicking on bots in the arena."""
        if self.current_frame >= len(self.timeline):
            return

        current_state = self.timeline[int(self.current_frame)]

        # Convert mouse coordinates to centered arena coordinates
        arena_x = (
            mouse_x - self.arena_offset_x
        ) / self.scale_x - self.arena_width / 2
        # Flip Y coordinate and convert to centered
        arena_y = (
            self.arena_height / 2 - (mouse_y - self.arena_offset_y) / self.scale_y
        )

        # Find closest bot
        closest_bot = None
        closest_dist = float("inf")

        for bot in current_state.get("bots", []):
            bot_x, bot_y = bot["x"], bot["y"]
            dist = math.sqrt((arena_x - bot_x) ** 2 + (arena_y - bot_y) ** 2)
            if dist < 1.0 and dist < closest_dist:  # Within 1 meter
                closest_bot = bot
                closest_dist = dist

        self.selected_bot = closest_bot
        if closest_bot:
            self.playing = False  # Pause to show bot info

    def _is_in_arena(self, x: int, y: int) -> bool:
        """Check if coordinates are within the arena display area."""
        return (
            self.arena_offset_x <= x <= self.arena_offset_x + self.ARENA_DISPLAY_SIZE
            and self.arena_offset_y
            <= y
            <= self.arena_offset_y + self.ARENA_DISPLAY_SIZE
        )

    def _update_playback(self, dt: float):
        """Update playback position."""
        if not self.timeline:
            return

        # Calculate frames to advance based on speed
        frames_per_second = 10  # 10Hz logging rate
        frame_advance = frames_per_second * self.playback_speed * dt

        self.current_frame += frame_advance

        # Wrap or stop at end
        if self.current_frame >= len(self.timeline):
            self.current_frame = len(self.timeline) - 1
            self.playing = False

    def _render(self):
        """Render the current frame."""
        self.screen.fill(self.BLACK)

        if not self.timeline:
            self._draw_text(
                "No battle data to display",
                (self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2),
                center=True,
            )
            return

        current_state = self._get_current_state()

        # Draw arena
        self._draw_arena()

        # Draw bots
        self._draw_bots(current_state)

        # Draw projectiles
        self._draw_projectiles(current_state)

        # Draw walls
        self._draw_walls(current_state)

        # Draw UI
        self._draw_ui(current_state)

    def _get_current_state(self) -> Dict:
        """Get current frame state with interpolation."""
        if not self.timeline:
            return {}

        frame_idx = int(self.current_frame)
        frame_idx = max(0, min(len(self.timeline) - 1, frame_idx))

        return self.timeline[frame_idx]

    def _draw_arena(self):
        """Draw the arena background and boundaries."""
        # Arena background
        arena_rect = (
            self.arena_offset_x,
            self.arena_offset_y,
            self.ARENA_DISPLAY_SIZE,
            self.ARENA_DISPLAY_SIZE,
        )
        pygame.draw.rect(self.screen, self.DARK_GRAY, arena_rect)

        # Arena border
        pygame.draw.rect(self.screen, self.WHITE, arena_rect, 2)

        # Grid lines (every 5 meters)
        for i in range(1, int(self.arena_width // 5)):
            x = self.arena_offset_x + i * 5 * self.scale_x
            pygame.draw.line(
                self.screen,
                self.GRAY,
                (x, self.arena_offset_y),
                (x, self.arena_offset_y + self.ARENA_DISPLAY_SIZE),
            )

        for i in range(1, int(self.arena_height // 5)):
            y = self.arena_offset_y + i * 5 * self.scale_y
            pygame.draw.line(
                self.screen,
                self.GRAY,
                (self.arena_offset_x, y),
                (self.arena_offset_x + self.ARENA_DISPLAY_SIZE, y),
            )

    def _draw_bots(self, state: Dict):
        """Draw bots with health bars and heading indicators."""
        for bot in state.get("bots", []):
            # Convert centered sim coordinates to screen coordinates
            screen_x = (
                self.arena_offset_x + (bot["x"] + self.arena_width / 2) * self.scale_x
            )
            screen_y = (
                self.arena_offset_y
                + (self.arena_height / 2 - bot["y"]) * self.scale_y
            )

            # Bot color based on team
            color = self.TEAM_0_COLOR if bot["team"] == 0 else self.TEAM_1_COLOR

            # Highlight if selected
            if self.selected_bot and self.selected_bot["id"] == bot["id"]:
                pygame.draw.circle(
                    self.screen, self.WHITE, (int(screen_x), int(screen_y)), 12, 2
                )

            # Draw bot body
            bot_radius = int(0.4 * self.scale_x)  # 0.4m radius
            pygame.draw.circle(
                self.screen, color, (int(screen_x), int(screen_y)), bot_radius
            )
            pygame.draw.circle(
                self.screen, self.WHITE, (int(screen_x), int(screen_y)), bot_radius, 1
            )

            # Draw heading indicator (flip Y direction for pygame coordinates)
            heading_rad = math.radians(
                bot["theta"]
            )  # bot['theta'] is in degrees from battle_arena
            end_x = screen_x + math.cos(heading_rad) * bot_radius * 1.5
            end_y = (
                screen_y - math.sin(heading_rad) * bot_radius * 1.5
            )  # negative sin for flipped Y
            pygame.draw.line(
                self.screen,
                self.WHITE,
                (int(screen_x), int(screen_y)),
                (int(end_x), int(end_y)),
                2,
            )

            # Draw FOV if enabled and this is the selected bot
            if (
                self.show_fov
                and self.selected_bot
                and self.selected_bot["id"] == bot["id"]
            ):
                self._draw_fov(screen_x, screen_y, heading_rad, color)

            # Draw health bar
            self._draw_health_bar(screen_x, screen_y - bot_radius - 15, bot["hp"])

            # Draw bot ID
            id_text = self.small_font.render(str(bot["id"]), True, self.WHITE)
            self.screen.blit(
                id_text, (int(screen_x - 5), int(screen_y + bot_radius + 5))
            )

    def _draw_fov(
        self, x: float, y: float, heading: float, color: Tuple[int, int, int]
    ):
        """Draw field of view as a 120-degree arc sector."""
        fov_range = 15 * self.scale_x  # 15m range
        fov_angle = math.radians(120)  # 120° FOV

        # Calculate arc angles
        start_angle = heading - fov_angle / 2
        end_angle = heading + fov_angle / 2

        # Create arc points for a smooth arc
        arc_points = [(x, y)]  # Start from bot center

        # Generate points along the arc (every 5 degrees for smoothness)
        angle_step = math.radians(5)
        current_angle = start_angle

        while current_angle <= end_angle:
            arc_x = x + math.cos(current_angle) * fov_range
            arc_y = (
                y - math.sin(current_angle) * fov_range
            )  # Flip Y for pygame coordinates
            arc_points.append((arc_x, arc_y))
            current_angle += angle_step

        # Ensure we include the end angle
        if current_angle - angle_step < end_angle:
            arc_x = x + math.cos(end_angle) * fov_range
            arc_y = y - math.sin(end_angle) * fov_range
            arc_points.append((arc_x, arc_y))

        # Draw semi-transparent FOV arc sector
        fov_surface = pygame.Surface(
            (self.ARENA_DISPLAY_SIZE, self.ARENA_DISPLAY_SIZE), pygame.SRCALPHA
        )
        fov_surface.set_alpha(80)

        # Convert points to surface coordinates
        surface_points = [
            (p[0] - self.arena_offset_x, p[1] - self.arena_offset_y) for p in arc_points
        ]

        # Draw filled arc sector
        if len(surface_points) >= 3:
            pygame.draw.polygon(fov_surface, (*color, 80), surface_points)

        # Draw arc outline for clarity
        if len(surface_points) >= 2:
            pygame.draw.lines(
                fov_surface, color, False, surface_points[1:], 2
            )  # Arc edge
            pygame.draw.line(
                fov_surface, color, surface_points[0], surface_points[1], 1
            )  # Left ray
            pygame.draw.line(
                fov_surface, color, surface_points[0], surface_points[-1], 1
            )  # Right ray

        self.screen.blit(fov_surface, (self.arena_offset_x, self.arena_offset_y))

    def _draw_health_bar(self, x: float, y: float, hp: int):
        """Draw health bar above bot."""
        bar_width = 20
        bar_height = 4

        # Background
        bg_rect = (int(x - bar_width // 2), int(y), bar_width, bar_height)
        pygame.draw.rect(self.screen, self.HEALTH_BG, bg_rect)

        # Health bar
        hp_ratio = max(0, min(1, hp / 100))
        hp_width = int(bar_width * hp_ratio)

        if hp_ratio > 0.6:
            hp_color = self.HEALTH_HIGH
        elif hp_ratio > 0.3:
            hp_color = self.HEALTH_MID
        else:
            hp_color = self.HEALTH_LOW

        if hp_width > 0:
            hp_rect = (int(x - bar_width // 2), int(y), hp_width, bar_height)
            pygame.draw.rect(self.screen, hp_color, hp_rect)

    def _draw_projectiles(self, state: Dict):
        """Draw projectiles."""
        projectiles = state.get("projectiles", [])

        # Draw projectiles
        for i, proj in enumerate(projectiles):
            screen_x = (
                self.arena_offset_x + (proj["x"] + self.arena_width / 2) * self.scale_x
            )
            screen_y = (
                self.arena_offset_y
                + (self.arena_height / 2 - proj["y"]) * self.scale_y
            )

            # Draw projectile with team color tint
            proj_color = self.PROJECTILE_COLOR
            outline_color = self.WHITE
            if proj.get("team") == 0:
                proj_color = (255, 200, 100)  # Yellow-blue tint
                outline_color = (100, 150, 255)  # Blue outline for team 0
            elif proj.get("team") == 1:
                proj_color = (255, 150, 150)  # Yellow-red tint
                outline_color = (255, 100, 100)  # Red outline for team 1

            # Draw projectile body
            pygame.draw.circle(
                self.screen, proj_color, (int(screen_x), int(screen_y)), 4
            )
            pygame.draw.circle(
                self.screen, outline_color, (int(screen_x), int(screen_y)), 4, 1
            )

            # Draw velocity indicator (small arrow)
            if "vx" in proj and "vy" in proj:
                vel_scale = 3
                end_x = (
                    screen_x + proj["vx"] * vel_scale / 6.0
                )  # Normalize to projectile speed
                end_y = screen_y - proj["vy"] * vel_scale / 6.0  # Y is inverted
                pygame.draw.line(
                    self.screen,
                    self.WHITE,
                    (int(screen_x), int(screen_y)),
                    (int(end_x), int(end_y)),
                    1,
                )

    def _draw_walls(self, state: Dict):
        """Draw interior walls/obstacles from metadata."""
        wall_color = (100, 100, 100)
        walls_data = self.metadata.get("walls", [])

        for wall_def in walls_data:
            center_x, center_y, w, h, angle_deg = wall_def

            # Create a surface for the wall, scaled to display size
            wall_w_px = w * self.scale_x
            wall_h_px = h * self.scale_y
            wall_surface = pygame.Surface((wall_w_px, wall_h_px), pygame.SRCALPHA)
            wall_surface.fill(wall_color)
            pygame.draw.rect(wall_surface, self.WHITE, wall_surface.get_rect(), 1)

            # Pygame rotates counter-clockwise, same as our angle definition
            rotated_surface = pygame.transform.rotate(wall_surface, angle_deg)
            new_rect = rotated_surface.get_rect()

            # Convert center to screen coordinates
            screen_cx = (
                self.arena_offset_x + (center_x + self.arena_width / 2) * self.scale_x
            )
            screen_cy = (
                self.arena_offset_y
                + (self.arena_height / 2 - center_y) * self.scale_y
            )
            new_rect.center = (screen_cx, screen_cy)

            self.screen.blit(rotated_surface, new_rect)

    def _draw_ui(self, state: Dict):
        """Draw UI elements (timeline, controls, info)."""
        # Timeline scrubber
        self._draw_timeline()

        # Control instructions
        controls_y = self.WINDOW_HEIGHT - 30
        self._draw_text(
            "Controls: SPACE=Play/Pause, ←→=Step, R=Reset, +/-=Speed, F=FOV, Q=Quit",
            (10, controls_y),
        )

        # Battle info
        info_x = self.ARENA_DISPLAY_SIZE + 100
        info_y = 50

        # Current time and frame
        time_info = f"Time: {state.get('time', 0):.1f}s"
        frame_info = f"Frame: {int(self.current_frame)}/{len(self.timeline) - 1}"
        speed_info = f"Speed: {self.playback_speed:.1f}x"

        self._draw_text(time_info, (info_x, info_y))
        self._draw_text(frame_info, (info_x, info_y + 25))
        self._draw_text(speed_info, (info_x, info_y + 50))

        # Battle metadata and scoring
        if self.metadata:
            winner = self.metadata.get("winner", "unknown")
            reason = self.metadata.get("reason", "unknown")
            self._draw_text(f"Winner: {winner} ({reason})", (info_x, info_y + 100))

        # Display MVP and performance highlights
        summary = self.battle_data.get("summary", {})
        if summary:
            mvp = summary.get("mvp", {})
            if mvp.get("bot_id") is not None:
                mvp_text = f"MVP: Bot {mvp['bot_id']} (Team {mvp['team']}) - {mvp['score']:.1f} pts"
                self._draw_text(
                    mvp_text,
                    (info_x, info_y + 125),
                    self.TEAM_0_COLOR if mvp["team"] == 0 else self.TEAM_1_COLOR,
                )

            # Overall battle stats
            intensity = summary.get("battle_intensity", 0)
            accuracy = summary.get("overall_accuracy", 0)
            self._draw_text(
                f"Intensity: {intensity:.1f} shots/sec", (info_x, info_y + 175)
            )
            self._draw_text(f"Overall Accuracy: {accuracy:.1%}", (info_x, info_y + 200))

        # Selected bot info - draw on left side to avoid overlap
        if self.selected_bot:
            self._draw_selected_bot_info(10, 50)

        # Events in current frame
        events = state.get("events", [])
        if events:
            self._draw_events(info_x, info_y + 250, events)

    def _draw_timeline(self):
        """Draw timeline scrubber."""
        scrubber_y = self.WINDOW_HEIGHT - 50
        scrubber_x_start = 50
        scrubber_width = self.WINDOW_WIDTH - 100
        scrubber_height = 20

        # Timeline background
        timeline_rect = (
            scrubber_x_start,
            scrubber_y - scrubber_height // 2,
            scrubber_width,
            scrubber_height,
        )
        pygame.draw.rect(self.screen, self.DARK_GRAY, timeline_rect)
        pygame.draw.rect(self.screen, self.WHITE, timeline_rect, 1)

        # Progress bar
        if len(self.timeline) > 1:
            progress = self.current_frame / (len(self.timeline) - 1)
            progress_width = int(scrubber_width * progress)
            progress_rect = (
                scrubber_x_start,
                scrubber_y - scrubber_height // 2,
                progress_width,
                scrubber_height,
            )
            pygame.draw.rect(self.screen, self.TEAM_0_COLOR, progress_rect)

        # Current position indicator
        if len(self.timeline) > 1:
            indicator_x = scrubber_x_start + int(scrubber_width * progress)
            pygame.draw.line(
                self.screen,
                self.WHITE,
                (indicator_x, scrubber_y - scrubber_height // 2),
                (indicator_x, scrubber_y + scrubber_height // 2),
                3,
            )

    def _draw_selected_bot_info(self, x: int, y: int):
        """Draw info panel for selected bot with function info and performance stats."""
        bot = self.selected_bot
        bot_id = bot["id"]

        # Draw semi-transparent background for bot info panel
        panel_width = 320
        panel_height = 400
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((20, 20, 20))
        self.screen.blit(panel_surface, (x, y))

        # Header
        self._draw_text(
            f"Bot {bot_id} (Team {bot['team']})", (x + 5, y + 5), self.WHITE
        )

        # Function info from bot_functions metadata
        bot_functions = self.battle_data.get("summary", {}).get("bot_functions", {})
        bot_func_data = bot_functions.get(str(bot_id), {})

        personality = bot_func_data.get("personality", "unknown")
        version = bot_func_data.get("version", "N/A")

        # Function name and LLM info
        self._draw_text(
            f"Function: {personality}_combat_v{version}", (x + 5, y + 30), self.WHITE
        )
        self._draw_text(
            f"LLM Generated: Python Function", (x + 5, y + 50), color=(150, 150, 255)
        )

        # Current signal with description
        current_signal = getattr(
            bot, "signal", "none"
        )  # Get signal from bot data if available
        signal_color = (255, 200, 100) if current_signal != "none" else (100, 100, 100)

        # Import signal definitions for descriptions
        signal_descriptions = {
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

        signal_desc = signal_descriptions.get(current_signal, "Unknown signal")
        self._draw_text(
            f'Signal: "{current_signal}"', (x + 5, y + 70), color=signal_color
        )

        # Show signal description in smaller text
        desc_surface = self.tiny_font.render(signal_desc, True, (150, 150, 150))
        self.screen.blit(desc_surface, (x + 10, y + 85))

        # Current state
        self._draw_text("--- Current State ---", (x + 5, y + 110), self.WHITE)
        self._draw_text(f"Position: ({bot['x']:.1f}, {bot['y']:.1f})", (x + 5, y + 130))
        self._draw_text(f"Heading: {bot['theta']:.0f}°", (x + 5, y + 150))
        self._draw_text(
            f"Speed: {math.sqrt(bot['vx'] ** 2 + bot['vy'] ** 2):.1f} m/s",
            (x + 5, y + 170),
        )
        self._draw_text(f"Health: {bot['hp']} HP", (x + 5, y + 190))

        # Visible units, projectiles, and walls (smaller font)
        current_state = self.timeline[int(self.current_frame)]
        visible_bots = self._get_visible_bots(bot, current_state)
        nearby_projectiles = self._get_nearby_projectiles(bot, current_state)
        visible_walls = self._get_visible_walls(bot, current_state)

        # Show counts of visible objects
        total_units = len(visible_bots)
        friends_count = len(
            [
                vis_bot
                for vis_bot, _, _ in visible_bots
                if vis_bot["team"] == bot["team"]
            ]
        )
        enemies_count = total_units - friends_count
        projectiles_count = len(nearby_projectiles)
        walls_count = len(visible_walls)

        self._draw_text(
            f"--- Tactical Situation ({friends_count}F, {enemies_count}E, {projectiles_count}P, {walls_count}W) ---",
            (x + 5, y + 220),
            self.WHITE,
        )

        # Left column: Visible units
        left_col_x = x + 5
        right_col_x = x + 165
        base_y = y + 240

        # Units header and content
        units_surface = self.small_font.render("Units:", True, self.WHITE)
        self.screen.blit(units_surface, (left_col_x, base_y))

        if visible_bots:
            for i, (vis_bot, distance, bearing) in enumerate(
                visible_bots
            ):  # Show ALL visible units
                if vis_bot["team"] == bot["team"]:
                    unit_type = "F"  # Friend (abbreviated)
                    color = (100, 255, 100)  # Green
                    # Include friend's signal
                    signal = vis_bot.get("signal", "none")
                    signal_part = f" [{signal}]" if signal != "none" else ""
                    vis_text = f"{unit_type}{vis_bot['id']}: {distance:.1f}m@{bearing:.0f}°{signal_part}"
                else:
                    unit_type = "E"  # Enemy (abbreviated)
                    color = (255, 100, 100)  # Red
                    vis_text = (
                        f"{unit_type}{vis_bot['id']}: {distance:.1f}m@{bearing:.0f}°"
                    )

                vis_surface = self.tiny_font.render(vis_text, True, color)
                self.screen.blit(vis_surface, (left_col_x, base_y + 15 + i * 12))
        else:
            vis_surface = self.tiny_font.render("None", True, (150, 150, 150))
            self.screen.blit(vis_surface, (left_col_x, base_y + 15))

        # Right column: Projectiles and Walls
        proj_surface = self.small_font.render("Objects:", True, self.WHITE)
        self.screen.blit(proj_surface, (right_col_x, base_y))

        line_offset = 15

        # Draw projectiles
        if nearby_projectiles:
            for i, (proj, distance, bearing) in enumerate(
                nearby_projectiles
            ):  # Show ALL visible projectiles
                proj_text = f"P: {distance:.1f}m@{bearing:.0f}°"
                proj_surface = self.tiny_font.render(
                    proj_text, True, (255, 255, 100)
                )  # Yellow
                self.screen.blit(proj_surface, (right_col_x, base_y + line_offset))
                line_offset += 12

        # Draw walls
        if visible_walls:
            for i, (wall, distance, bearing) in enumerate(visible_walls):
                wall_text = f"W: {distance:.1f}m@{bearing:.0f}°"
                wall_surface = self.tiny_font.render(
                    wall_text, True, (150, 150, 150)
                )  # Gray
                self.screen.blit(wall_surface, (right_col_x, base_y + line_offset))
                line_offset += 12

        # Show "None" only if no projectiles or walls
        if not nearby_projectiles and not visible_walls:
            none_surface = self.tiny_font.render("None", True, (150, 150, 150))
            self.screen.blit(none_surface, (right_col_x, base_y + 15))

        # Performance stats from summary
        summary = self.battle_data.get("summary", {})
        bot_scores = summary.get("bot_scores", [])

        # Find this bot's score data
        bot_score = None
        for score in bot_scores:
            if score["bot_id"] == bot_id:
                bot_score = score
                break

        if bot_score:
            perf_y = y + 305  # Adjusted down due to added signal description
            self._draw_text("--- Performance ---", (x + 5, perf_y), self.WHITE)
            self._draw_text(
                f"Score: {bot_score['total_score']:.1f} pts", (x + 5, perf_y + 20)
            )
            self._draw_text(
                f"Accuracy: {bot_score['hit_rate']:.1%} ({bot_score['shots_hit']}/{bot_score['shots_fired']})",
                (x + 5, perf_y + 40),
            )
            self._draw_text(
                f"Damage: {bot_score['damage_dealt']:.0f} dealt, {bot_score['damage_taken']:.0f} taken",
                (x + 5, perf_y + 60),
            )
            self._draw_text(
                f"K/D: {bot_score['kills']}/{bot_score['deaths']}", (x + 5, perf_y + 80)
            )

    def _get_visible_objects_for_bot(self, selected_bot, current_state):
        """Get visible objects for a bot using the same system as bot programs."""
        if self.llm_controller is None:
            # Fallback to simple distance-based visibility if LLM controller unavailable
            return self._get_visible_objects_fallback(selected_bot, current_state)

        # We need to reconstruct arena state from the logged state.
        # Create a minimal mock that provides the necessary interface.
        # Pass `self` (the viewer instance) to access metadata.
        from battle_arena import Arena

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
                    corners = [
                        (-hw, -hh),
                        (hw, -hh),
                        (hw, hh),
                        (-hw, hh),
                    ]
                    rotated_corners = [
                        (p[0] * c - p[1] * s, p[0] * s + p[1] * c) for p in corners
                    ]
                    abs_corners = [(p[0] + cx, p[1] + cy) for p in rotated_corners]

                    # Create a single polygon shape instead of segments
                    wall_shape = MockWallShape(abs_corners)
                    self.wall_bodies.append((None, wall_shape))

            def _is_bot_alive(self, bot_id):
                return bot_id in self.bot_data

        # Create mock arena and use LLM controller's visibility system
        mock_arena = MockArena(self, current_state)
        bot_id = selected_bot["id"]

        try:
            visible_objects = self.llm_controller.generate_visible_objects(
                mock_arena, bot_id
            )
            return visible_objects
        except Exception as e:
            # Fallback if visibility system fails
            print(f"Warning: Visibility system failed, using fallback: {e}")
            return self._get_visible_objects_fallback(selected_bot, current_state)

    def _get_visible_objects_fallback(self, selected_bot, current_state):
        """Fallback visibility system using simple distance checks."""
        visible_objects = []
        max_range = 15.0

        # Check bots
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

        # Check projectiles
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
        """Calculate which bots are visible to the selected bot using same visibility system as bot programs."""
        # Use the same visibility system that bot programs use
        visible_objects = self._get_visible_objects_for_bot(selected_bot, current_state)

        # Filter for bots only and convert to expected format
        visible_bots = []
        for obj in visible_objects:
            if obj["type"] in ["enemy", "friend"]:
                # Find the actual bot data from current_state
                bot_data = None
                for bot in current_state.get("bots", []):
                    if bot["id"] == obj.get("id"):
                        bot_data = bot
                        break

                if bot_data:
                    visible_bots.append((bot_data, obj["distance"], obj["angle"]))

        # Sort by distance (closest first)
        visible_bots.sort(key=lambda x: x[1])
        return visible_bots

    def _get_nearby_projectiles(self, selected_bot, current_state):
        """Calculate which projectiles are near the selected bot using same visibility system as bot programs."""
        # Use the same visibility system that bot programs use
        visible_objects = self._get_visible_objects_for_bot(selected_bot, current_state)

        # Filter for projectiles only and convert to expected format
        nearby_projectiles = []
        for obj in visible_objects:
            if obj["type"] == "projectile":
                # Create projectile data structure matching current format
                proj_data = {
                    "x": obj["x"],
                    "y": obj["y"],
                    "velocity_x": obj.get("velocity_x", 0),
                    "velocity_y": obj.get("velocity_y", 0),
                    "team": obj.get("team", "unknown"),
                    "ttl": obj.get("ttl", 0),
                }
                nearby_projectiles.append((proj_data, obj["distance"], obj["angle"]))

        # Sort by distance (closest first)
        nearby_projectiles.sort(key=lambda x: x[1])
        return nearby_projectiles

    def _get_visible_walls(self, selected_bot, current_state=None):
        """Calculate which walls are visible to the selected bot using same visibility system as bot programs."""
        # Use the same visibility system that bot programs use
        if current_state is None:
            current_state = self.timeline[int(self.current_frame)]

        visible_objects = self._get_visible_objects_for_bot(selected_bot, current_state)

        # Filter for walls only and convert to expected format
        visible_walls = []
        for obj in visible_objects:
            if obj["type"] == "wall":
                # The wall object itself is not used, just distance and angle for display
                visible_walls.append((obj, obj["distance"], obj["angle"]))

        # Sort by distance (closest first)
        visible_walls.sort(key=lambda x: x[1])
        return visible_walls

    def _draw_events(self, x: int, y: int, events: List[Dict]):
        """Draw recent events."""
        self._draw_text("Recent Events:", (x, y), self.WHITE)

        for i, event in enumerate(events[-5:]):  # Show last 5 events
            event_y = y + 20 + i * 15
            event_type = event.get("type", "unknown")

            if event_type == "shot":
                text = f"Bot {event['bot_id']} fired"
            elif event_type == "hit":
                text = f"Bot {event['projectile_shooter']} hit Bot {event['target']}"
            elif event_type == "death":
                text = f"Bot {event['bot_id']} destroyed"
            else:
                text = f"{event_type}: {event}"

            self._draw_text(text, (x, event_y), color=self.LIGHT_GRAY)

    def _draw_text(
        self,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        center: bool = False,
    ):
        """Draw text at position."""
        if color is None:
            color = self.WHITE

        text_surface = self.font.render(text, True, color)

        if center:
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, pos)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graphics.py <battle_log.json>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename) as f:
            battle_data = json.load(f)
        viewer = BattleViewer(battle_data)
        viewer.run()
    except FileNotFoundError:
        print(f"Battle log file not found: {filename}")
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {filename}")
    except Exception as e:
        print(f"Error loading battle data: {e}")
