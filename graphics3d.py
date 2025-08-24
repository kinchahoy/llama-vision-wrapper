"""
Panda3D Battle Viewer
Interactive 3D visualization of battle simulations using the Panda3D engine.
"""

import math
import json
import sys
from typing import Dict, List, Tuple, Optional

import simplepbr
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectFrame, DirectSlider, DirectButton, OnscreenText
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    NodePath,
    TextNode,
    LPoint3f,
    LVecBase3f,
    LineSegs,
    CardMaker,
    CollisionTraverser,
    CollisionHandlerQueue,
    CollisionRay,
    CollisionNode,
    GeomNode,
    BitMask32,
)


class Battle3DViewer(ShowBase):
    """Interactive 3D viewer for battle simulations using Panda3D."""

    def __init__(self, battle_data: Dict):
        """Initialize the Panda3D viewer."""
        ShowBase.__init__(self)
        simplepbr.init()

        self.battle_data = battle_data
        self.timeline = battle_data["timeline"]
        self.metadata = battle_data["metadata"]

        # Playback state
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0

        # Arena dimensions
        arena_size = self.metadata.get("arena_size", [20, 20])
        self.arena_width, self.arena_height = arena_size

        # UI state
        self.show_fov = False
        self.selected_bot = None

        # Panda3D object management
        self.bot_nodepaths = {}
        self.projectile_nodepaths = []
        self.fov_nodepath = None

        # Setup
        self._setup_scene()
        self._setup_ui()
        self._setup_controls()
        self._setup_mouse_picking()

        # Start the main update loop
        self.taskMgr.add(self._update_task, "update_battle_task")

    def _setup_scene(self):
        """Set up the 3D scene, camera, and lighting."""
        self.setBackgroundColor(0.1, 0.1, 0.1, 1)
        self.disableMouse()  # Disable default camera controls

        # Set up an isometric-style camera
        arena_diagonal = math.sqrt(self.arena_width**2 + self.arena_height**2)
        self.cam.setPos(0, -arena_diagonal * 1.2, arena_diagonal * 1.1)
        self.cam.lookAt(0, 0, 0)

        # Lighting
        alight = AmbientLight("ambient")
        alight.setColor((0.4, 0.4, 0.4, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        dlight = DirectionalLight("directional")
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(30, -60, 0)
        self.render.setLight(dlnp)

        # Arena floor
        cm = CardMaker("floor")
        cm.setFrame(
            -self.arena_width / 2,
            self.arena_width / 2,
            -self.arena_height / 2,
            self.arena_height / 2,
        )
        floor = self.render.attachNewNode(cm.generate())
        floor.setP(-90)  # Rotate to be flat on the XY plane
        floor.setColor(0.3, 0.3, 0.3, 1)

        self._create_walls()

    def _create_walls(self):
        """Create interior walls matching the 2D version."""
        width = self.arena_width
        height = self.arena_height
        wall_height = 1.5

        # Interior walls data
        walls_data = [
            ((width * 0.2, height * 0.7), (width * 0.2 + 10, height * 0.7)),
            ((width * 0.4, height * 0.3), (width * 0.4, height * 0.3 + 8)),
            ((width * 0.6, height * 0.2), (width * 0.6 + 9, height * 0.2)),
            ((width * 0.8, height * 0.6), (width * 0.8, height * 0.6 + 6)),
        ]

        for start, end in walls_data:
            # Convert to centered coordinates
            start_x, start_y = start[0] - width / 2, start[1] - height / 2
            end_x, end_y = end[0] - width / 2, end[1] - height / 2

            # Use LineSegs to draw a thick line for the wall
            lines = LineSegs()
            lines.setThickness(5)
            lines.setColor(0.6, 0.6, 0.6, 1)
            lines.moveTo(start_x, start_y, 0)
            lines.drawTo(end_x, end_y, 0)

            # Create a simple mesh for the wall's height
            cm = CardMaker(f"wall_{start}_{end}")
            wall_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            cm.setFrame(0, wall_length, 0, wall_height)
            wall_node = self.render.attachNewNode(cm.generate())
            wall_node.setColor(0.5, 0.5, 0.5, 1)
            wall_node.setPos(start_x, start_y, 0)
            wall_node.lookAt(end_x, end_y, 0)

    def _setup_ui(self):
        """Set up the DirectGUI elements for controls and info."""
        # UI Panel
        self.ui_frame = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0.8),
            frameSize=(-1, 1, -1, 1),
            pos=(0, 0, -0.85),
            scale=0.15,
        )

        # Timeline Slider
        self.timeline_slider = DirectSlider(
            parent=self.ui_frame,
            range=(0, len(self.timeline) - 1),
            value=0,
            pageSize=1,
            command=self._on_slider_move,
            pos=(0, 0, 0.5),
            scale=(0.8, 1, 1),
        )

        # Control Buttons
        self.play_pause_btn = DirectButton(
            parent=self.ui_frame,
            text="Play",
            command=self._toggle_play,
            pos=(-0.8, 0, -0.3),
            scale=0.4,
        )
        DirectButton(
            parent=self.ui_frame,
            text="<",
            command=self._step_frame,
            extraArgs=[-1],
            pos=(-0.6, 0, -0.3),
            scale=0.4,
        )
        DirectButton(
            parent=self.ui_frame,
            text=">",
            command=self._step_frame,
            extraArgs=[1],
            pos=(-0.4, 0, -0.3),
            scale=0.4,
        )
        DirectButton(
            parent=self.ui_frame,
            text="Reset",
            command=self._reset_sim,
            pos=(-0.2, 0, -0.3),
            scale=0.4,
        )

        # Info Text
        self.info_text = OnscreenText(
            parent=self.a2dTopRight,
            text="",
            pos=(-0.05, -0.1),
            scale=0.05,
            align=TextNode.ARight,
        )
        self.bot_info_text = OnscreenText(
            parent=self.a2dTopLeft,
            text="Click on a bot to select it",
            pos=(0.05, -0.1),
            scale=0.05,
            align=TextNode.ALeft,
        )

    def _setup_controls(self):
        """Set up keyboard controls."""
        self.accept("space", self._toggle_play)
        self.accept("arrow_left", self._step_frame, [-1])
        self.accept("arrow_right", self._step_frame, [1])
        self.accept("r", self._reset_sim)
        self.accept("f", self._toggle_fov)
        self.accept("escape", sys.exit)
        self.accept("q", sys.exit)
        self.accept("mouse1", self._handle_mouse_click)

    def _setup_mouse_picking(self):
        """Set up the collision system for mouse picking."""
        self.picker = CollisionTraverser()
        self.pq = CollisionHandlerQueue()
        self.pickerNode = CollisionNode("mouseRay")
        self.pickerNP = self.cam.attachNewNode(self.pickerNode)
        self.pickerNode.setFromCollideMask(GeomNode.getDefaultCollideMask())
        self.pickerRay = CollisionRay()
        self.pickerNode.addSolid(self.pickerRay)
        self.picker.addCollider(self.pickerNP, self.pq)

    def _handle_mouse_click(self):
        """Handle mouse clicks for bot selection."""
        if self.mouseWatcherNode.hasMouse():
            mpos = self.mouseWatcherNode.getMouse()
            self.pickerRay.setFromLens(self.camNode, mpos.getX(), mpos.getY())
            self.picker.traverse(self.render)
            if self.pq.getNumEntries() > 0:
                self.pq.sortEntries()
                picked_obj = self.pq.getEntry(0).getIntoNodePath()
                if picked_obj.hasNetTag("bot_id"):
                    bot_id = int(picked_obj.getNetTag("bot_id"))
                    current_state = self._get_current_state()
                    for bot in current_state.get("bots", []):
                        if bot["id"] == bot_id:
                            self.selected_bot = bot
                            self.playing = False
                            self._update_fov_display()
                            break

    def _update_task(self, task):
        """Main update loop."""
        if self.playing:
            dt = globalClock.getDt()
            frame_advance = 10 * self.playback_speed * dt
            self.current_frame += frame_advance
            if self.current_frame >= len(self.timeline) - 1:
                self.current_frame = len(self.timeline) - 1
                self._toggle_play()

        # Ensure slider is updated if frame changes
        self.timeline_slider.setValue(self.current_frame)

        current_state = self._get_current_state()
        self._update_bots(current_state)
        self._update_projectiles(current_state)
        self._update_ui(current_state)
        return task.cont

    def _get_current_state(self) -> Dict:
        """Get the state for the current frame."""
        frame_idx = int(self.current_frame)
        frame_idx = max(0, min(len(self.timeline) - 1, frame_idx))
        return self.timeline[frame_idx]

    def _update_bots(self, state: Dict):
        """Update bot models in the scene."""
        current_bot_ids = {bot["id"] for bot in state.get("bots", []) if bot["alive"]}

        # Remove dead bots
        for bot_id in list(self.bot_nodepaths.keys()):
            if bot_id not in current_bot_ids:
                self.bot_nodepaths[bot_id].removeNode()
                del self.bot_nodepaths[bot_id]

        # Update or create bots
        for bot in state.get("bots", []):
            if not bot["alive"]:
                continue

            bot_id = bot["id"]
            pos = LPoint3f(
                bot["x"] - self.arena_width / 2, bot["y"] - self.arena_height / 2, 0.5
            )

            if bot_id not in self.bot_nodepaths:
                bot_model = self.loader.loadModel("models/misc/sphere")
                bot_model.reparentTo(self.render)
                bot_model.setTag("bot_id", str(bot_id))
                self.bot_nodepaths[bot_id] = bot_model

                # Heading indicator
                heading_indicator = self.loader.loadModel("models/misc/box")
                heading_indicator.reparentTo(bot_model)
                heading_indicator.setPos(0, 0.7, 0)
                heading_indicator.setScale(0.1, 0.5, 0.1)

            np = self.bot_nodepaths[bot_id]
            np.setPos(pos)
            np.setH(bot["theta"] - 90)  # Adjust for model orientation
            np.setScale(0.4)

            # Color by team
            color = (0, 0.5, 1, 1) if bot["team"] == 0 else (1, 0.3, 0.3, 1)
            np.setColor(color)

    def _update_projectiles(self, state: Dict):
        """Update projectile models in the scene."""
        for p in self.projectile_nodepaths:
            p.removeNode()
        self.projectile_nodepaths.clear()

        for proj in state.get("projectiles", []):
            pos = LPoint3f(
                proj["x"] - self.arena_width / 2, proj["y"] - self.arena_height / 2, 0.5
            )
            proj_model = self.loader.loadModel("models/misc/sphere")
            proj_model.reparentTo(self.render)
            proj_model.setPos(pos)
            proj_model.setScale(0.15)
            color = (0.2, 1, 1, 1) if proj.get("team") == 0 else (1, 0.5, 1, 1)
            proj_model.setColor(color)
            self.projectile_nodepaths.append(proj_model)

    def _update_ui(self, state: Dict):
        """Update the text in the UI panels."""
        # General info
        time_info = f"Time: {state.get('time', 0):.1f}s"
        frame_info = f"Frame: {int(self.current_frame)}/{len(self.timeline) - 1}"
        speed_info = f"Speed: {self.playback_speed:.1f}x"
        self.info_text.setText(f"{time_info}\n{frame_info}\n{speed_info}")

        # Selected bot info
        if self.selected_bot:
            bot = self.selected_bot
            info = [
                f"Bot {bot['id']} (Team {bot['team']})",
                f"HP: {bot['hp']}",
                f"Pos: ({bot['x']:.1f}, {bot['y']:.1f})",
                f"Signal: {bot.get('signal', 'none')}",
            ]
            self.bot_info_text.setText("\n".join(info))
        else:
            self.bot_info_text.setText("Click on a bot to select it")

    def _on_slider_move(self):
        """Handle timeline slider movement."""
        self.current_frame = self.timeline_slider.getValue()
        self.playing = False
        self.play_pause_btn["text"] = "Play"

    def _toggle_play(self):
        """Toggle play/pause state."""
        self.playing = not self.playing
        self.play_pause_btn["text"] = "Pause" if self.playing else "Play"

    def _step_frame(self, direction: int):
        """Step forward or backward one frame."""
        self.playing = False
        self.play_pause_btn["text"] = "Play"
        self.current_frame += direction
        self.current_frame = max(0, min(len(self.timeline) - 1, self.current_frame))

    def _reset_sim(self):
        """Reset simulation to the first frame."""
        self.playing = False
        self.play_pause_btn["text"] = "Play"
        self.current_frame = 0

    def _toggle_fov(self):
        """Toggle the FOV display for the selected bot."""
        self.show_fov = not self.show_fov
        self._update_fov_display()

    def _update_fov_display(self):
        """Create or destroy the FOV indicator."""
        if self.fov_nodepath:
            self.fov_nodepath.removeNode()
            self.fov_nodepath = None

        if self.show_fov and self.selected_bot:
            bot = self.selected_bot
            fov_range = 15.0
            fov_angle_deg = 120

            cm = CardMaker("fov")
            cm.setFrame(0, 1, -0.5, 0.5)
            self.fov_nodepath = self.render.attachNewNode(cm.generate())
            self.fov_nodepath.setPos(
                bot["x"] - self.arena_width / 2, bot["y"] - self.arena_height / 2, 0.1
            )
            self.fov_nodepath.setScale(fov_range)
            self.fov_nodepath.setH(bot["theta"] - 90)
            self.fov_nodepath.setTransparency(1)
            color = (0, 0.5, 1, 0.3) if bot["team"] == 0 else (1, 0.3, 0.3, 0.3)
            self.fov_nodepath.setColor(color)


def run_3d_viewer(battle_file: str):
    """Launch 3D viewer with a saved battle JSON file."""
    print(f"\n=== Panda3D Battle Viewer: {battle_file} ===")

    try:
        with open(battle_file, "r") as f:
            battle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Battle file '{battle_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{battle_file}'")
        return

    print("Launching Panda3D viewer...")
    print("Controls:")
    print("  SPACE = Play/Pause")
    print("  ←/→ = Step frame by frame")
    print("  R = Reset to start")
    print("  F = Toggle FOV display")
    print("  Q/ESC = Quit")
    print("  Click bots to select them")

    app = Battle3DViewer(battle_data)
    app.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graphics3d.py <battle_log.json>")
        sys.exit(1)

    battle_file = sys.argv[1]
    run_3d_viewer(battle_file)
