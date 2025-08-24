"""
Panda3D Battle Viewer
Interactive 3D visualization of battle simulations using the Panda3D engine.
"""

from panda3d.core import loadPrcFileData

# Force OpenGL core profile (mac wants this), and linear workflow.
loadPrcFileData("", "load-display pandagl")
loadPrcFileData("", "gl-version 4 1")  # macOS core profile unlocks modern GLSL
loadPrcFileData("", "framebuffer-srgb true")  # linear/sRGB correct output
loadPrcFileData("", "textures-power-2 none")
loadPrcFileData("", "sync-video true")
loadPrcFileData("", "gl-debug true")
loadPrcFileData("", "notify-level-glgsg info")

import math
import json
import sys
from typing import Dict, List, Tuple, Optional

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
    Geom,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomTriangles,
    Vec3,
    AntialiasAttrib,
)


def _init_hq_rendering(app):
    """Try complexpbr first; fall back to simplepbr if needed."""
    SHADOW_MAP = 4096

    try:
        # Only enable complexpbr when the driver supports GLSL 4.3+ and we're not on macOS.
        import sys

        supports_430 = False
        gsg = getattr(app.win, "getGsg", lambda: None)()
        if gsg and hasattr(gsg, "getDriverShaderVersionMajor"):
            major = gsg.getDriverShaderVersionMajor()
            minor = gsg.getDriverShaderVersionMinor()
            supports_430 = (major, minor) >= (4, 3)
        if sys.platform == "darwin" or not supports_430:
            raise RuntimeError(
                "complexpbr requires GLSL 4.3+; falling back to simplepbr on this driver"
            )

        import complexpbr

        # Apply PBR+IBL and enable screen-space effects (AA/SSAO/HSV; SSR tweakable)
        complexpbr.apply_shader(app.render, default_lighting=True)
        complexpbr.screenspace_init()

        # Optional quality knobs (only if screen_quad exists on your build)
        if hasattr(app, "screen_quad"):
            q = app.screen_quad
            q.set_shader_input("ssao_samples", 24)
            q.set_shader_input("ssao_radius", 0.35)
            q.set_shader_input("bloom_intensity", 0.25)
            q.set_shader_input("ssr_samples", 64)  # 0 disables SSR
            q.set_shader_input("ssr_intensity", 0.5)
            q.set_shader_input("ssr_step", 4.0)
            q.set_shader_input("ssr_fresnel_pow", 3.0)
        app.render.set_shader_input("specular_factor", 1.0)  # global specular tweak

        # Sun + shadows
        sun = DirectionalLight("sun")
        sun.setShadowCaster(True, SHADOW_MAP, SHADOW_MAP)
        sun.getLens().setFilmSize(150, 150)
        sun.getLens().setNearFar(0.5, 800)
        sun_np = app.render.attachNewNode(sun)
        sun_np.setHpr(45, -55, 0)
        app.render.setLight(sun_np)

        # Extra smoothing
        app.render.setAntialias(AntialiasAttrib.MMultisample)

        app._pbr_pipeline = "complexpbr"
        print("[HQ] complexpbr active.")
    except Exception as e:
        print("[HQ] complexpbr unavailable or failed; using simplepbr. Reason:", e)
        import simplepbr

        pbr = simplepbr.init(
            use_330=True,
            enable_shadows=True,
            use_normal_maps=True,
            use_occlusion_maps=True,
            use_emission_maps=True,
        )
        # Sun + shadows (same as above)
        sun = DirectionalLight("sun")
        sun.setShadowCaster(True, SHADOW_MAP, SHADOW_MAP)
        sun.getLens().setFilmSize(150, 150)
        sun.getLens().setNearFar(0.5, 800)
        sun_np = app.render.attachNewNode(sun)
        sun_np.setHpr(45, -55, 0)
        app.render.setLight(sun_np)
        pbr.shadow_bias = 0.003

        # Enable MSAA only; no CommonFilters post-processing
        app.render.setAntialias(AntialiasAttrib.MMultisample)

        app._pbr_pipeline = "simplepbr"
        print("[HQ] simplepbr active.")


class Battle3DViewer(ShowBase):
    """Interactive 3D viewer for battle simulations using Panda3D."""

    def __init__(self, battle_data: Dict):
        """Initialize the Panda3D viewer."""
        ShowBase.__init__(self)

        # >>> NEW: high-quality rendering init
        _init_hq_rendering(self)

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
        # Tighten camera frustum for better SSAO and depth precision
        self.cam.node().getLens().setNearFar(10, 100)

        # Add ambient light for fill lighting (the main sun is created in _init_hq_rendering)
        alight = AmbientLight("ambient")
        alight.setColor((0.3, 0.3, 0.35, 1))  # Moderate ambient
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # Add a second directional light from opposite side for better illumination
        dlight2 = DirectionalLight("fill_light")
        dlight2.setColor((0.4, 0.4, 0.5, 1))  # Cooler fill light
        dl2np = self.render.attachNewNode(dlight2)
        dl2np.setHpr(-135, -30, 0)  # Opposite side
        self.render.setLight(dl2np)

        # Arena floor (procedurally generated)
        cm = CardMaker("floor")
        cm.setFrame(
            -self.arena_width / 2,
            self.arena_width / 2,
            -self.arena_height / 2,
            self.arena_height / 2,
        )
        floor = self.render.attachNewNode(cm.generate())
        floor.setPos(0, 0, 0)  # Floor at ground level (z=0)
        floor.setP(-90)  # Rotate to lie flat on XY plane

        # Apply PBR materials - make floor highly reflective
        floor.set_shader_input(
            "metallic", 1.0
        )  # Maximum metallic for mirror-like reflections
        floor.set_shader_input(
            "roughness", 0.0
        )  # Perfect smoothness for clear reflections
        floor.set_shader_input(
            "basecolor", (0.2, 0.2, 0.25, 1.0)
        )  # Set base color via shader input
        floor.setColor(0.2, 0.2, 0.25, 1)  # Dark metallic floor

        # Try to force material properties
        floor.setRenderModeWireframe()
        floor.clearRenderMode()  # Reset to solid

        self._create_walls()

    def _create_walls(self):
        """Create interior and perimeter walls from metadata."""
        # Use a reasonable default height for 3D visualization (walls extend upward)
        wall_3d_height = 2.0
        walls_data = self.metadata.get("walls", [])

        for i, wall_def in enumerate(walls_data):
            center_x, center_y, width, height, angle_deg = wall_def

            # In the 2D battle sim JSON:
            # - width and height represent the wall's 2D footprint (length and thickness)
            # - For 0° walls (horizontal): width=length, height=thickness
            # - For 90° walls (vertical): width=thickness, height=length
            # We need to determine which dimension is the thickness (should be smaller)

            # The thickness should be the smaller dimension (typically 0.2m from WALL_THICKNESS)
            wall_thickness = min(width, height)
            wall_length = max(width, height)

            # For 3D rendering, we need to map these to X and Y dimensions based on angle
            if angle_deg == 0:  # Horizontal wall (runs along X-axis)
                wall_x_size = wall_length  # Length along X
                wall_y_size = wall_thickness  # Thickness along Y
            elif angle_deg == 90:  # Vertical wall (runs along Y-axis)
                wall_x_size = wall_thickness  # Thickness along X
                wall_y_size = wall_length  # Length along Y
            else:  # Angled wall - use original dimensions
                wall_x_size = width
                wall_y_size = height

            # Create a procedural box for the wall
            wall_node = self._create_wall_geometry(
                wall_x_size, wall_y_size, wall_3d_height
            )
            wall_node.reparentTo(self.render)

            # Position the wall - battle sim uses (x,y) but Panda3D uses (x,y,z)
            # Battle sim: +X = East, +Y = North
            # Panda3D: +X = East, +Y = North, +Z = Up
            # Z-pos is half height to sit on the floor
            wall_node.setPos(center_x, center_y, wall_3d_height / 2)

            # Rotation conversion:
            # Battle sim: 0° = along +X axis (horizontal), 90° = along +Y axis (vertical)
            # Panda3D: 0° heading = facing +Y axis
            # For walls: angle_deg tells us the wall's orientation, not heading
            # A 0° wall runs horizontally (along X), a 90° wall runs vertically (along Y)
            wall_node.setHpr(angle_deg, 0, 0)

            # Apply PBR materials after positioning - different colors for different wall types
            # (no setShaderAuto for simplepbr)

            # First 4 walls are perimeter walls (outside boundary)
            if i < 4:
                # Perimeter walls - metallic with moderate reflectivity
                wall_node.set_shader_input("metallic", 0.8)
                wall_node.set_shader_input("roughness", 0.1)
                wall_node.set_shader_input("basecolor", (0.4, 0.4, 0.45, 1.0))
                wall_node.setColor(
                    0.4, 0.4, 0.45, 1
                )  # Medium gray for better visibility
            else:
                # Interior walls - brighter with higher reflectivity
                wall_node.set_shader_input("metallic", 1.0)
                wall_node.set_shader_input("roughness", 0.0)
                wall_node.set_shader_input("basecolor", (0.6, 0.8, 0.9, 1.0))
                wall_node.setColor(0.6, 0.8, 0.9, 1)  # Brighter light blue metallic

    def _create_wall_geometry(self, width, height, wall_height):
        """Create a procedural box geometry for walls."""
        from panda3d.core import GeomPrimitive

        # Create vertex data
        vdata = GeomVertexData("wall", GeomVertexFormat.getV3n3(), Geom.UHStatic)
        vdata.setNumRows(24)  # 6 faces * 4 vertices each

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")

        # Half dimensions
        hw, hh, hz = width / 2, height / 2, wall_height / 2

        # Define the 8 corners of the box
        corners = [
            (-hw, -hh, -hz),
            (hw, -hh, -hz),
            (hw, hh, -hz),
            (-hw, hh, -hz),  # bottom
            (-hw, -hh, hz),
            (hw, -hh, hz),
            (hw, hh, hz),
            (-hw, hh, hz),  # top
        ]

        # Define faces (each face has 4 vertices)
        faces = [
            # Bottom face (z = -hz)
            [(0, 1, 2, 3), (0, 0, -1)],
            # Top face (z = hz)
            [(4, 7, 6, 5), (0, 0, 1)],
            # Front face (y = -hh)
            [(0, 4, 5, 1), (0, -1, 0)],
            # Back face (y = hh)
            [(2, 6, 7, 3), (0, 1, 0)],
            # Left face (x = -hw)
            [(0, 3, 7, 4), (-1, 0, 0)],
            # Right face (x = hw)
            [(1, 5, 6, 2), (1, 0, 0)],
        ]

        # Add vertices and normals for each face
        for face_indices, face_normal in faces:
            for idx in face_indices:
                corner = corners[idx]
                vertex.addData3f(*corner)
                normal.addData3f(*face_normal)

        # Create geometry and add triangles
        geom = Geom(vdata)

        # Each face needs 2 triangles (6 vertices total, but we use 4 unique vertices per face)
        for face_idx in range(6):
            tris = GeomTriangles(Geom.UHStatic)
            base = face_idx * 4
            # First triangle: 0, 1, 2
            tris.addVertices(base, base + 1, base + 2)
            # Second triangle: 0, 2, 3
            tris.addVertices(base, base + 2, base + 3)
            geom.addPrimitive(tris)

        # Create the geometry node
        geom_node = GeomNode("wall_geom")
        geom_node.addGeom(geom)

        return NodePath(geom_node)

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
            pos=(-0.75, 0, -0.3),
            scale=0.3,
        )
        DirectButton(
            parent=self.ui_frame,
            text="<",
            command=self._step_frame,
            extraArgs=[-1],
            pos=(-0.25, 0, -0.3),
            scale=0.3,
        )
        DirectButton(
            parent=self.ui_frame,
            text=">",
            command=self._step_frame,
            extraArgs=[1],
            pos=(0.25, 0, -0.3),
            scale=0.3,
        )
        DirectButton(
            parent=self.ui_frame,
            text="Reset",
            command=self._reset_sim,
            pos=(0.75, 0, -0.3),
            scale=0.3,
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
            # Clamp dt to avoid large jumps after pausing
            dt = min(task.getDt(), 1.0 / 30.0)
            frame_advance = 10 * self.playback_speed * dt
            self.current_frame += frame_advance
            if self.current_frame >= len(self.timeline) - 1:
                self.current_frame = len(self.timeline) - 1
                self._toggle_play()

        # Ensure slider is updated if frame changes
        self.timeline_slider["value"] = self.current_frame

        current_state = self._get_current_state()

        # Update selected bot with data from the current frame
        if self.selected_bot:
            bot_id = self.selected_bot["id"]
            found_bot = None
            for bot in current_state.get("bots", []):
                if bot["id"] == bot_id:
                    found_bot = bot
                    break
            self.selected_bot = found_bot

        self._update_bots(current_state)
        self._update_projectiles(current_state)
        self._update_ui(current_state)
        self._update_fov_display()
        return task.cont

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

        # Interpolate projectiles by matching them based on their unique ID.
        projectiles1 = {p["id"]: p for p in state1.get("projectiles", []) if "id" in p}
        projectiles2 = {p["id"]: p for p in state2.get("projectiles", []) if "id" in p}
        interp_projectiles = []

        for proj_id, proj1 in projectiles1.items():
            if proj_id in projectiles2:
                proj2 = projectiles2[proj_id]
                interp_proj = proj1.copy()
                interp_proj["x"] = proj1["x"] + (proj2["x"] - proj1["x"]) * interp
                interp_proj["y"] = proj1["y"] + (proj2["y"] - proj1["y"]) * interp
                interp_projectiles.append(interp_proj)
            # If a projectile from state1 is not in state2, it has been removed
            # (e.g., hit a wall or expired), so we don't add it to the interpolated state.

        interp_state["projectiles"] = interp_projectiles

        return interp_state

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
            pos = LPoint3f(bot["x"], bot["y"], 0.5)

            if bot_id not in self.bot_nodepaths:
                # Try to load smiley model, fallback to procedural sphere
                bot_model = self.loader.loadModel("smiley")
                if not bot_model:
                    # Create a procedural sphere as fallback
                    cm = CardMaker("bot_sphere")
                    cm.setFrame(-0.5, 0.5, -0.5, 0.5)
                    bot_model = NodePath(cm.generate())

                # No setShaderAuto for simplepbr - let simplepbr handle shading
                bot_model.reparentTo(self.render)
                bot_model.setTag("bot_id", str(bot_id))
                self.bot_nodepaths[bot_id] = bot_model

                # Heading indicator (procedurally generated quad)
                cm = CardMaker("heading_indicator")
                # Creates a rectangle that is 0.1 wide and 0.5 long,
                # starting just outside the bot's sphere model.
                cm.setFrame(-0.05, 0.05, 0.45, 0.95)  # x1, x2, y1, y2
                heading_indicator = NodePath(cm.generate())
                heading_indicator.reparentTo(bot_model)
                heading_indicator.setColor(1, 1, 1, 1)  # White indicator

            np = self.bot_nodepaths[bot_id]
            np.setPos(pos)
            # Convert from battle sim angle to Panda3D heading
            # Battle sim: 0° = facing +X axis (East), Panda3D: 0° heading = facing +Y axis (North)
            # So: panda3d_heading = 90 - battle_sim_angle
            np.setH(90 - bot["theta"])
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
            pos = LPoint3f(proj["x"], proj["y"], 0.5)
            # Try to load smiley model, fallback to procedural sphere
            proj_model = self.loader.loadModel("smiley")
            if not proj_model:
                # Create a procedural sphere as fallback
                cm = CardMaker("proj_sphere")
                cm.setFrame(-0.5, 0.5, -0.5, 0.5)
                proj_model = NodePath(cm.generate())

            # No setShaderAuto for simplepbr - let simplepbr handle shading
            proj_model.reparentTo(self.render)
            proj_model.setPos(pos)
            proj_model.setScale(0.15)
            color = (0.2, 1, 1, 1) if proj.get("team") == 0 else (1, 0.5, 1, 1)
            proj_model.setColor(color)
            # Make projectiles glow with PBR emission
            proj_model.set_shader_input("emission", (*color[:3], 1.0))
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

    def _toggle_play(self):
        """Toggle play/pause state."""
        self.playing = not self.playing
        self.play_pause_btn["text"] = "Pause" if self.playing else "Play"
        if self.playing:
            self.playback_speed = 5.0
        else:
            self.playback_speed = 1.0  # Reset to normal speed when paused

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
        """Update the FOV indicator's visibility and position."""
        should_show = self.show_fov and self.selected_bot

        if should_show:
            if not self.fov_nodepath:
                # Create it if it doesn't exist
                fov_range = 15.0
                fov_angle_deg = 120
                fov_geom_node = self._create_fov_geom(fov_angle_deg, fov_range)
                self.fov_nodepath = self.render.attachNewNode(fov_geom_node)
                self.fov_nodepath.setTransparency(1)

            # Update position, orientation, and color
            bot = self.selected_bot
            self.fov_nodepath.setPos(bot["x"], bot["y"], 0.1)
            # Convert from battle sim angle to Panda3D heading
            # Battle sim: 0° = facing +X axis (East), Panda3D: 0° heading = facing +Y axis (North)
            # So: panda3d_heading = 90 - battle_sim_angle
            self.fov_nodepath.setH(90 - bot["theta"])
            color = (0, 0.5, 1, 0.3) if bot["team"] == 0 else (1, 0.3, 0.3, 0.3)
            self.fov_nodepath.setColor(color)
        elif self.fov_nodepath:
            # It exists but shouldn't be shown, so remove it
            self.fov_nodepath.removeNode()
            self.fov_nodepath = None

    def _create_fov_geom(self, fov_angle_deg, fov_range, num_segments=20):
        """Create a Geom for a 2D FOV fan."""
        fov_angle_rad = math.radians(fov_angle_deg)

        vdata = GeomVertexData("fov_vertices", GeomVertexFormat.getV3(), Geom.UHStatic)
        vdata.setNumRows(num_segments + 2)

        vertex = GeomVertexWriter(vdata, "vertex")

        # Center vertex (at the bot's location)
        vertex.addData3f(0, 0, 0)

        # Arc vertices
        angle_step = fov_angle_rad / num_segments
        # Start angle relative to forward direction (Y axis)
        start_angle = -fov_angle_rad / 2

        for i in range(num_segments + 1):
            angle = start_angle + i * angle_step
            # Create fan along +X axis (0 degrees)
            x = fov_range * math.cos(angle)
            y = fov_range * math.sin(angle)
            vertex.addData3f(x, y, 0)

        # Create triangles for the fan
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(num_segments):
            # Triangle from center to two adjacent arc points
            tris.addVertices(0, i + 1, i + 2)

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        node = GeomNode("fov_geom_node")
        node.addGeom(geom)

        return node


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
