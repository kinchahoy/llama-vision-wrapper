"""
Panda3D Battle Viewer
Interactive 3D visualization of battle simulations using the Panda3D engine.
"""

import json
import math
import sys
from typing import Any, Dict, cast

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectFrame, DirectSlider, DirectButton, DirectLabel
from direct.gui import DirectGuiGlobals as DGG
from direct.task import Task
from panda3d.core import (  # ty: ignore[unresolved-import]
    AmbientLight,
    DirectionalLight,
    NodePath,
    TextNode,
    LPoint3f,
    CardMaker,
    CollisionTraverser,
    CollisionHandlerQueue,
    CollisionRay,
    CollisionNode,
    GeomNode,
    Geom,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomTriangles,
    Vec3,
    AntialiasAttrib,
    loadPrcFileData,
)

from viewer_core import iter_wall_params

# Force OpenGL core profile (mac wants this), and linear workflow.
loadPrcFileData("", "load-display pandagl")
loadPrcFileData("", "gl-version 4 1")  # macOS core profile unlocks modern GLSL
loadPrcFileData("", "framebuffer-srgb true")  # linear/sRGB correct output
loadPrcFileData("", "textures-power-2 none")
loadPrcFileData("", "sync-video true")
# Disable noisy GL debug / buffer warnings
# loadPrcFileData("", "gl-debug true")
# loadPrcFileData("", "notify-level-glgsg info")

# Import visibility system to use same logic as 2D viewer
PythonLLMController: Any
try:
    from llm_bot_controller import PythonLLMController as _PythonLLMController
    PythonLLMController = cast(Any, _PythonLLMController)
except ImportError:
    PythonLLMController = cast(object, None)


def _init_hq_rendering(app):
    """Try complexpbr first; fall back to simplepbr if needed."""
    SHADOW_MAP = 4096

    try:
        # Only enable complexpbr when the driver supports GLSL 4.3+ and we're not on macOS.
        import sys

        supports_430 = True
        gsg = getattr(app.win, "getGsg", lambda: None)()
        if gsg and hasattr(gsg, "getDriverShaderVersionMajor"):
            major = gsg.getDriverShaderVersionMajor()
            minor = gsg.getDriverShaderVersionMinor()
            supports_430 = (major, minor) >= (4, 3)
        if sys.platform == "darwin" or not supports_430 or True:
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
        # Display region state for keeping 3D separate from UI panels
        self._three_d_region = None
        self._disabled_default_dr = False
        self._initial_setup_done = False

        # Arena dimensions
        arena_size = self.metadata.get("arena_size", [20, 20])
        self.arena_width, self.arena_height = arena_size

        # UI state
        self.show_fov = False
        self.selected_bot = None
        self.camera_target = LPoint3f(0, 0, 0)
        self.last_mouse_pos = None

        # Initialize visibility system (same as 2D viewer)
        if PythonLLMController is not None:
            self.llm_controller = PythonLLMController()
        else:
            self.llm_controller = None

        # Panda3D object management
        self.bot_nodepaths = {}
        self.projectile_nodepaths = {}
        self.fov_nodepath = None
        self.bot_healthbars = {}
        self.bot_id_labels = {}
        self.bot_heading_indicators = {}

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
        self.cam.lookAt(self.camera_target)
        # Tighten camera frustum for better SSAO and depth precision
        self.cam.node().getLens().setNearFar(10, 100)

        # Add ambient light for fill lighting (the main sun is created in _init_hq_rendering)
        alight = AmbientLight("ambient")
        alight.setColor((0.15, 0.15, 0.18, 1))  # Lower ambient for better contrast
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # Add a second directional light from opposite side for better illumination
        dlight2 = DirectionalLight("fill_light")
        dlight2.setColor((0.2, 0.2, 0.25, 1))  # Subtle fill to avoid washout
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
        floor.setP(-90)  # Rotate card to lie flat in the XY plane

        # Apply PBR materials - tuned for clearer, higher-contrast look (less mirror-like)
        floor.set_shader_input("metallic", 0.0)   # Non-metal floor
        floor.set_shader_input("roughness", 0.8)  # Rougher surface to reduce glare
        floor.set_shader_input("basecolor", (0.16, 0.16, 0.18, 1.0))
        floor.setColor(0.16, 0.16, 0.18, 1)  # Slightly darker base color

        self._create_walls()

    def _create_walls(self):
        """Create interior and perimeter walls from metadata."""
        # Use a reasonable default height for 3D visualization (walls extend upward)
        wall_3d_height = 2.0
        walls_data = self.metadata.get("walls", [])

        for i, (center_x, center_y, width, height, angle_deg) in enumerate(
            self._iter_wall_params(walls_data)
        ):

            # Mirror the battle_arena transform: build a box of (width, height) in local X/Y
            # and rotate the node by angle_deg. Do not pre-swap dimensions based on angle,
            # since the simulator already expresses the unrotated rectangle plus rotation.
            wall_x_size = width
            wall_y_size = height

            # Create a procedural box for the wall
            wall_node = self._create_wall_geometry(
                wall_x_size, wall_y_size, wall_3d_height
            )
            wall_node.reparentTo(self.render)
            wall_node.setTwoSided(True)

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
                # Perimeter walls - more diffuse to avoid blown highlights
                wall_node.set_shader_input("metallic", 0.2)
                wall_node.set_shader_input("roughness", 0.4)
                wall_node.set_shader_input("basecolor", (0.35, 0.35, 0.40, 1.0))
                wall_node.setColor(0.35, 0.35, 0.40, 1)  # Medium gray
            else:
                # Interior walls - slightly brighter with moderate reflectivity
                wall_node.set_shader_input("metallic", 0.6)
                wall_node.set_shader_input("roughness", 0.3)
                wall_node.set_shader_input("basecolor", (0.55, 0.65, 0.80, 1.0))
                wall_node.setColor(0.55, 0.65, 0.80, 1)  # Softer blue

    def _iter_wall_params(self, walls_data):
        def _log_wall_error(wall_def, error: Exception):
            print(f"Warning: {error}; skipping wall definition: {wall_def}")

        yield from iter_wall_params(walls_data, on_error=_log_wall_error)

    def _create_wall_geometry(self, width, height, wall_height):
        """Create a procedural box geometry for walls with correct face winding."""
        # Create vertex data
        vformat = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData("wall", vformat, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")

        hw, hh, hz = width / 2.0, height / 2.0, wall_height / 2.0

        def add_face(verts, n):
            start = vertex.getWriteRow()
            for v in verts:
                vertex.addData3f(*v)
                normal.addData3f(*n)
            tris = GeomTriangles(Geom.UHStatic)
            tris.addVertices(start + 0, start + 1, start + 2)
            tris.addVertices(start + 0, start + 2, start + 3)
            return tris

        geom = Geom(vdata)
        primitives = []

        # +Z (top) - CCW when looking from above
        primitives.append(
            add_face(
                [(-hw, -hh, hz), (hw, -hh, hz), (hw, hh, hz), (-hw, hh, hz)],
                (0, 0, 1),
            )
        )
        # -Z (bottom) - CCW when looking from below
        primitives.append(
            add_face(
                [(-hw, hh, -hz), (hw, hh, -hz), (hw, -hh, -hz), (-hw, -hh, -hz)],
                (0, 0, -1),
            )
        )
        # +Y (front) - CCW when looking from +Y
        primitives.append(
            add_face(
                [(-hw, hh, -hz), (hw, hh, -hz), (hw, hh, hz), (-hw, hh, hz)],
                (0, 1, 0),
            )
        )
        # -Y (back) - CCW when looking from -Y
        primitives.append(
            add_face(
                [(hw, -hh, -hz), (-hw, -hh, -hz), (-hw, -hh, hz), (hw, -hh, hz)],
                (0, -1, 0),
            )
        )
        # +X (right) - CCW when looking from +X
        primitives.append(
            add_face(
                [(hw, hh, -hz), (hw, -hh, -hz), (hw, -hh, hz), (hw, hh, hz)],
                (1, 0, 0),
            )
        )
        # -X (left) - CCW when looking from -X
        primitives.append(
            add_face(
                [(-hw, -hh, -hz), (-hw, hh, -hz), (-hw, hh, hz), (-hw, -hh, hz)],
                (-1, 0, 0),
            )
        )

        for prim in primitives:
            geom.addPrimitive(prim)

        geom_node = GeomNode("wall_geom")
        geom_node.addGeom(geom)
        return NodePath(geom_node)

    def _create_procedural_sphere(self, radius=1.0, num_segments=16, num_rings=8):
        """Create a procedural sphere geometry."""
        vformat = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData("sphere", vformat, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")

        # Create vertices
        for i in range(num_rings + 1):
            phi = math.pi * i / num_rings
            for j in range(num_segments + 1):
                theta = 2 * math.pi * j / num_segments
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)

                n = Vec3(x, y, z)
                if n.length() > 0:
                    n.normalize()

                vertex.addData3f(x, y, z)
                normal.addData3f(n)

        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(num_rings):
            for j in range(num_segments):
                v1 = i * (num_segments + 1) + j
                v2 = v1 + num_segments + 1
                v3 = v1 + 1
                v4 = v2 + 1

                tris.addVertices(v1, v2, v3)
                tris.addVertices(v3, v2, v4)

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        node = GeomNode("procedural_sphere")
        node.addGeom(geom)
        return NodePath(node)

    def _create_procedural_cone(self, radius=1.0, height=1.0, num_segments=12):
        """Create a procedural cone geometry pointing along the +Y axis."""
        vformat = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData("cone", vformat, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")

        tris = GeomTriangles(Geom.UHStatic)

        # Create side faces
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi

            v_tip = Vec3(0, height, 0)
            v_base1 = Vec3(radius * math.cos(angle1), 0, radius * math.sin(angle1))
            v_base2 = Vec3(radius * math.cos(angle2), 0, radius * math.sin(angle2))

            n = (v_base2 - v_tip).cross(v_base1 - v_tip)
            if n.length() > 0:
                n.normalize()

            for v in [v_tip, v_base1, v_base2]:
                vertex.addData3f(v)
                normal.addData3f(n)
            tris.addConsecutiveVertices(vertex.getWriteRow() - 3, 3)

        # Create base faces
        n_base = Vec3(0, -1, 0)
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi

            v_center = Vec3(0, 0, 0)
            v_outer1 = Vec3(radius * math.cos(angle1), 0, radius * math.sin(angle1))
            v_outer2 = Vec3(radius * math.cos(angle2), 0, radius * math.sin(angle2))

            for v in [v_center, v_outer2, v_outer1]:
                vertex.addData3f(v)
                normal.addData3f(n_base)
            tris.addConsecutiveVertices(vertex.getWriteRow() - 3, 3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        node = GeomNode("procedural_cone")
        node.addGeom(geom)
        return NodePath(node)

    def _setup_ui(self):
        """Set up the DirectGUI elements for controls and info with modern styling."""
        # Build a sleek bottom toolbar with gradient-like appearance
        self.ui_bar = DirectFrame(
            parent=self.pixel2d,
            frameColor=(0.08, 0.08, 0.10, 0.95),
            frameSize=(0, 0, 0, 0),  # sized in layout()
            pos=(0, 0, 0),
            relief=DGG.RIDGE,
            borderWidth=(2, 2),
        )
        # Ensure UI renders on top of 3D and isn't affected by depth buffer
        self.ui_bar.setBin("gui-popup", 10)
        self.ui_bar.setDepthWrite(False)
        self.ui_bar.setDepthTest(False)

        # Helper to create modern, polished buttons
        def make_btn(text, cmd, extra=None, width=120, height=36, primary=False):
            # Color scheme based on button importance
            if primary:
                colors = (
                    (0.15, 0.45, 0.75, 1.0),  # normal - blue primary
                    (0.12, 0.35, 0.60, 1.0),  # click - darker blue
                    (0.18, 0.55, 0.85, 1.0),  # rollover - lighter blue
                    (0.08, 0.25, 0.40, 1.0),  # disabled - muted blue
                )
                text_color = (1.0, 1.0, 1.0, 1.0)
            else:
                colors = (
                    (0.22, 0.24, 0.28, 1.0),  # normal - dark gray
                    (0.18, 0.20, 0.24, 1.0),  # click - darker
                    (0.28, 0.30, 0.34, 1.0),  # rollover - lighter
                    (0.12, 0.14, 0.16, 1.0),  # disabled - very dark
                )
                text_color = (0.95, 0.95, 0.95, 1.0)
            
            btn = DirectButton(
                parent=self.ui_bar,
                text=text,
                command=cmd,
                extraArgs=extra or [],
                relief=DGG.RAISED,
                frameColor=colors,
                borderWidth=(1.5, 1.5),
                text_fg=text_color,
                text_scale=15,
                text_shadow=(0.1, 0.1, 0.1, 0.7),
                text_shadowOffset=(1, -1),
                rolloverSound=None,
                clickSound=None,
            )
            # Centered origin with explicit size (pixels)
            btn["frameSize"] = (-width / 2, width / 2, -height / 2, height / 2)
            return btn

        # Controls (left to right) with improved visual hierarchy
        self.play_pause_btn = make_btn("▶ Play", self._toggle_play, width=120, primary=True)
        self.step_back_btn = make_btn("◀", self._step_frame, extra=[-1], width=48)
        self.step_forward_btn = make_btn("▶", self._step_frame, extra=[1], width=48)
        self.reset_btn = make_btn("↺ Reset", self._reset_sim, width=110)
        self.reset_view_btn = make_btn("🎥 View", self._reset_camera_view, width=120)
        self.fov_btn = make_btn("👁 FOV", self._toggle_fov, width=80)
        self.help_btn = make_btn("? Help", self._toggle_help, width=85)
        self.speed_down_btn = make_btn("−", self._change_playback_speed, extra=[-1], width=48)
        self.speed_up_btn = make_btn("+", self._change_playback_speed, extra=[1], width=48)

        # Premium timeline slider with modern styling
        self.timeline_slider = DirectSlider(
            parent=self.ui_bar,
            range=(0, max(1, len(self.timeline) - 1)),
            value=0,
            pageSize=1,
            command=self._on_slider_move,
            relief=DGG.SUNKEN,
            frameColor=(0.15, 0.15, 0.18, 1.0),
            borderWidth=(1, 1),
        )
        try:
            self.timeline_slider.incButton.hide()
            self.timeline_slider.decButton.hide()
            # Style the thumb with gradient-like appearance
            self.timeline_slider.thumb["frameColor"] = (
                (0.25, 0.55, 0.85, 1.0),  # normal - bright blue
                (0.20, 0.45, 0.75, 1.0),  # click - darker
                (0.30, 0.65, 0.95, 1.0),  # rollover - lighter
                (0.15, 0.35, 0.55, 1.0),  # disabled - muted
            )
            self.timeline_slider.thumb["relief"] = DGG.RAISED
            self.timeline_slider.thumb["borderWidth"] = (1, 1)
        except Exception:
            pass

        # Modern side panels with sophisticated styling
        self.left_panel = DirectFrame(
            parent=self.pixel2d,
            frameColor=(0.06, 0.08, 0.12, 0.92),
            frameSize=(0, 0, 0, 0),  # sized in layout()
            pos=(0, 0, 0),
            relief=DGG.RIDGE,
            borderWidth=(1, 1),
        )
        self.left_panel.setBin("gui-popup", 11)
        self.left_panel.setDepthWrite(False)
        self.left_panel.setDepthTest(False)

        self.right_panel = DirectFrame(
            parent=self.pixel2d,
            frameColor=(0.06, 0.08, 0.12, 0.92),
            frameSize=(0, 0, 0, 0),  # sized in layout()
            pos=(0, 0, 0),
            relief=DGG.RIDGE,
            borderWidth=(1, 1),
        )
        self.right_panel.setBin("gui-popup", 11)
        self.right_panel.setDepthWrite(False)
        self.right_panel.setDepthTest(False)

        # Enhanced text labels with improved typography and styling
        self.info_text = DirectLabel(
            parent=self.right_panel,
            text="",
            text_fg=(0.95, 0.95, 1.0, 1.0),  # Slightly blue-tinted white
            text_scale=17,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            text_wordwrap=None,
            text_shadow=(0.1, 0.1, 0.15, 0.5),
            text_shadowOffset=(1, -1),
            textMayChange=True,
        )
        self.events_text = DirectLabel(
            parent=self.right_panel,
            text="",
            text_fg=(0.85, 0.88, 0.92, 1.0),  # Muted light blue
            text_scale=14,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            text_wordwrap=None,
            text_shadow=(0.05, 0.05, 0.1, 0.4),
            text_shadowOffset=(1, -1),
            textMayChange=True,
        )

        # Left panel with enhanced styling
        self.fps_text = DirectLabel(
            parent=self.left_panel,
            text="FPS: 0",
            text_fg=(0.2, 0.9, 0.3, 1.0),  # Bright green for performance metrics
            text_scale=15,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            text_wordwrap=None,
            text_shadow=(0.0, 0.2, 0.0, 0.6),
            text_shadowOffset=(1, -1),
            textMayChange=True,
        )
        self.bot_info_text = DirectLabel(
            parent=self.left_panel,
            text="🎯 Click on a bot to select it",
            text_fg=(0.95, 0.95, 1.0, 1.0),
            text_scale=16,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            text_wordwrap=None,
            text_shadow=(0.1, 0.1, 0.15, 0.5),
            text_shadowOffset=(1, -1),
            textMayChange=True,
        )
        self.tactical_info_text = DirectLabel(
            parent=self.left_panel,
            text="",
            text_fg=(0.88, 0.92, 0.98, 1.0),  # Light cyan for tactical info
            text_scale=14,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            text_wordwrap=None,
            text_shadow=(0.05, 0.1, 0.15, 0.4),
            text_shadowOffset=(1, -1),
            textMayChange=True,
        )
        self.help_visible = False
        self.help_text = DirectLabel(
            parent=self.left_panel,
            text=self._generate_help_text(),
            text_fg=(1.0, 0.95, 0.7, 1.0),  # Warm yellow for help text
            text_scale=13,
            text_align=TextNode.ALeft,
            frameColor=(0.05, 0.05, 0.08, 0.3),  # Subtle background for help
            relief=DGG.SUNKEN,
            borderWidth=(1, 1),
            text_wordwrap=None,
            text_shadow=(0.1, 0.05, 0.0, 0.6),
            text_shadowOffset=(1, -1),
            textMayChange=True,
        )
        self.help_text.hide()

        # Enable proper slider dragging without fighting the update loop
        self.slider_dragging = False
        self.timeline_slider.bind(DGG.B1PRESS, lambda event: setattr(self, "slider_dragging", True))
        self.timeline_slider.bind(DGG.B1RELEASE, lambda event: setattr(self, "slider_dragging", False))

        # Enhanced layout function with improved spacing and proportions
        def layout():
            w = self.win.getXSize() if self.win else 1280
            h = self.win.getYSize() if self.win else 720
            bar_h = 100  # Slightly taller for better proportions
            margin = 16  # Increased margins for better breathing room
            spacing = 12  # More generous spacing between elements

            # Wider side panels for better information display
            left_w = 360
            right_w = 360
            panel_h = max(0, h - bar_h)

            # Bottom toolbar spans full width at window bottom in pixel2d (origin bottom-left, z up)
            self.ui_bar["frameSize"] = (0, w, 0, bar_h)
            self.ui_bar.setPos(0, 0, 0)

            # Premium slider with better proportions
            slider_h = 22  # Taller for easier interaction
            slider_w = max(0, w - 4 * margin)  # More margin for cleaner look
            slider_y = bar_h - margin - slider_h / 2 - 6  # Better positioning
            self.timeline_slider.setPos(w / 2, 0, slider_y)
            try:
                self.timeline_slider["frameSize"] = (
                    -slider_w / 2,
                    slider_w / 2,
                    -slider_h / 2,
                    slider_h / 2,
                )
            except Exception:
                pass

            # Button row with better vertical positioning
            y = margin + 24  # Higher position for better balance
            x = margin * 1.5  # More left margin

            def place(btn, width, height=40):  # Taller buttons
                center_x = x + width / 2
                btn.setPos(center_x, 0, y)
                btn["frameSize"] = (-width / 2, width / 2, -height / 2, height / 2)

            # Main control buttons
            place(self.play_pause_btn, 130, 42)  # Larger primary button
            x += 130 + spacing
            place(self.step_back_btn, 52)
            x += 52 + spacing
            place(self.step_forward_btn, 52)
            x += 52 + spacing
            place(self.reset_btn, 120)
            x += 120 + spacing * 2  # Extra space before secondary buttons
            place(self.reset_view_btn, 120)
            x += 120 + spacing
            place(self.fov_btn, 90)
            x += 90 + spacing
            place(self.help_btn, 95)

            # Speed controls aligned right with better spacing
            right_x = w - margin * 1.5
            for btn, width in ((self.speed_up_btn, 52), (self.speed_down_btn, 52)):
                center_x = right_x - width / 2
                btn.setPos(center_x, 0, y)
                btn["frameSize"] = (-width / 2, width / 2, -20, 20)
                right_x -= width + spacing

            # Left and right panels (fill remaining height above bottom bar)
            self.left_panel["frameSize"] = (0, left_w, 0, panel_h)
            self.left_panel.setPos(0, 0, bar_h)

            self.right_panel["frameSize"] = (0, right_w, 0, panel_h)
            self.right_panel.setPos(max(0, w - right_w), 0, bar_h)

            # Enhanced text positioning with better spacing and hierarchy
            text_margin = 20  # More generous margins
            top_y = panel_h - text_margin
            
            # Left panel texts with improved spacing
            self.fps_text.setPos(text_margin, 0, top_y)
            self.bot_info_text.setPos(text_margin, 0, max(0, top_y - 35))
            self.tactical_info_text.setPos(text_margin, 0, max(50, panel_h * 0.42))
            self.help_text.setPos(text_margin, 0, text_margin)

            # Right panel texts with better spacing
            self.info_text.setPos(text_margin, 0, top_y)
            self.events_text.setPos(text_margin, 0, text_margin)

            # Configure the 3D display region to exclude the panels and bottom bar
            if w > 0 and h > 0 and not self._initial_setup_done:
                left_ratio = min(1.0, max(0.0, left_w / float(w)))
                right_ratio = min(1.0, max(0.0, 1.0 - (right_w / float(w))))
                bottom_ratio = min(1.0, max(0.0, bar_h / float(h)))
                top_ratio = 1.0

                # Simply modify the default display region dimensions
                try:
                    default_dr = self.win.getDisplayRegion(0)
                    if default_dr:
                        default_dr.setDimensions(
                            left_ratio, right_ratio, bottom_ratio, top_ratio
                        )
                        default_dr.setSort(0)  # Normal sort order for 3D
                        self._three_d_region = default_dr
                        self._initial_setup_done = True
                except Exception as e:
                    print(f"Warning: Could not configure display region: {e}")
                    self._initial_setup_done = True

        layout()
        # Also perform layout on the next frame to ensure window/pixel2d metrics are initialized
        def _ui_initial_layout(task):
            layout()
            return Task.done
        self.taskMgr.doMethodLater(0, _ui_initial_layout, "ui_initial_layout")
        # Re-layout when window changes size
        self.accept("window-event", lambda win: layout())

        # Text panels created above; positioned in layout()
        # All text now lives inside left/right DirectFrame panels outside 3D area.

    def _setup_controls(self):
        """Set up keyboard controls."""
        self.accept("space", self._toggle_play)
        self.accept("arrow_left", self._step_frame, [-1])
        self.accept("arrow_right", self._step_frame, [1])
        self.accept("r", self._reset_sim)
        self.accept("c", self._reset_camera_view)
        self.accept("f", self._toggle_fov)
        self.accept("escape", sys.exit)
        self.accept("q", sys.exit)
        self.accept("mouse1", self._handle_mouse_click)
        self.accept("wheel_up", self._handle_zoom, [1.0])
        self.accept("wheel_down", self._handle_zoom, [-1.0])
        self.accept("equal", self._handle_zoom, [1.0])
        self.accept("minus", self._handle_zoom, [-1.0])
        self.accept("h", self._toggle_help)

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

    def _handle_zoom(self, direction: float):
        """Handle mouse wheel zooming."""
        zoom_speed = 5.0
        # Move camera along its local Y axis (forward/backward)
        current_dist = (self.cam.getPos() - self.camera_target).length()
        # Prevent zooming in too close or past the target
        if current_dist - (direction * zoom_speed) > 2.0:
            self.cam.setY(self.cam, direction * zoom_speed)

    def _update_camera_controls(self):
        """Handle camera panning with the right mouse button."""
        # Panning with right-mouse drag (mouse3)
        if self.mouseWatcherNode.is_button_down("mouse3"):
            if self.mouseWatcherNode.hasMouse():
                mpos = self.mouseWatcherNode.getMouse()
                if self.last_mouse_pos is not None:
                    dx = mpos.getX() - self.last_mouse_pos.getX()
                    dy = mpos.getY() - self.last_mouse_pos.getY()

                    pan_speed = 40.0

                    # Get camera's right and up vectors in world space
                    cam_right = self.render.getRelativeVector(self.cam, Vec3.right())
                    cam_up = self.render.getRelativeVector(self.cam, Vec3.up())

                    # Move camera and target together for panning
                    move_vec = (cam_right * -dx * pan_speed) + (
                        cam_up * -dy * pan_speed
                    )
                    self.cam.setPos(self.cam.getPos() + move_vec)
                    self.camera_target += move_vec
                    self.cam.lookAt(self.camera_target)

                self.last_mouse_pos = mpos
        else:
            self.last_mouse_pos = None

    def _update_task(self, task):
        """Main update loop."""
        self._update_camera_controls()

        if self.playing:
            # Clamp dt to avoid large jumps after pausing
            dt = min(task.getDt(), 1.0 / 30.0)
            frame_advance = 10 * self.playback_speed * dt
            self.current_frame += frame_advance
            if self.current_frame >= len(self.timeline) - 1:
                self.current_frame = len(self.timeline) - 1
                self._toggle_play()

        # Ensure slider is updated if frame changes and we're not dragging it
        if not getattr(self, "slider_dragging", False):
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

        # Update FPS counter
        fps = self.taskMgr.globalClock.getAverageFrameRate()
        self.fps_text["text"] = f"FPS: {fps:.1f}"

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

        class MockArena:
            class _Vec2:
                __slots__ = ("x", "y")

                def __init__(self, x: float, y: float):
                    self.x = float(x)
                    self.y = float(y)

                def __iter__(self):
                    yield self.x
                    yield self.y

                def __getitem__(self, index: int) -> float:
                    if index == 0:
                        return self.x
                    if index == 1:
                        return self.y
                    raise IndexError(index)

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
                                self.position = MockArena._Vec2(
                                    bot_info["x"], bot_info["y"]
                                )
                                self.angle = math.radians(bot_info["theta"])
                                self.velocity = MockArena._Vec2(
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
                            self.position = MockArena._Vec2(
                                proj_info["x"], proj_info["y"]
                            )
                            self.velocity = MockArena._Vec2(
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
                for cx, cy, w, h, angle_deg in viewer._iter_wall_params(walls_data):
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
        for bot_id in list(self.bot_nodepaths.keys()):
            if bot_id not in current_bot_ids:
                self.bot_nodepaths[bot_id].removeNode()
                del self.bot_nodepaths[bot_id]
                # Remove health bars and labels if present
                if bot_id in self.bot_healthbars:
                    hb_bg, hb_fill = self.bot_healthbars[bot_id]
                    hb_bg.removeNode()
                    hb_fill.removeNode()
                    del self.bot_healthbars[bot_id]
                if bot_id in self.bot_id_labels:
                    self.bot_id_labels[bot_id].removeNode()
                    del self.bot_id_labels[bot_id]
                if bot_id in self.bot_heading_indicators:
                    self.bot_heading_indicators[bot_id].removeNode()
                    del self.bot_heading_indicators[bot_id]

        # Update or create bots
        for bot in state.get("bots", []):
            if not bot["alive"]:
                continue

            bot_id = bot["id"]
            pos = LPoint3f(bot["x"], bot["y"], 0.5)

            if bot_id not in self.bot_nodepaths:
                # Create a procedural sphere for the bot
                bot_model = self._create_procedural_sphere(radius=0.5)

                # No setShaderAuto for simplepbr - let simplepbr handle shading
                bot_model.reparentTo(self.render)
                bot_model.setTag("bot_id", str(bot_id))
                self.bot_nodepaths[bot_id] = bot_model

                # Health bar (background + fill)
                bg_maker = CardMaker("hb_bg")
                bg_maker.setFrame(-0.3, 0.3, 0.0, 0.06)
                hb_bg = NodePath(bg_maker.generate())
                hb_bg.reparentTo(bot_model)
                hb_bg.setPos(0, 0, 0.9)
                hb_bg.setTwoSided(True)
                hb_bg.setColor(0.2, 0.2, 0.2, 1.0)

                fill_maker = CardMaker("hb_fill")
                fill_maker.setFrame(0.0, 0.6, 0.0, 0.06)
                hb_fill = NodePath(fill_maker.generate())
                hb_fill.reparentTo(bot_model)
                hb_fill.setPos(-0.3, 0, 0.9)
                hb_fill.setTwoSided(True)
                self.bot_healthbars[bot_id] = (hb_bg, hb_fill)

                # Bot ID label
                text_node = TextNode(f"bot_id_{bot_id}")
                text_node.setText(str(bot_id))
                text_node.setAlign(TextNode.ACenter)
                label_np = bot_model.attachNewNode(text_node)
                label_np.setScale(1)
                label_np.setPos(0, 0, 1.0)
                label_np.setColor(1, 1, 1, 1)
                try:
                    label_np.setBillboardPointEye()
                except Exception:
                    pass
                self.bot_id_labels[bot_id] = label_np

                # Heading indicator cone
                heading_cone = self._create_procedural_cone(radius=0.2, height=0.4)
                heading_cone.reparentTo(bot_model)
                heading_cone.setPos(
                    0, 0.5, 0
                )  # Position cone at the front of the sphere

                # Color will be set below based on team
                self.bot_heading_indicators[bot_id] = heading_cone

            np = self.bot_nodepaths[bot_id]
            heading_cone = self.bot_heading_indicators[bot_id]
            np.setPos(pos)
            # Convert from battle sim angle to Panda3D heading
            # Battle sim: 0° = facing +X axis (East), Panda3D: 0° heading = facing +Y axis (North)
            # So: panda3d_heading = bot_theta - 90
            np.setH(bot["theta"] - 90)
            np.setScale(0.4)

            # Color by team
            color = (0, 0.5, 1, 1) if bot["team"] == 0 else (1, 0.3, 0.3, 1)
            np.setColor(color)
            # Make cone a lighter shade of the bot's color
            lighter_color = (
                min(1.0, color[0] + 0.4),
                min(1.0, color[1] + 0.4),
                min(1.0, color[2] + 0.4),
                1,
            )
            heading_cone.setColor(lighter_color)

            # Update health bar fill and color
            if bot_id in self.bot_healthbars:
                hb_bg, hb_fill = self.bot_healthbars[bot_id]
            else:
                # Create on the fly if missing
                bg_maker = CardMaker("hb_bg")
                bg_maker.setFrame(-0.3, 0.3, 0.0, 0.06)
                hb_bg = NodePath(bg_maker.generate())
                hb_bg.reparentTo(np)
                hb_bg.setPos(0, 0, 0.9)
                hb_bg.setTwoSided(True)
                hb_bg.setColor(0.2, 0.2, 0.2, 1.0)

                fill_maker = CardMaker("hb_fill")
                fill_maker.setFrame(0.0, 0.6, 0.0, 0.06)
                hb_fill = NodePath(fill_maker.generate())
                hb_fill.reparentTo(np)
                hb_fill.setPos(-0.3, 0, 0.9)
                hb_fill.setTwoSided(True)
                self.bot_healthbars[bot_id] = (hb_bg, hb_fill)

            hp_ratio = max(0.0, min(1.0, bot["hp"] / 100.0))
            if hp_ratio > 0.6:
                hp_color = (0, 1, 0, 1)  # Green
            elif hp_ratio > 0.3:
                hp_color = (1, 1, 0, 1)  # Yellow
            else:
                hp_color = (1, 0, 0, 1)  # Red
            hb_fill.setColor(hp_color)
            hb_fill.setScale(hp_ratio, 1, 1)

            # Update ID label color for contrast
            if bot_id in self.bot_id_labels:
                self.bot_id_labels[bot_id].setColor(1, 1, 1, 1)

    def _update_projectiles(self, state: Dict):
        """Update projectile models in the scene."""
        projectiles_in_state = {
            p["id"]: p for p in state.get("projectiles", []) if "id" in p
        }
        current_proj_ids = set(projectiles_in_state.keys())

        # Remove nodepaths for projectiles that are no longer present
        for proj_id in list(self.projectile_nodepaths.keys()):
            if proj_id not in current_proj_ids:
                self.projectile_nodepaths[proj_id].removeNode()
                del self.projectile_nodepaths[proj_id]

        # Update existing projectiles and create new ones
        for proj_id, proj in projectiles_in_state.items():
            pos = LPoint3f(proj["x"], proj["y"], 0.5)

            if proj_id not in self.projectile_nodepaths:
                # Create a new model for this projectile
                proj_model = self._create_procedural_sphere(
                    radius=0.5, num_segments=8, num_rings=4
                )
                proj_model.reparentTo(self.render)
                self.projectile_nodepaths[proj_id] = proj_model

            np = self.projectile_nodepaths[proj_id]
            np.setPos(pos)
            np.setScale(0.15)
            color = (0.2, 1, 1, 1) if proj.get("team") == 0 else (1, 0.5, 1, 1)
            # Make projectiles pulse brightly using color, as PBR emission is not reliable on all platforms
            pulse = (
                math.sin(self.taskMgr.globalClock.getFrameTime() * 12) + 1
            ) / 2  # Varies 0..1
            brightness = 0.8 + pulse * 0.7  # Varies 0.8..1.5 for a visible pulse
            pulsing_color = (
                min(1.0, color[0] * brightness),
                min(1.0, color[1] * brightness),
                min(1.0, color[2] * brightness),
                1,
            )
            np.setColor(pulsing_color)
            # Disable emission to avoid conflicts
            np.set_shader_input("emission", (0, 0, 0, 1))

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
        self.info_text["text"] = "\n".join(info_lines)

        # Show winner prominently if battle is over
        if state.get("time") >= meta.get("duration", 0):
            winner_banner = f"🏆 Winner: {winner.upper()} 🏆"
            self.info_text["text"] = self.info_text["text"] + f"\n\n{winner_banner}"

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
                f"Function version: v{version}",
                f"HP: {bot['hp']}",
                f"Pos: ({bot['x']:.1f}, {bot['y']:.1f})",
                f"Heading: {heading:.0f}°  Speed: {speed:.1f} m/s",
                f"Signal: {signal}",
                f"{signal_desc}",
                f"Tactical: {friends_count}F, {enemies_count}E, {len(nearby_projectiles)}P, {len(visible_walls)}W",
            ]
            self.bot_info_text["text"] = "\n".join(info)

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
                self.tactical_info_text["text"] = "\n".join(tactical_lines)
            else:
                self.tactical_info_text["text"] = ""
        else:
            self.bot_info_text["text"] = "Click on a bot to select it"
            self.tactical_info_text["text"] = ""

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
            self.events_text["text"] = "\n".join(lines)
        else:
            self.events_text["text"] = ""

    def _on_slider_move(self, value=None):
        """Handle timeline slider movement."""
        try:
            if value is None:
                value = self.timeline_slider.getValue()
            self.current_frame = float(value)
        except Exception:
            # Fallback to current slider value if command didn't pass any
            self.current_frame = float(self.timeline_slider.getValue())

    def _toggle_play(self):
        """Toggle play/pause state with enhanced button styling."""
        self.playing = not self.playing
        self.play_pause_btn["text"] = "⏸ Pause" if self.playing else "▶ Play"
        if self.playing:
            self.playback_speed = 5.0
        else:
            self.playback_speed = 1.0  # Reset to normal speed when paused

    def _step_frame(self, direction: int):
        """Step forward or backward one frame."""
        self.playing = False
        self.play_pause_btn["text"] = "▶ Play"
        self.current_frame += direction
        self.current_frame = max(0, min(len(self.timeline) - 1, self.current_frame))

    def _change_playback_speed(self, direction: float):
        """Increase or decrease playback speed."""
        if direction > 0:
            self.playback_speed = min(16.0, self.playback_speed * 1.5)
        else:
            self.playback_speed = max(0.1, self.playback_speed / 1.5)

    def _reset_sim(self):
        """Reset simulation to the first frame."""
        self.playing = False
        self.play_pause_btn["text"] = "▶ Play"
        self.current_frame = 0

    def _reset_camera_view(self):
        """Reset camera position and target to default."""
        self.camera_target = LPoint3f(0, 0, 0)
        arena_diagonal = math.sqrt(self.arena_width**2 + self.arena_height**2)
        self.cam.setPos(0, -arena_diagonal * 1.2, arena_diagonal * 1.1)
        self.cam.lookAt(self.camera_target)

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
                self.fov_nodepath.setTwoSided(
                    True
                )  # Ensure visible regardless of winding
                self.fov_nodepath.setDepthWrite(False)  # Proper transparency rendering
                self.fov_nodepath.setBin(
                    "fixed", 0
                )  # Render on top of floor to avoid z-fighting

            # Update position, orientation, and color
            bot = self.selected_bot
            self.fov_nodepath.setPos(bot["x"], bot["y"], 0.1)
            # Convert from battle sim angle to Panda3D heading
            # Battle sim: 0° = facing +X axis (East), Panda3D: 0° heading = facing +Y axis (North)
            self.fov_nodepath.setH(bot["theta"] - 90)
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
            # Create fan along +Y axis (0 degrees)
            x = fov_range * math.sin(angle)
            y = fov_range * math.cos(angle)
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

    def _toggle_help(self):
        """Toggle help overlay visibility."""
        self.help_visible = not self.help_visible
        if self.help_visible:
            self.help_text.show()
        else:
            self.help_text.hide()

    def _generate_help_text(self) -> str:
        """Generate the multi-line help text showing controls with enhanced formatting."""
        return (
            "🎮 CONTROLS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Space: ⏯ Play/Pause\n"
            "← →: Step frame by frame\n"
            "R: ↺ Reset battle to start\n"
            "C: 🎥 Reset camera view\n"
            "F: 👁 Toggle FOV display\n"
            "H: 📖 Toggle this help\n"
            "Q/Esc: ❌ Quit application\n\n"
            "📹 CAMERA CONTROLS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Right Click + Drag: Pan view\n"
            "Mouse Wheel / +/−: Zoom in/out\n\n"
            "🤖 INTERACTION\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Click Bot: 🎯 Select & view details\n"
            "Selected Bot: Shows tactical info\n"
        )


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

    print("🚀 Launching Premium Panda3D Battle Viewer...")
    print("✨ Features: Modern UI, Enhanced Graphics, Tactical Analysis")
    print("")
    print("🎮 Quick Controls:")
    print("  SPACE = ⏯ Play/Pause    |  R = ↺ Reset Battle")
    print("  ← → = Step Frame       |  C = 🎥 Reset Camera")
    print("  F = 👁 Toggle FOV       |  H = 📖 Help Overlay")
    print("  Q/ESC = ❌ Quit         |  🤖 Click Bots = Select")
    print("")
    print("📹 Camera: Right-click + Drag = Pan  |  Wheel/+- = Zoom")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    app = Battle3DViewer(battle_data)
    run_fn = getattr(app, "run")
    run_fn()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graphics3d.py <battle_log.json>")
        sys.exit(1)

    battle_file = sys.argv[1]
    run_3d_viewer(battle_file)
