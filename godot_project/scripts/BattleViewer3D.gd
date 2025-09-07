extends Control
class_name BattleViewer3D

# Battle state
var current_frame: float = 0.0
var playing: bool = false
var playback_speed: float = 1.0
var selected_bot: Dictionary = {}

# Interpolation state for smooth playback
var interpolation_factor: float = 0.0
var target_fps: float = 120.0  # Target smooth playback FPS

# Scene references
@onready var viewport_3d: SubViewport = $VBoxContainer/MainArea/ViewportContainer/SubViewport
@onready var camera_controller: Node3D = $VBoxContainer/MainArea/ViewportContainer/SubViewport/CameraController
@onready var camera_3d: Camera3D = $VBoxContainer/MainArea/ViewportContainer/SubViewport/CameraController/Camera3D
@onready var world_root: Node3D = $VBoxContainer/MainArea/ViewportContainer/SubViewport/World
@onready var lighting: Node3D = $VBoxContainer/MainArea/ViewportContainer/SubViewport/Lighting

# UI references
@onready var play_button: Button = $VBoxContainer/BottomPanel/Toolbar/PlayButton
@onready var timeline_slider: HSlider = $VBoxContainer/BottomPanel/Timeline/TimelineSlider
@onready var fps_label: Label = $VBoxContainer/MainArea/LeftPanel/VBoxContainer/FPSLabel
@onready var bot_info_label: RichTextLabel = $VBoxContainer/MainArea/LeftPanel/VBoxContainer/BotInfoLabel
@onready var tactical_info_label: RichTextLabel = $VBoxContainer/MainArea/LeftPanel/VBoxContainer/TacticalInfoLabel
@onready var battle_info_label: RichTextLabel = $VBoxContainer/MainArea/RightPanel/VBoxContainer/BattleInfoLabel
@onready var events_label: RichTextLabel = $VBoxContainer/MainArea/RightPanel/VBoxContainer/EventsLabel

# Components
var arena_manager
var ui_manager
var camera_controller_script

# 3D objects
# Bot and projectile nodes are now managed by ArenaManager

func _ready():
	print("BattleViewer3D initializing...")
	
	# Initialize components by preloading scripts
	var arena_manager_script = preload("res://scripts/ArenaManager.gd")
	var ui_manager_script = preload("res://scripts/UIManager.gd")
	var camera_controller_script_class = preload("res://scripts/CameraController.gd")
	
	if arena_manager_script and ui_manager_script and camera_controller_script_class:
		arena_manager = arena_manager_script.new()
		ui_manager = ui_manager_script.new()
		camera_controller_script = camera_controller_script_class.new()
		
		# Add as children to prevent garbage collection
		add_child(arena_manager)
		add_child(ui_manager)
		add_child(camera_controller_script)
		
		# Setup components
		if arena_manager and world_root:
			arena_manager.setup(world_root)
		if ui_manager:
			ui_manager.setup_ui_styling(self)
		if camera_controller_script and camera_controller and camera_3d and camera_controller_script.has_method("setup"):
			camera_controller_script.setup(camera_controller, camera_3d)
		
		print("Components initialized successfully")
	else:
		print("Error: Failed to load component scripts")
		# Initialize with null checks
		camera_controller_script = null
	
	# Connect to battle data
	BattleData.battle_data_loaded.connect(_on_battle_data_loaded)
	
	# Setup lighting
	setup_lighting()
	
	print("BattleViewer3D ready!")

func _on_battle_data_loaded():
	print("Battle data received, setting up arena...")
	if arena_manager:
		arena_manager.setup_arena(BattleData.get_metadata())
	setup_timeline_slider()
	if ui_manager:
		ui_manager.update_battle_info(BattleData.get_metadata())

func setup_lighting():
	var lighting_manager_script = preload("res://scripts/LightingManager.gd")
	var lighting_manager = lighting_manager_script.new()
	lighting.add_child(lighting_manager)
	lighting_manager.setup_modern_lighting()
	
	# Apply environment to camera
	var environment = lighting_manager.get_environment()
	if camera_3d and environment:
		camera_3d.environment = environment

func setup_timeline_slider():
	var timeline = BattleData.get_timeline()
	if timeline.size() > 0 and timeline_slider:
		timeline_slider.max_value = timeline.size() - 1
		timeline_slider.set_value_no_signal(0)
		timeline_slider.step = 0.001 # Allow fractional values for smooth playback tracking

func _input(event):
	if camera_controller_script and camera_controller_script.has_method("handle_input"):
		camera_controller_script.handle_input(event)
	
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		handle_bot_selection(event.position)
	elif event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_SPACE:
				toggle_playback()
			KEY_LEFT:
				step_frame(-1)
			KEY_RIGHT:
				step_frame(1)
			KEY_R:
				reset_simulation()
			KEY_ESCAPE, KEY_Q:
				get_tree().quit()

func handle_bot_selection(mouse_pos: Vector2):
	if not viewport_3d or not viewport_3d.world_3d or not camera_3d:
		return
		
	var from = camera_3d.project_ray_origin(mouse_pos)
	var to = from + camera_3d.project_ray_normal(mouse_pos) * 1000.0
	
	var space_state = viewport_3d.world_3d.direct_space_state
	if not space_state:
		return
		
	var query = PhysicsRayQueryParameters3D.create(from, to)
	var result = space_state.intersect_ray(query)
	
	if result and result.collider and result.collider.has_meta("bot_id"):
		var bot_id = result.collider.get_meta("bot_id")
		select_bot_by_id(bot_id)

func select_bot_by_id(bot_id: int):
	var current_state = get_current_state()
	for bot in current_state.get("bots", []):
		if bot.get("id") == bot_id:
			selected_bot = bot
			if ui_manager:
				ui_manager.update_bot_info(selected_bot)
			break

func toggle_playback():
	playing = not playing
	play_button.text = "⏸ Pause" if playing else "▶ Play"

func step_frame(direction: int):
	playing = false
	play_button.text = "▶ Play"
	var timeline = BattleData.get_timeline()
	# Snap to the nearest integer frame before stepping
	var new_frame = round(current_frame) + direction
	current_frame = clamp(new_frame, 0, timeline.size() - 1)
	timeline_slider.set_value_no_signal(current_frame)

func reset_simulation():
	playing = false
	play_button.text = "▶ Play"
	current_frame = 0
	timeline_slider.set_value_no_signal(0)

func _process(delta):
	var timeline = BattleData.get_timeline()
	if playing and timeline.size() > 0:
		# Smooth frame advance with interpolation
		var frames_per_second = 10.0  # Battle data logging rate
		var frame_advance = frames_per_second * playback_speed * delta
		var new_frame = current_frame + frame_advance
		
		# Clamp to valid range
		new_frame = clamp(new_frame, 0.0, float(timeline.size() - 1))
		
		# Only update if we're actually advancing
		if new_frame != current_frame:
			current_frame = new_frame
			timeline_slider.set_value_no_signal(current_frame)
		
		if current_frame >= timeline.size() - 1:
			playing = false
			play_button.text = "▶ Play"
	
	update_simulation_smooth()
	
	# Update FPS with null check
	if ui_manager and fps_label:
		ui_manager.update_fps(fps_label)

func get_current_state() -> Dictionary:
	var timeline = BattleData.get_timeline()
	if timeline.is_empty():
		return {}
	
	# Simple frame selection for non-interpolated access
	var frame_idx = int(current_frame)
	frame_idx = clamp(frame_idx, 0, timeline.size() - 1)
	return timeline[frame_idx]

func get_interpolated_state() -> Dictionary:
	var timeline = BattleData.get_timeline()
	if timeline.is_empty():
		return {}
	
	var frame_idx = int(current_frame)
	frame_idx = clamp(frame_idx, 0, timeline.size() - 1)
	
	# If we're at the last frame, return the last frame. Otherwise, always interpolate
	# to ensure smooth playback and scrubbing.
	if frame_idx >= timeline.size() - 1:
		return timeline[frame_idx]
	
	# Get current and next frame for interpolation
	var current_state = timeline[frame_idx]
	var next_frame_idx = min(frame_idx + 1, timeline.size() - 1)
	var next_state = timeline[next_frame_idx]
	
	# Calculate interpolation factor (0.0 to 1.0)
	var t = current_frame - float(frame_idx)
	
	# Interpolate between states
	return interpolate_states(current_state, next_state, t)

func interpolate_states(state1: Dictionary, state2: Dictionary, t: float) -> Dictionary:
	var interpolated = state1.duplicate(true)
	
	# Interpolate bot positions and rotations by matching IDs
	if state1.has("bots") and state2.has("bots"):
		var bots1 = state1.get("bots", [])
		var bots2_map = {}
		for bot2 in state2.get("bots", []):
			if bot2.has("id"):
				bots2_map[bot2["id"]] = bot2
		
		var interpolated_bots = []
		for bot1 in bots1:
			if bot1.has("id") and bots2_map.has(bot1["id"]):
				var bot2 = bots2_map[bot1["id"]]
				var interpolated_bot = bot1.duplicate()
				
				interpolated_bot["x"] = lerp(bot1.get("x", 0.0), bot2.get("x", 0.0), t)
				interpolated_bot["y"] = lerp(bot1.get("y", 0.0), bot2.get("y", 0.0), t)
				
				var theta1 = bot1.get("theta", 0.0)
				var theta2 = bot2.get("theta", 0.0)
				interpolated_bot["theta"] = lerp_angle(deg_to_rad(theta1), deg_to_rad(theta2), t) * 180.0 / PI
				
				interpolated_bots.append(interpolated_bot)
			else:
				interpolated_bots.append(bot1) # Bot probably died, don't interpolate
		
		interpolated["bots"] = interpolated_bots

	# Interpolate projectile positions by proximity matching
	if state1.has("projectiles") and state2.has("projectiles"):
		var projs1 = state1.get("projectiles", [])
		var projs2 = state2.get("projectiles", []).duplicate() # Mutable copy for matching
		var interpolated_projs = []
		
		# Max distance a projectile might travel in a log step (0.1s)
		# Projectile speed is ~6m/s, so it moves ~0.6m. A 2m search radius is safe.
		var search_radius_sq = 2.0 * 2.0
		
		for proj1 in projs1:
			var best_match_proj2 = null
			var best_match_idx = -1
			var best_dist_sq = search_radius_sq
			
			for i in range(projs2.size()):
				var proj2 = projs2[i]
				
				# Basic check: projectiles should be from the same team if that info is available
				if proj1.get("team", -1) != proj2.get("team", -1):
					continue
					
				var dx = proj2.get("x", 0.0) - proj1.get("x", 0.0)
				var dy = proj2.get("y", 0.0) - proj1.get("y", 0.0)
				var dist_sq = dx*dx + dy*dy
				
				if dist_sq < best_dist_sq:
					best_dist_sq = dist_sq
					best_match_proj2 = proj2
					best_match_idx = i

			if best_match_proj2:
				var interpolated_proj = proj1.duplicate()
				interpolated_proj["x"] = lerp(proj1.get("x", 0.0), best_match_proj2.get("x", 0.0), t)
				interpolated_proj["y"] = lerp(proj1.get("y", 0.0), best_match_proj2.get("y", 0.0), t)
				interpolated_projs.append(interpolated_proj)
				projs2.remove_at(best_match_idx) # Prevent re-matching
			else:
				# Projectile likely expired, don't interpolate
				interpolated_projs.append(proj1)
		
		interpolated["projectiles"] = interpolated_projs
	
	return interpolated

func update_simulation():
	var state = get_current_state()
	if arena_manager and state:
		arena_manager.update_bots(state)
		arena_manager.update_projectiles(state)

func update_simulation_smooth():
	var state = get_interpolated_state()
	if arena_manager and state:
		arena_manager.update_bots_smooth(state)
		arena_manager.update_projectiles_smooth(state)

# Signal handlers
func _on_play_button_pressed():
	toggle_playback()

func _on_timeline_slider_value_changed(value: float):
	var timeline = BattleData.get_timeline()
	if timeline.size() > 0:
		current_frame = clamp(value, 0.0, float(timeline.size() - 1))

func _on_timeline_slider_drag_started():
	playing = false
	play_button.text = "▶ Play"

func _on_reset_button_pressed():
	reset_simulation()

func _on_step_back_pressed():
	step_frame(-1)

func _on_step_forward_pressed():
	step_frame(1)
