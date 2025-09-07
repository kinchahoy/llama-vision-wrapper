extends Control
class_name BattleViewer3D

# Battle state
var current_frame: float = 0.0
var playing: bool = false
var playback_speed: float = 1.0
var selected_bot: Dictionary = {}

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
var bot_nodes: Dictionary = {}
var projectile_nodes: Dictionary = {}

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
		timeline_slider.value = 0
		timeline_slider.step = 0.1

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
	current_frame = clamp(current_frame + direction, 0, timeline.size() - 1)
	timeline_slider.value = current_frame

func reset_simulation():
	playing = false
	play_button.text = "▶ Play"
	current_frame = 0
	timeline_slider.value = 0

func _process(delta):
	var timeline = BattleData.get_timeline()
	if playing and timeline.size() > 0:
		current_frame += 10.0 * playback_speed * delta
		if current_frame >= timeline.size() - 1:
			current_frame = timeline.size() - 1
			playing = false
			play_button.text = "▶ Play"
		timeline_slider.value = current_frame
	
	update_simulation()
	
	# Update FPS with null check
	if ui_manager and fps_label:
		ui_manager.update_fps(fps_label)

func get_current_state() -> Dictionary:
	var timeline = BattleData.get_timeline()
	if timeline.is_empty():
		return {}
	
	var frame_idx = int(current_frame)
	frame_idx = clamp(frame_idx, 0, timeline.size() - 1)
	return timeline[frame_idx]

func update_simulation():
	var state = get_current_state()
	if arena_manager and state:
		arena_manager.update_bots(state, bot_nodes)
		arena_manager.update_projectiles(state, projectile_nodes)

# Signal handlers
func _on_play_button_pressed():
	toggle_playback()

func _on_timeline_slider_value_changed(value: float):
	current_frame = value

func _on_timeline_slider_drag_started():
	playing = false
	play_button.text = "▶ Play"

func _on_reset_button_pressed():
	reset_simulation()

func _on_step_back_pressed():
	step_frame(-1)

func _on_step_forward_pressed():
	step_frame(1)
