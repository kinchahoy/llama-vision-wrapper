extends Node
class_name CameraController

var camera_controller: Node3D
var camera_3d: Camera3D
var camera_target: Vector3 = Vector3.ZERO
var camera_distance: float = 25.0
var camera_angle_h: float = 0.0
var camera_angle_v: float = -45.0
var is_panning: bool = false
var last_mouse_pos: Vector2

var mode: String = "overhead"
var follow_node: Node3D = null
var eye_height: float = 1.2

func setup(controller: Node3D, camera: Camera3D):
	camera_controller = controller
	camera_3d = camera
	update_camera_position()

func handle_input(event: InputEvent):
	if mode != "overhead":
		return
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT:
			is_panning = event.pressed
			last_mouse_pos = event.position
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_distance = max(5.0, camera_distance - 2.0)
			update_camera_position()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_distance = min(100.0, camera_distance + 2.0)
			update_camera_position()
	
	elif event is InputEventMouseMotion and is_panning:
		var delta = (event.position - last_mouse_pos) * 0.01
		camera_angle_h -= delta.x
		camera_angle_v = clamp(camera_angle_v - delta.y, -80, 80)
		last_mouse_pos = event.position
		update_camera_position()

func update_camera_position():
	if not camera_controller or not camera_3d:
		return
		
	if mode == "first_person":
		if not is_instance_valid(follow_node):
			return
		var basis = follow_node.global_transform.basis
		var origin = follow_node.global_transform.origin + Vector3(0, eye_height, 0)
		camera_controller.global_transform = Transform3D(basis, origin)
		camera_3d.transform = Transform3D.IDENTITY
		return

	var h_rad = deg_to_rad(camera_angle_h)
	var v_rad = deg_to_rad(camera_angle_v)

	var offset = Vector3(
		cos(v_rad) * sin(h_rad),
		-sin(v_rad),  # Negative to look down from above
		cos(v_rad) * cos(h_rad)
	) * camera_distance
	
	camera_controller.position = camera_target + offset
	camera_3d.look_at(camera_target, Vector3.UP)

func reset_camera():
	mode = "overhead"
	follow_node = null
	is_panning = false
	camera_target = Vector3.ZERO
	camera_angle_h = 0.0
	camera_angle_v = 45.0  # Positive angle to look down from above
	update_camera_position()

func set_overhead_mode(target: Vector3 = Vector3.ZERO):
	mode = "overhead"
	follow_node = null
	camera_target = target
	is_panning = false
	update_camera_position()

func set_first_person_target(node: Node3D):
	if not is_instance_valid(node):
		return
	follow_node = node
	mode = "first_person"
	is_panning = false
	update_camera_position()
