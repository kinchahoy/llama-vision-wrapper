extends Node3D
class_name HealthBarComponent

var health_bg: MeshInstance3D
var health_fill: MeshInstance3D

func _ready():
	create_health_bar()

func create_health_bar():
	# Background
	health_bg = create_health_bar_quad(0.6, 0.08, Color.BLACK)
	health_bg.position = Vector3(0, 0.8, 0)
	add_child(health_bg)
	
	# Fill
	health_fill = create_health_bar_quad(0.6, 0.08, Color.GREEN)
	health_fill.position = Vector3(0, 0.8, 0.01)
	add_child(health_fill)

func create_health_bar_quad(width: float, height: float, color: Color) -> MeshInstance3D:
	var quad_mesh = QuadMesh.new()
	quad_mesh.size = Vector2(width, height)
	
	var material = StandardMaterial3D.new()
	material.albedo_color = color
	material.flags_unshaded = true
	material.no_depth_test = true
	
	var quad_node = MeshInstance3D.new()
	quad_node.mesh = quad_mesh
	quad_node.material_override = material
	
	return quad_node

func update_health(hp: float):
	var hp_ratio = clamp(hp / 100.0, 0.0, 1.0)
	
	health_fill.scale.x = hp_ratio
	health_fill.position.x = -0.3 + (0.3 * hp_ratio)
	
	# Color based on health
	var health_color = Color.GREEN
	if hp_ratio < 0.3:
		health_color = Color.RED
	elif hp_ratio < 0.6:
		health_color = Color.YELLOW
	
	health_fill.material_override.albedo_color = health_color
