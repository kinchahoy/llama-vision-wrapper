extends Node3D
class_name HealthBarComponent

var health_bg: MeshInstance3D
var health_fill: MeshInstance3D
var fill_material: StandardMaterial3D

# --- Cached Resources for Performance ---
# We create these once and reuse them for all health bars to avoid overhead.
static var quad_mesh: QuadMesh
static var bg_material: StandardMaterial3D
static var fill_material_template: StandardMaterial3D

static func _init_static_resources():
	if quad_mesh == null:
		quad_mesh = QuadMesh.new()
		quad_mesh.size = Vector2(0.6, 0.08)
	
	if bg_material == null:
		bg_material = StandardMaterial3D.new()
		bg_material.albedo_color = Color.BLACK
		bg_material.flags_unshaded = true
		bg_material.no_depth_test = true
	
	if fill_material_template == null:
		fill_material_template = StandardMaterial3D.new()
		fill_material_template.albedo_color = Color.GREEN
		fill_material_template.flags_unshaded = true
		fill_material_template.no_depth_test = true

func _ready():
	_init_static_resources()
	create_health_bar()

func create_health_bar():
	# Background (uses shared mesh and material)
	health_bg = MeshInstance3D.new()
	health_bg.mesh = quad_mesh
	health_bg.material_override = bg_material
	health_bg.position = Vector3(0, 0.8, 0)
	add_child(health_bg)
	
	# Fill (uses shared mesh, but a duplicated material for unique color)
	fill_material = fill_material_template.duplicate()
	
	health_fill = MeshInstance3D.new()
	health_fill.mesh = quad_mesh
	health_fill.material_override = fill_material
	health_fill.position = Vector3(0, 0.8, 0.01) # Slightly in front of bg
	add_child(health_fill)

func update_health(hp: float):
	var hp_ratio = clamp(hp / 100.0, 0.0, 1.0)
	
	# Scale the fill bar and adjust its position to shrink from the right
	health_fill.scale.x = hp_ratio
	health_fill.position.x = -0.3 * (1.0 - hp_ratio)
	
	# Color based on health
	var health_color = Color.GREEN
	if hp_ratio < 0.3:
		health_color = Color.RED
	elif hp_ratio < 0.6:
		health_color = Color.YELLOW
	
	# Update the unique fill material's color
	if fill_material:
		fill_material.albedo_color = health_color
