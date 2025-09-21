extends RefCounted
class_name BotFactory

# Cached resources for performance
var body_mesh: SphereMesh
var collision_shape_3d: SphereShape3D
var direction_indicator_mesh: CylinderMesh
var selection_indicator_mesh: TorusMesh

func _init():
	# Create meshes once and reuse them for all bots
	body_mesh = SphereMesh.new()
	body_mesh.radius = 0.5
	body_mesh.height = 1.0
	
	collision_shape_3d = SphereShape3D.new()
	collision_shape_3d.radius = 0.6 # Slightly larger for easier clicking
	
	direction_indicator_mesh = CylinderMesh.new()
	direction_indicator_mesh.top_radius = 0.0
	direction_indicator_mesh.bottom_radius = 0.2
	direction_indicator_mesh.height = 0.4
	
	selection_indicator_mesh = TorusMesh.new()
	selection_indicator_mesh.inner_radius = 0.6
	selection_indicator_mesh.outer_radius = 0.7

func create_bot(bot_id: int, team: int, materials) -> Area3D:
	var bot_container = Area3D.new()
	bot_container.collision_layer = 1
	bot_container.collision_mask = 0
	
	# Collision shape for picking (reusing shape resource)
	var collision_shape = CollisionShape3D.new()
	collision_shape.shape = collision_shape_3d
	bot_container.add_child(collision_shape)
	
	# Main bot body
	var body_node = create_bot_body(team, materials)
	body_node.name = "Body"
	bot_container.add_child(body_node)
	
	# Selection indicator
	var selection_indicator = create_selection_indicator(materials)
	selection_indicator.name = "SelectionIndicator"
	selection_indicator.visible = false
	body_node.add_child(selection_indicator)
	
	# Direction indicator
	var cone_node = create_direction_indicator(materials.get_bot_material(team))
	cone_node.name = "DirectionIndicator"
	body_node.add_child(cone_node)
	
	# Health component
	var health_component_script = preload("res://scripts/HealthBarComponent.gd")
	var health_component = health_component_script.new()
	health_component.name = "HealthBar"
	body_node.add_child(health_component)
	bot_container.set_meta("health_component", health_component)
	
	# ID label
	var label = create_id_label(bot_id)
	label.name = "IDLabel"
	body_node.add_child(label)
	
	# Store bot ID for selection
	bot_container.set_meta("bot_id", bot_id)
	
	return bot_container

func create_bot_body(team: int, materials) -> MeshInstance3D:
	var body_node = MeshInstance3D.new()
	body_node.mesh = body_mesh # Reuse mesh
	body_node.material_override = materials.get_bot_material(team)
	
	return body_node

func create_direction_indicator(material: StandardMaterial3D) -> MeshInstance3D:
	var cone_node = MeshInstance3D.new()
	cone_node.mesh = direction_indicator_mesh # Reuse mesh
	cone_node.material_override = material
	# Point along local forward (-Z) and place slightly in front of the body
	cone_node.position = Vector3(0, 0, -0.6)
	cone_node.rotation_degrees = Vector3(-90, 0, 0)
	
	return cone_node

func create_selection_indicator(materials) -> MeshInstance3D:
	var indicator = MeshInstance3D.new()
	indicator.mesh = selection_indicator_mesh # Reuse mesh
	indicator.material_override = materials.get_selection_material()
	indicator.rotation_degrees.x = 90
	indicator.position.y = -0.45 # Relative to body center, near the floor
	
	return indicator

func create_id_label(bot_id: int) -> Label3D:
	var label = Label3D.new()
	label.text = str(bot_id)
	label.position = Vector3(0, 1.2, 0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.modulate = Color.WHITE
	
	return label
