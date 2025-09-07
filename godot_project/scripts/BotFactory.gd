extends RefCounted
class_name BotFactory

func create_bot(bot_id: int, team: int, materials) -> Area3D:
	var bot_container = Area3D.new()
	bot_container.collision_layer = 1
	bot_container.collision_mask = 0
	
	# Collision shape for picking
	var shape = SphereShape3D.new()
	shape.radius = 0.6 # Slightly larger for easier clicking
	var collision_shape = CollisionShape3D.new()
	collision_shape.shape = shape
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
	body_node.add_child(cone_node)
	
	# Health component
	var health_component_script = preload("res://scripts/HealthBarComponent.gd")
	var health_component = health_component_script.new()
	body_node.add_child(health_component)
	bot_container.set_meta("health_component", health_component)
	
	# ID label
	var label = create_id_label(bot_id)
	body_node.add_child(label)
	
	# Store bot ID for selection
	bot_container.set_meta("bot_id", bot_id)
	
	return bot_container

func create_bot_body(team: int, materials) -> MeshInstance3D:
	var body_mesh = SphereMesh.new()
	body_mesh.radius = 0.5
	body_mesh.height = 1.0
	
	var body_node = MeshInstance3D.new()
	body_node.mesh = body_mesh
	body_node.material_override = materials.get_bot_material(team)
	
	return body_node

func create_direction_indicator(material: StandardMaterial3D) -> MeshInstance3D:
	var cone_mesh = CylinderMesh.new()
	cone_mesh.top_radius = 0.0
	cone_mesh.bottom_radius = 0.2
	cone_mesh.height = 0.4
	
	var cone_node = MeshInstance3D.new()
	cone_node.mesh = cone_mesh
	cone_node.material_override = material
	cone_node.position = Vector3(0, 0, 0.6)
	cone_node.rotation_degrees = Vector3(-90, 0, 0)
	
	return cone_node

func create_selection_indicator(materials) -> MeshInstance3D:
	var ring_mesh = TorusMesh.new()
	ring_mesh.inner_radius = 0.6
	ring_mesh.outer_radius = 0.7
	
	var indicator = MeshInstance3D.new()
	indicator.mesh = ring_mesh
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
