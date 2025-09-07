extends RefCounted
class_name BotFactory

func create_bot(bot_id: int, team: int, materials: MaterialManager) -> Node3D:
	var bot_container = Node3D.new()
	
	# Main bot body
	var body_node = create_bot_body(team, materials)
	bot_container.add_child(body_node)
	
	# Direction indicator
	var cone_node = create_direction_indicator(materials.get_bot_material(team))
	body_node.add_child(cone_node)
	
	# Health component
	var health_component = HealthBarComponent.new()
	body_node.add_child(health_component)
	bot_container.set_meta("health_component", health_component)
	
	# ID label
	var label = create_id_label(bot_id)
	body_node.add_child(label)
	
	# Store bot ID for selection
	bot_container.set_meta("bot_id", bot_id)
	
	return bot_container

func create_bot_body(team: int, materials: MaterialManager) -> MeshInstance3D:
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

func create_id_label(bot_id: int) -> Label3D:
	var label = Label3D.new()
	label.text = str(bot_id)
	label.position = Vector3(0, 1.2, 0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.modulate = Color.WHITE
	
	return label
