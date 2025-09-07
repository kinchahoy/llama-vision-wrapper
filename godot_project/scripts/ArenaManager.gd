extends Node
class_name ArenaManager

var world_root: Node3D
var materials: MaterialManager

func _ready():
	materials = MaterialManager.new()

func setup(world_node: Node3D):
	world_root = world_node

func setup_arena(metadata: Dictionary):
	if not world_root:
		print("Error: world_root not set!")
		return
	
	# Clear existing objects
	for child in world_root.get_children():
		child.queue_free()
	
	var arena_size = metadata.get("arena_size", [20, 20])
	var arena_width = arena_size[0]
	var arena_height = arena_size[1]
	
	# Create arena floor
	create_floor(arena_width, arena_height)
	
	# Create walls
	var walls_data = metadata.get("walls", [])
	for wall_def in walls_data:
		create_wall(wall_def)

func create_floor(width: float, height: float):
	var floor_mesh = BoxMesh.new()
	floor_mesh.size = Vector3(width, 0.2, height)
	
	var floor_node = MeshInstance3D.new()
	floor_node.mesh = floor_mesh
	floor_node.material_override = materials.get_floor_material()
	floor_node.position = Vector3(0, -0.1, 0)
	world_root.add_child(floor_node)

func create_wall(wall_def: Array):
	var center_x = wall_def[0]
	var center_y = wall_def[1]
	var width = wall_def[2]
	var height = wall_def[3]
	var angle_deg = wall_def[4]
	
	var wall_mesh = BoxMesh.new()
	wall_mesh.size = Vector3(width, 2.0, height)
	
	var wall_node = MeshInstance3D.new()
	wall_node.mesh = wall_mesh
	wall_node.material_override = materials.get_wall_material()
	wall_node.position = Vector3(center_x, 1.0, center_y)
	wall_node.rotation_degrees = Vector3(0, angle_deg, 0)
	
	world_root.add_child(wall_node)

func update_bots(state: Dictionary, bot_nodes: Dictionary):
	var current_bot_ids = {}
	for bot in state.get("bots", []):
		if bot.get("alive", true):
			current_bot_ids[bot.id] = true
	
	# Remove dead bots
	for bot_id in bot_nodes.keys():
		if not bot_id in current_bot_ids:
			bot_nodes[bot_id].queue_free()
			bot_nodes.erase(bot_id)
	
	# Update or create bots
	for bot in state.get("bots", []):
		if not bot.get("alive", true):
			continue
		
		var bot_id = bot.id
		var bot_node = bot_nodes.get(bot_id)
		
		if bot_node == null:
			bot_node = create_bot(bot_id, bot.team)
			bot_nodes[bot_id] = bot_node
		
		# Update position and rotation
		bot_node.position = Vector3(bot.x, 0.5, bot.y)
		bot_node.rotation_degrees = Vector3(0, bot.theta - 90, 0)
		
		# Update health
		update_bot_health(bot_node, bot.hp)

func create_bot(bot_id: int, team: int) -> Node3D:
	var bot_factory_script = load("res://scripts/BotFactory.gd")
	var bot_factory = bot_factory_script.new()
	var bot_node = bot_factory.create_bot(bot_id, team, materials)
	world_root.add_child(bot_node)
	return bot_node

func update_bot_health(bot_node: Node3D, hp: float):
	if bot_node.has_meta("health_component"):
		var health_component = bot_node.get_meta("health_component")
		health_component.update_health(hp)

func update_projectiles(state: Dictionary, projectile_nodes: Dictionary):
	var current_proj_ids = {}
	for proj in state.get("projectiles", []):
		if proj.has("id"):
			current_proj_ids[proj.id] = true
	
	# Remove expired projectiles
	for proj_id in projectile_nodes.keys():
		if not proj_id in current_proj_ids:
			projectile_nodes[proj_id].queue_free()
			projectile_nodes.erase(proj_id)
	
	# Update or create projectiles
	for proj in state.get("projectiles", []):
		if not proj.has("id"):
			continue
		
		var proj_id = proj.id
		var proj_node = projectile_nodes.get(proj_id)
		
		if proj_node == null:
			proj_node = create_projectile(proj.get("team", 0))
			projectile_nodes[proj_id] = proj_node
		
		proj_node.position = Vector3(proj.x, 0.5, proj.y)

func create_projectile(team: int) -> MeshInstance3D:
	var proj_mesh = SphereMesh.new()
	proj_mesh.radius = 0.15
	proj_mesh.height = 0.3
	
	var proj_node = MeshInstance3D.new()
	proj_node.mesh = proj_mesh
	proj_node.material_override = materials.get_projectile_material(team)
	
	world_root.add_child(proj_node)
	return proj_node
