extends Node
class_name ArenaManager

var world_root: Node3D
var materials
var bot_nodes: Dictionary = {}
var projectile_nodes: Array = []

func _ready():
	var material_manager_script = preload("res://scripts/MaterialManager.gd")
	materials = material_manager_script.new()

func setup(world_node: Node3D):
	world_root = world_node

func setup_arena(metadata: Dictionary):
	if not world_root:
		print("Error: world_root not set!")
		return
	
	# Clear existing objects
	for child in world_root.get_children():
		child.queue_free()
	
	bot_nodes.clear()
	projectile_nodes.clear()
	
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
	wall_node.position = Vector3(center_x, 1.0, -center_y)  # Negative Y to match coordinate system
	wall_node.rotation_degrees = Vector3(0, -angle_deg, 0)  # Negative angle to match coordinate flip
	
	world_root.add_child(wall_node)

func update_bots(state: Dictionary):
	var current_bot_ids = {}
	var current_bots = {}
	
	# Build lookup of current bots
	for bot in state.get("bots", []):
		if bot.get("alive", true):
			var bot_id = bot.get("id")
			if bot_id != null:
				current_bot_ids[bot_id] = true
				current_bots[bot_id] = bot
	
	# Remove dead or missing bots
	var bots_to_remove = []
	for bot_id in bot_nodes.keys():
		if not bot_id in current_bot_ids:
			bots_to_remove.append(bot_id)
	
	for bot_id in bots_to_remove:
		if bot_nodes[bot_id] and is_instance_valid(bot_nodes[bot_id]):
			bot_nodes[bot_id].queue_free()
		bot_nodes.erase(bot_id)
	
	# Update or create bots with smooth movement
	for bot_id in current_bots.keys():
		var bot = current_bots[bot_id]
		var bot_node = bot_nodes.get(bot_id)
		
		if bot_node == null or not is_instance_valid(bot_node):
			bot_node = create_bot(bot_id, bot.get("team", 0))
			bot_nodes[bot_id] = bot_node
		
		# Smooth position and rotation updates using interpolated data
		var target_pos = Vector3(bot.get("x", 0.0), 0.5, -bot.get("y", 0.0))
		var target_rot = Vector3(0, -bot.get("theta", 0.0) - 90, 0)
		
		# Direct assignment of interpolated positions for crisp movement
		bot_node.position = target_pos
		bot_node.rotation_degrees = target_rot
		
		# Update health
		update_bot_health(bot_node, bot.get("hp", 100))

func create_bot(bot_id: int, team: int) -> Node3D:
	var bot_factory_script = preload("res://scripts/BotFactory.gd")
	var bot_factory = bot_factory_script.new()
	var bot_node = bot_factory.create_bot(bot_id, team, materials)
	world_root.add_child(bot_node)
	return bot_node

func update_bot_health(bot_node: Node3D, hp: float):
	if bot_node.has_meta("health_component"):
		var health_component = bot_node.get_meta("health_component")
		health_component.update_health(hp)

func update_projectiles(state: Dictionary):
	var current_projectiles = state.get("projectiles", [])
	
	# Ensure projectile pool is large enough
	while projectile_nodes.size() < current_projectiles.size():
		var new_proj_node = create_projectile(0) # Team will be set below
		projectile_nodes.append(new_proj_node)

	# Update visible projectiles
	for i in range(projectile_nodes.size()):
		var proj_node = projectile_nodes[i]
		if i < current_projectiles.size():
			var proj_data = current_projectiles[i]
			
			# Set interpolated position
			var target_pos = Vector3(proj_data.get("x", 0.0), 0.5, -proj_data.get("y", 0.0))
			proj_node.position = target_pos
			
			proj_node.material_override = materials.get_projectile_material(proj_data.get("team", 0))
			proj_node.visible = true
		else:
			# Hide unused pooled projectiles
			proj_node.visible = false

func create_projectile(team: int) -> MeshInstance3D:
	var proj_mesh = SphereMesh.new()
	proj_mesh.radius = 0.15
	proj_mesh.height = 0.3
	
	var proj_node = MeshInstance3D.new()
	proj_node.mesh = proj_mesh
	proj_node.material_override = materials.get_projectile_material(team)
	
	world_root.add_child(proj_node)
	return proj_node
