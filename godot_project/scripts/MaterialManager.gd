extends RefCounted
class_name MaterialManager

var bot_material_blue: StandardMaterial3D
var bot_material_red: StandardMaterial3D
var projectile_material_blue: StandardMaterial3D
var projectile_material_red: StandardMaterial3D
var wall_material: StandardMaterial3D
var floor_material: StandardMaterial3D
var selection_material: StandardMaterial3D

func _init():
	setup_materials()

func setup_materials():
	# Bot materials
	bot_material_blue = StandardMaterial3D.new()
	bot_material_blue.albedo_color = Color(0.2, 0.6, 1.0, 1.0)
	bot_material_blue.metallic = 0.3
	bot_material_blue.roughness = 0.2
	bot_material_blue.emission_enabled = true
	bot_material_blue.emission = Color(0.1, 0.3, 0.5, 1.0)
	bot_material_blue.rim_enabled = true
	bot_material_blue.rim = 0.8
	bot_material_blue.rim_tint = 0.5
	
	bot_material_red = StandardMaterial3D.new()
	bot_material_red.albedo_color = Color(1.0, 0.3, 0.2, 1.0)
	bot_material_red.metallic = 0.3
	bot_material_red.roughness = 0.2
	bot_material_red.emission_enabled = true
	bot_material_red.emission = Color(0.5, 0.1, 0.1, 1.0)
	bot_material_red.rim_enabled = true
	bot_material_red.rim = 0.8
	bot_material_red.rim_tint = 0.5
	
	# Projectile materials
	projectile_material_blue = StandardMaterial3D.new()
	projectile_material_blue.albedo_color = Color(0.1, 0.1, 1.0, 1.0)
	projectile_material_blue.emission_enabled = true
	projectile_material_blue.emission = Color(0.1, 0.1, 1.0, 1.0)
	projectile_material_blue.metallic = 0.8
	projectile_material_blue.roughness = 0.1
	
	projectile_material_red = StandardMaterial3D.new()
	projectile_material_red.albedo_color = Color(1.0, 0.1, 0.1, 1.0)
	projectile_material_red.emission_enabled = true
	projectile_material_red.emission = Color(1.0, 0.1, 0.1, 1.0)
	projectile_material_red.metallic = 0.8
	projectile_material_red.roughness = 0.1
	
	# Wall material
	wall_material = StandardMaterial3D.new()
	wall_material.albedo_color = Color(0.2, 0.45, 0.85, 1.0)
	wall_material.metallic = 0.2
	wall_material.roughness = 0.7
	
	# Floor material
	floor_material = StandardMaterial3D.new()
	floor_material.albedo_color = Color(0.45, 0.26, 0.40, 1.0)
	floor_material.metallic = 0.6
	floor_material.roughness = 0.6
	#floor_material.clearcoat = 0.3
	#floor_material.clearcoat_roughness = 0.15
	
	
	# Selection material
	selection_material = StandardMaterial3D.new()
	selection_material.albedo_color = Color(1.0, 1.0, 0.0, 0.7)
	selection_material.emission_enabled = true
	selection_material.emission = Color(1.5, 1.5, 0.0)
	selection_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	selection_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	selection_material.cull_mode = BaseMaterial3D.CULL_DISABLED

func get_bot_material(team: int) -> StandardMaterial3D:
	return bot_material_blue if team == 0 else bot_material_red

func get_projectile_material(team: int) -> StandardMaterial3D:
	return projectile_material_blue if team == 0 else projectile_material_red

func get_wall_material() -> StandardMaterial3D:
	return wall_material

func get_floor_material() -> StandardMaterial3D:
	return floor_material

func get_selection_material() -> StandardMaterial3D:
	return selection_material
