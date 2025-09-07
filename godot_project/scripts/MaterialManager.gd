extends RefCounted
class_name MaterialManager

var bot_material_blue: StandardMaterial3D
var bot_material_red: StandardMaterial3D
var projectile_material_blue: StandardMaterial3D
var projectile_material_red: StandardMaterial3D
var wall_material: StandardMaterial3D
var floor_material: StandardMaterial3D

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
	projectile_material_blue.albedo_color = Color(0.3, 0.8, 1.0, 1.0)
	projectile_material_blue.emission_enabled = true
	projectile_material_blue.emission = Color(0.5, 1.2, 2.0, 1.0)
	projectile_material_blue.metallic = 0.8
	projectile_material_blue.roughness = 0.1
	
	projectile_material_red = StandardMaterial3D.new()
	projectile_material_red.albedo_color = Color(1.0, 0.4, 0.8, 1.0)
	projectile_material_red.emission_enabled = true
	projectile_material_red.emission = Color(2.0, 0.8, 1.2, 1.0)
	projectile_material_red.metallic = 0.8
	projectile_material_red.roughness = 0.1
	
	# Wall material
	wall_material = StandardMaterial3D.new()
	wall_material.albedo_color = Color(0.4, 0.45, 0.55, 1.0)
	wall_material.metallic = 0.6
	wall_material.roughness = 0.3
	
	# Floor material
	floor_material = StandardMaterial3D.new()
	floor_material.albedo_color = Color(0.15, 0.16, 0.20, 1.0)
	floor_material.metallic = 0.1
	floor_material.roughness = 0.8

func get_bot_material(team: int) -> StandardMaterial3D:
	return bot_material_blue if team == 0 else bot_material_red

func get_projectile_material(team: int) -> StandardMaterial3D:
	return projectile_material_blue if team == 0 else projectile_material_red

func get_wall_material() -> StandardMaterial3D:
	return wall_material

func get_floor_material() -> StandardMaterial3D:
	return floor_material
