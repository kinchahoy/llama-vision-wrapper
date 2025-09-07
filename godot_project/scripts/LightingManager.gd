extends Node3D
class_name LightingManager

var environment: Environment

func setup_modern_lighting():
	# Main directional light
	var sun_light = DirectionalLight3D.new()
	sun_light.light_energy = 1.2
	sun_light.light_color = Color(1.0, 0.95, 0.8, 1.0)
	sun_light.shadow_enabled = true
	sun_light.directional_shadow_mode = DirectionalLight3D.SHADOW_ORTHOGONAL
	sun_light.rotation_degrees = Vector3(-45, 45, 0)
	add_child(sun_light)
	
	# Fill light
	var fill_light = DirectionalLight3D.new()
	fill_light.light_energy = 0.3
	fill_light.light_color = Color(0.8, 0.9, 1.0, 1.0)
	fill_light.rotation_degrees = Vector3(-20, -135, 0)
	add_child(fill_light)
	
	# Setup environment
	setup_environment()

func setup_environment():
	environment = Environment.new()
	environment.background_mode = Environment.BG_SKY
	environment.sky = Sky.new()
	environment.sky.sky_material = ProceduralSkyMaterial.new()
	environment.sky.sky_material.sky_top_color = Color(0.2, 0.25, 0.35, 1.0)
	environment.sky.sky_material.sky_horizon_color = Color(0.6, 0.7, 0.8, 1.0)
	environment.ambient_light_color = Color(0.3, 0.35, 0.4, 1.0)
	environment.ambient_light_energy = 0.3
	
	# Post-processing
	environment.tonemap_mode = Environment.TONE_MAPPER_ACES
	environment.glow_enabled = true
	environment.glow_intensity = 1.0
	environment.glow_strength = 0.8
	environment.glow_bloom = 0.1
	environment.ssao_enabled = true
	environment.ssao_radius = 1.0
	environment.ssao_intensity = 1.0

func get_environment() -> Environment:
	return environment
