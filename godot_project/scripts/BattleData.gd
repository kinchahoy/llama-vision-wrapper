extends Node

var battle_data: Dictionary = {}
var timeline: Array = []
var metadata: Dictionary = {}

signal battle_data_loaded

func _ready():
	# Try to load from command line arguments or environment
	var args = OS.get_cmdline_args()
	var battle_file_path = ""
	
	# Check for battle data file in arguments
	for i in range(args.size()):
		if args[i].ends_with(".json"):
			battle_file_path = args[i]
			break
	
	# Check environment variable
	if battle_file_path.is_empty():
		battle_file_path = OS.get_environment("GODOT_BATTLE_DATA")
	
	# Default fallback
	if battle_file_path.is_empty():
		battle_file_path = "res://battle_data.json"
	
	# Load the battle data
	call_deferred("load_battle_data_from_file", battle_file_path)

func load_battle_data_from_file(file_path: String) -> bool:
	print("Loading battle data from: ", file_path)
	
	var file = FileAccess.open(file_path, FileAccess.READ)
	if file == null:
		print("Error: Could not open battle file: ", file_path)
		return false
	
	var json_string = file.get_as_text()
	file.close()
	
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	if parse_result != OK:
		print("Error parsing JSON: ", json.get_error_message())
		return false
	
	battle_data = json.data
	timeline = battle_data.get("timeline", [])
	metadata = battle_data.get("metadata", {})
	
	print("Battle data loaded - Timeline frames: ", timeline.size())
	battle_data_loaded.emit()
	return true

func get_timeline() -> Array:
	return timeline

func get_metadata() -> Dictionary:
	return metadata

func get_battle_data() -> Dictionary:
	return battle_data
