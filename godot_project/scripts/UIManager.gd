extends Node
class_name UIManager

# Keep references to UI nodes
var fps_label: Label
var bot_info_label: RichTextLabel
var tactical_info_label: RichTextLabel
var battle_info_label: RichTextLabel
var events_label: RichTextLabel

# Caching for performance
var last_updated_frame: int = -1
var last_selected_bot_id: int = -2 # Use -2 to distinguish from deselected (-1)

# Called by BattleViewer3D._ready()
func setup(viewer: Control):
	# Get UI node references from viewer scene
	fps_label = viewer.get_node("VBoxContainer/MainArea/LeftPanel/VBoxContainer/FPSLabel")
	bot_info_label = viewer.get_node("VBoxContainer/MainArea/LeftPanel/VBoxContainer/BotInfoLabel")
	tactical_info_label = viewer.get_node("VBoxContainer/MainArea/LeftPanel/VBoxContainer/TacticalInfoLabel")
	battle_info_label = viewer.get_node("VBoxContainer/MainArea/RightPanel/VBoxContainer/BattleInfoLabel")
	events_label = viewer.get_node("VBoxContainer/MainArea/RightPanel/VBoxContainer/EventsLabel")
	
	_setup_ui_styling(viewer)
	# Clear labels initially
	bot_info_label.text = "[center][font_size=16]🎯 Click a bot to view details[/font_size][/center]"
	tactical_info_label.text = ""
	battle_info_label.text = ""
	events_label.text = ""

func _setup_ui_styling(viewer: Control):
	var left_panel = viewer.get_node("VBoxContainer/MainArea/LeftPanel")
	var right_panel = viewer.get_node("VBoxContainer/MainArea/RightPanel")
	
	# Panel styling
	var style_box_panel = StyleBoxFlat.new()
	style_box_panel.bg_color = Color(0.08, 0.10, 0.14, 0.9)
	style_box_panel.border_width_left = 1
	style_box_panel.border_width_top = 1
	style_box_panel.border_width_right = 1
	style_box_panel.border_width_bottom = 1
	style_box_panel.border_color = Color(0.25, 0.35, 0.45, 1.0)
	style_box_panel.corner_radius_top_left = 8
	style_box_panel.corner_radius_top_right = 8
	style_box_panel.corner_radius_bottom_left = 8
	style_box_panel.corner_radius_bottom_right = 8
	
	if left_panel:
		left_panel.add_theme_stylebox_override("panel", style_box_panel)
	if right_panel:
		right_panel.add_theme_stylebox_override("panel", style_box_panel.duplicate())

# --- Update Functions ---

func update_all_ui(viewer: Control):
	_update_fps()
	
	var frame_idx = int(viewer.current_frame)
	var selected_bot_id = viewer.selected_bot.get("id", -1)
	
	# PERFORMANCE: Only update expensive UI text if frame or selection has changed
	if frame_idx == last_updated_frame and selected_bot_id == last_selected_bot_id:
		return
		
	_update_right_panel(viewer)
	update_selected_bot_info(viewer)
	
	last_updated_frame = frame_idx
	last_selected_bot_id = selected_bot_id

func _update_fps():
	if fps_label:
		fps_label.text = "FPS: " + str(Engine.get_frames_per_second())

func _update_right_panel(viewer: Control):
	var state = viewer.get_current_state()
	var metadata = viewer.get_metadata()
	var summary = viewer.battle_data.get("summary", {})
	
	# Update Battle Info
	var battle_text = "[font_size=18][b]Battle Status[/b][/font_size]\n"
	battle_text += "Time: [color=yellow]%.1f[/color]s\n" % state.get("time", 0)
	battle_text += "Frame: [color=cyan]%d / %d[/color]\n" % [int(viewer.current_frame), viewer.get_timeline().size() - 1]
	battle_text += "Speed: [color=orange]%.1f[/color]x\n" % viewer.playback_speed
	
	if metadata:
		var winner = metadata.get("winner", "Unknown")
		var reason = metadata.get("reason", "Unknown")
		battle_text += "\nWinner: [color=green]%s[/color] (%s)\n" % [winner, reason]
	
	if summary:
		var mvp = summary.get("mvp", {})
		if mvp:
			var team_color = "6480ff" if mvp.get("team", 0) == 0 else "ff4040"
			battle_text += "\n[font_size=16][b]🏆 MVP[/b][/font_size]\n"
			battle_text += "Bot %d (Team %d) - [color=#%s]%.1f pts[/color]\n" % [mvp.get("bot_id"), mvp.get("team"), team_color, mvp.get("score", 0.0)]
		
		battle_text += "\n[font_size=16][b]Battle Stats[/b][/font_size]\n"
		battle_text += "Intensity: [color=yellow]%.1f[/color] shots/sec\n" % summary.get("battle_intensity", 0.0)
		battle_text += "Accuracy: [color=cyan]%.1f%%[/color]\n" % (summary.get("overall_accuracy", 0.0) * 100.0)

	if battle_info_label:
		battle_info_label.text = battle_text
	
	# Update Events
	var events_text = "[font_size=18][b]Recent Events[/b][/font_size]\n"
	var events = state.get("events", [])
	if events.is_empty():
		events_text += "[color=gray]No events in this frame.[/color]"
	else:
		for event in events.slice(-5): # Last 5 events
			var event_type = event.get("type", "unknown")
			var text = ""
			if event_type == "shot":
				text = "Bot %d fired" % event.get("bot_id")
			elif event_type == "hit":
				text = "Bot %d [color=red]hit[/color] Bot %d" % [event.get("projectile_shooter"), event.get("target")]
			elif event_type == "death":
				text = "[color=gray]Bot %d destroyed[/color]" % event.get("bot_id")
			else:
				text = "Unknown event"
			events_text += "• %s\n" % text

	if events_label:
		events_label.text = events_text

func update_selected_bot_info(viewer: Control):
	if viewer.selected_bot.is_empty():
		bot_info_label.text = "[center][font_size=16]🎯 Click a bot to view details[/font_size][/center]"
		tactical_info_label.text = ""
		return

	var bot = viewer.selected_bot
	var bot_id = bot.get("id")
	var team_color = "6480ff" if bot.get("team", 0) == 0 else "ff4040"
	var summary = viewer.battle_data.get("summary", {})
	
	# --- Bot Info (Top Left Panel) ---
	var info_text = "[font_size=18][b]Bot [color=#%s]%d[/color] (Team %d)[/b][/font_size]\n" % [team_color, bot_id, bot.get("team")]
	
	var bot_funcs = summary.get("bot_functions", {}).get(str(bot_id), {})
	info_text += "Function version: [color=cyan]v%s[/color]\n" % bot_funcs.get("version", "N/A")
	info_text += "Signal: [color=yellow]\"%s\"[/color]\n" % bot.get("signal", "none")
	
	info_text += "\n[b]-- Current State --[/b]\n"
	info_text += "Position: (%.1f, %.1f)\n" % [bot.get("x", 0.0), bot.get("y", 0.0)]
	info_text += "Heading: %.0f°\n" % bot.get("theta", 0.0)
	var speed = Vector2(bot.get("vx", 0.0), bot.get("vy", 0.0)).length()
	info_text += "Speed: %.1f m/s\n" % speed
	info_text += "Health: [color=green]%d[/color] HP\n" % bot.get("hp", 100)
	
	var bot_score = null
	for score in summary.get("bot_scores", []):
		if score.get("bot_id") == bot_id:
			bot_score = score
			break
	
	if bot_score:
		info_text += "\n[b]-- Performance --[/b]\n"
		info_text += "Score: [color=yellow]%.1f[/color] pts\n" % bot_score.get("total_score", 0.0)
		info_text += "Accuracy: %.1f%% (%d/%d)\n" % [bot_score.get("hit_rate", 0.0) * 100.0, bot_score.get("shots_hit", 0), bot_score.get("shots_fired", 0)]
		info_text += "Damage: %d dealt, %d taken\n" % [bot_score.get("damage_dealt", 0), bot_score.get("damage_taken", 0)]
		info_text += "K/D: [color=green]%d[/color]/[color=red]%d[/color]\n" % [bot_score.get("kills", 0), bot_score.get("deaths", 0)]

	if bot_info_label:
		bot_info_label.text = info_text

	# --- Tactical Info (Bottom Left Panel) ---
	var current_state = viewer.get_current_state()
	var visible_objects = _get_visible_objects(bot, current_state)
	
	var friends = []
	var enemies = []
	var projectiles = []
	
	for obj in visible_objects:
		if obj.type == "friend": friends.append(obj)
		elif obj.type == "enemy": enemies.append(obj)
		elif obj.type == "projectile": projectiles.append(obj)
	
	var tactical_text = "[font_size=16][b]Tactical Situation[/b][/font_size]\n"
	tactical_text += "Sensing: %d Friend(s), %d Enemie(s), %d Projectile(s)\n" % [friends.size(), enemies.size(), projectiles.size()]

	tactical_text += "\n[color=green]Visible Friends:[/color]\n"
	if friends.is_empty():
		tactical_text += "  [color=gray]None[/color]\n"
	else:
		for f in friends:
			tactical_text += "  F%d: %.1fm @ %.0f°\n" % [f.id, f.distance, f.angle]

	tactical_text += "\n[color=red]Visible Enemies:[/color]\n"
	if enemies.is_empty():
		tactical_text += "  [color=gray]None[/color]\n"
	else:
		for e in enemies:
			tactical_text += "  E%d: %.1fm @ %.0f°\n" % [e.id, e.distance, e.angle]
			
	tactical_text += "\n[color=yellow]Nearby Projectiles:[/color]\n"
	if projectiles.is_empty():
		tactical_text += "  [color=gray]None[/color]\n"
	else:
		for p in projectiles:
			tactical_text += "  P: %.1fm @ %.0f°\n" % [p.distance, p.angle]

	if tactical_info_label:
		tactical_info_label.text = tactical_text

# Re-implementation of the simple distance-based visibility from graphics.py
func _get_visible_objects(selected_bot: Dictionary, current_state: Dictionary) -> Array:
	var visible_objects = []
	var max_range = 15.0
	var bot_pos = Vector2(selected_bot.get("x", 0.0), selected_bot.get("y", 0.0))

	# Check bots
	for bot in current_state.get("bots", []):
		if bot.get("id") == selected_bot.get("id") or not bot.get("alive", true):
			continue
		
		var target_pos = Vector2(bot.get("x", 0.0), bot.get("y", 0.0))
		var distance = bot_pos.distance_to(target_pos)
		
		if distance <= max_range:
			var angle = rad_to_deg(bot_pos.angle_to_point(target_pos))
			var bot_type = "friend" if bot.get("team") == selected_bot.get("team") else "enemy"
			visible_objects.append({
				"type": bot_type, "id": bot.get("id"), "distance": distance, "angle": angle
			})

	# Check projectiles
	for proj in current_state.get("projectiles", []):
		var target_pos = Vector2(proj.get("x", 0.0), proj.get("y", 0.0))
		var distance = bot_pos.distance_to(target_pos)

		if distance <= max_range:
			var angle = rad_to_deg(bot_pos.angle_to_point(target_pos))
			visible_objects.append({
				"type": "projectile", "distance": distance, "angle": angle
			})
			
	return visible_objects
