extends Node
class_name UIManager

func setup_ui_styling(viewer: Control):
	var left_panel = viewer.get_node("VBoxContainer/MainArea/LeftPanel")
	var right_panel = viewer.get_node("VBoxContainer/MainArea/RightPanel")
	var play_button = viewer.get_node("VBoxContainer/BottomPanel/Toolbar/PlayButton")
	var fps_label = viewer.get_node("VBoxContainer/MainArea/LeftPanel/VBoxContainer/FPSLabel")
	
	# Panel styling
	var style_box_panel = StyleBoxFlat.new()
	style_box_panel.bg_color = Color(0.08, 0.10, 0.14, 0.95)
	style_box_panel.border_width_left = 2
	style_box_panel.border_width_right = 2
	style_box_panel.border_width_top = 2
	style_box_panel.border_width_bottom = 2
	style_box_panel.border_color = Color(0.25, 0.35, 0.45, 1.0)
	style_box_panel.corner_radius_top_left = 8
	style_box_panel.corner_radius_top_right = 8
	style_box_panel.corner_radius_bottom_left = 8
	style_box_panel.corner_radius_bottom_right = 8
	
	if left_panel:
		left_panel.add_theme_stylebox_override("panel", style_box_panel)
	if right_panel:
		right_panel.add_theme_stylebox_override("panel", style_box_panel.duplicate())
	
	# Button styling
	var button_style = StyleBoxFlat.new()
	button_style.bg_color = Color(0.15, 0.45, 0.75, 1.0)
	button_style.border_width_left = 1
	button_style.border_width_right = 1
	button_style.border_width_top = 1
	button_style.border_width_bottom = 1
	button_style.border_color = Color(0.3, 0.6, 0.9, 1.0)
	button_style.corner_radius_top_left = 6
	button_style.corner_radius_top_right = 6
	button_style.corner_radius_bottom_left = 6
	button_style.corner_radius_bottom_right = 6
	
	if play_button:
		play_button.add_theme_stylebox_override("normal", button_style)
	
	# Text styling
	if fps_label:
		fps_label.add_theme_color_override("font_color", Color(0.2, 0.9, 0.3, 1.0))

func update_fps(fps_label: Label):
	if fps_label:
		fps_label.text = "FPS: " + str(Engine.get_frames_per_second())

func update_battle_info(metadata: Dictionary):
	# Implementation for updating battle info display
	pass

func update_bot_info(bot: Dictionary):
	# Implementation for updating bot info display
	pass
