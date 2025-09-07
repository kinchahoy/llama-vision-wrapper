"""
Godot 4 Battle Viewer
Modern, vibrant 3D visualization of battle simulations using Godot 4 engine.
Features rich graphics, modern UI, and advanced visual effects.
"""

import json
import sys
import math
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Load Godot project files from separate files
def _load_godot_files():
    """Load Godot project template files from the godot-specific directory."""
    base_dir = Path(__file__).parent
    godot_dir = base_dir / "godot-specific"
    
    try:
        project_template_path = godot_dir / "godot_project_template"
        battle_viewer_gd_path = godot_dir / "godot_battle_viewer_gd"
        battle_viewer_tscn_path = godot_dir / "godot_battle_viewer_tscn"
        
        project_template = project_template_path.read_text(encoding='utf-8') if project_template_path.exists() else ""
        battle_viewer_gd = battle_viewer_gd_path.read_text(encoding='utf-8') if battle_viewer_gd_path.exists() else ""
        battle_viewer_tscn = battle_viewer_tscn_path.read_text(encoding='utf-8') if battle_viewer_tscn_path.exists() else ""
        
        return project_template, battle_viewer_gd, battle_viewer_tscn
    except Exception as e:
        print(f"Warning: Could not load Godot template files: {e}")
        import traceback
        traceback.print_exc()
        return "", "", ""

# Load the template files at module level
GODOT_PROJECT_TEMPLATE, GODOT_BATTLE_VIEWER_GD, GODOT_BATTLE_VIEWER_TSCN = _load_godot_files()

class GodotBattleViewer:
    """
    Modern Godot 4 Battle Viewer with rich graphics and vibrant UI.
    """
    
    def __init__(self, battle_data: Dict):
        self.battle_data = battle_data
        self.timeline = battle_data["timeline"]
        self.metadata = battle_data["metadata"]
        self.temp_dir = None
        self.godot_process = None
        
    def create_godot_project(self) -> str:
        """Create a temporary Godot project with all necessary files."""
        self.temp_dir = tempfile.mkdtemp(prefix="godot_battle_viewer_")
        project_path = Path(self.temp_dir)
        
        # Validate that we have the required template files
        if not GODOT_PROJECT_TEMPLATE or not GODOT_BATTLE_VIEWER_GD or not GODOT_BATTLE_VIEWER_TSCN:
            raise RuntimeError("Missing Godot template files. Ensure godot-specific/ directory contains the required files.")
        
        # Create project.godot
        (project_path / "project.godot").write_text(GODOT_PROJECT_TEMPLATE)
        
        # Create main script
        (project_path / "BattleViewer.gd").write_text(GODOT_BATTLE_VIEWER_GD)
        
        # Create main scene
        (project_path / "BattleViewer.tscn").write_text(GODOT_BATTLE_VIEWER_TSCN)
        
        # Save battle data as JSON for Godot to load
        battle_json_path = project_path / "battle_data.json"
        with open(battle_json_path, 'w') as f:
            json.dump(self.battle_data, f, indent=2)
        
        # Create a more robust global script to pass battle data
        global_script = f'''
extends Node

var battle_data_path = "res://battle_data.json"

func _ready():
    print("Global script loaded, waiting for viewer...")
    # Use a timer to ensure the scene is fully loaded
    await get_tree().create_timer(0.5).timeout
    var viewer = get_tree().get_first_node_in_group("battle_viewer")
    if not viewer:
        viewer = get_node_or_null("/root/BattleViewer3D")
    if viewer and viewer.has_method("load_battle_data_from_file"):
        print("Loading battle data...")
        viewer.load_battle_data_from_file(battle_data_path)
    else:
        print("Warning: Could not find battle viewer node")
'''
        (project_path / "Global.gd").write_text(global_script)
        
        # Add global to project settings
        project_config = GODOT_PROJECT_TEMPLATE + '''

[autoload]
Global="*res://Global.gd"
'''
        (project_path / "project.godot").write_text(project_config)
        
        print(f"🎮 Created Godot project at: {project_path}")
        return str(project_path)
    
    def launch_godot(self, project_path: str) -> bool:
        """Launch Godot with the generated project."""
        godot_commands = ["godot4", "godot", "/usr/bin/godot4", "/usr/local/bin/godot4", "/snap/bin/godot4"]
        
        for cmd in godot_commands:
            try:
                # Try to run Godot
                self.godot_process = subprocess.Popen([
                    cmd, 
                    "--path", project_path
                ])
                print(f"✨ Launched Godot with command: {cmd}")
                return True
            except FileNotFoundError:
                continue
        
        print("❌ Error: Godot 4 not found. Please install Godot 4 and ensure it's in your PATH.")
        print("   Try: sudo apt install godot4 (Ubuntu/Debian)")
        print("   Or: sudo snap install godot-4")
        print("   Or download from: https://godotengine.org/download")
        return False
    
    def run(self):
        """Run the Godot battle viewer."""
        try:
            project_path = self.create_godot_project()
            
            if not self.launch_godot(project_path):
                return False
            
            print("🚀 Godot Battle Viewer launched successfully!")
            print("✨ Features:")
            print("   • Modern PBR materials with emission and rim lighting")
            print("   • Advanced lighting with shadows and post-processing")
            print("   • Rich UI with BBCode formatting and modern styling")
            print("   • Smooth camera controls and bot selection")
            print("   • Vibrant colors and glowing effects")
            print("")
            print("🎮 Controls:")
            print("   SPACE = Play/Pause  |  ← → = Step Frame")
            print("   Right Click + Drag = Rotate Camera  |  Wheel = Zoom")
            print("   Left Click = Select Bot  |  R = Reset  |  Q/ESC = Quit")
            
            # Wait for process to complete
            if self.godot_process:
                self.godot_process.wait()
            
            return True
            
        except Exception as e:
            print(f"❌ Error running Godot viewer: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary files: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temp dir: {e}")

def run_godot_viewer(battle_file: str):
    """Launch Godot 4 viewer with a saved battle JSON file."""
    print(f"\n=== 🎮 Godot 4 Battle Viewer: {battle_file} ===")
    
    try:
        with open(battle_file, "r") as f:
            battle_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Battle file '{battle_file}' not found")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{battle_file}'")
        return False
    
    viewer = GodotBattleViewer(battle_data)
    return viewer.run()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graphicsgodot3d.py <battle_log.json>")
        print("Example: python graphicsgodot3d.py python_battle_42.json")
        sys.exit(1)
    
    battle_file = sys.argv[1]
    success = run_godot_viewer(battle_file)
    sys.exit(0 if success else 1)
