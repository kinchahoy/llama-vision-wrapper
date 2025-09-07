"""
Godot 4 Battle Viewer
Modern, vibrant 3D visualization of battle simulations using Godot 4 engine.
Features rich graphics, modern UI, and advanced visual effects.
"""

import json
import sys
import os
import subprocess
import tempfile
import shutil
from typing import Dict
from pathlib import Path

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
        
        print(f"Creating Godot project at: {project_path}")
        
        try:
            # Copy the entire godot_project directory structure
            source_project_dir = Path(__file__).parent / "godot_project"
            
            if not source_project_dir.exists():
                raise RuntimeError(f"Godot project template not found at: {source_project_dir}")
            
            # Copy all files and directories
            shutil.copytree(source_project_dir, project_path, dirs_exist_ok=True)
            
            # Save battle data as JSON for Godot to load
            battle_json_path = project_path / "battle_data.json"
            with open(battle_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.battle_data, f, indent=2)
            
            print(f"🎮 Created Godot project at: {project_path}")
            return str(project_path)
            
        except Exception as e:
            print(f"Error creating Godot project: {e}")
            raise
    
    def launch_godot(self, project_path: str) -> bool:
        """Launch Godot with the generated project."""
        godot_commands = ["godot4", "godot", "/usr/bin/godot4", "/usr/local/bin/godot4", "/snap/bin/godot4"]
        
        for cmd in godot_commands:
            try:
                # Try to run Godot with the project path and auto-load battle data
                env = os.environ.copy()
                env["GODOT_BATTLE_DATA"] = str(Path(project_path) / "battle_data.json")
                
                self.godot_process = subprocess.Popen([
                    cmd, 
                    "--path", project_path,
                    "--", 
                    str(Path(project_path) / "battle_data.json")
                ], env=env)
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
