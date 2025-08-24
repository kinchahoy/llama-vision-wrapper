# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Evo-LLM Battle Sim** - a 2D battle simulator designed to evolve and train small LLMs to write reactive combat programs. The project is currently in the planning phase with a comprehensive design document but no implementation yet.

## Architecture & Core Concepts

### Two-Layer Control System
- **DSL Layer**: Bots execute tiny programs (≤10 lines) at 120Hz using an additive-vote DSL
- **LLM Layer**: LLM rewrites the DSL program every ~0.2s based on rich text observations

### Key Technical Specifications
- **Physics**: 240Hz continuous physics with acceleration/turn limits
- **Control**: 120Hz DSL execution, ~5Hz LLM updates  
- **Scale**: Target 100k+ bot-ticks/sec/core, 10-100 bots per team
- **Deterministic**: Fixed update order, seeded RNG for reproducible replays

### DSL Grammar
The Domain Specific Language uses additive voting with these channels:
- **Locomotion**: `ROTATE | MOVE | DODGE` (exclusive)
- **Weapon**: `FIRE` (independent)
- **Weights**: `{0,1,5}` - where 5 is decisive, 1 is a hint
- **Absolute headings**: 0°=North (+Y), 90°=East (+X), CCW positive

## Implementation Plan (from design doc)

### Sprint 1 (Engine MVP, 1-2 weeks)
- SoA state management, physics, obstacle collision
- Neighborhood grid + Top-K selection + sector summaries  
- DSL parser and action resolver with vote carryover
- Deterministic replay logging

### Sprint 2 (LLM Integration, 1-2 weeks)  
- LLM I/O adapter with prompt templates
- Signal/role management system
- glTF/JSON export for Unity/Unreal visualization

### Data Layout
- **Structure of Arrays (SoA)** per arena: `x[], y[], θ[], v[], ω[], hp[], role_id[], signal_id[], flags[]`
- **Spatial indexing**: Uniform hash grid for neighbor queries
- **Obstacles**: List of AABBs `[xmin,ymin,xmax,ymax]`

## Core Systems to Implement

1. **Physics Engine**: Continuous motion with kinematic limits
2. **Line-of-Sight**: Segment vs AABB occlusion testing  
3. **Perception**: Top-K enemy/friend selection, 8-sector summaries
4. **DSL Interpreter**: Parse rules → votes → actions with carryover
5. **LLM Interface**: Rich prompt generation and response parsing
6. **Export System**: 240Hz logs → glTF 2.0 + JSON for visualization

## Development Notes

### Development Guidelines
- **Always use uv** for Python execution (e.g. `uv run`, `uv add`)
- **Never run python directly** - user handles all execution and testing
- **Ask before adding new libraries** - provide options with tradeoffs
- **Performance Targets**: ≥100k bot-ticks/sec/core (Python), ≥5-20M/sec (optimized)
- Consider Madrona/GPUDrive port after profiling

### Key Design Constraints
- Friendly fire enabled
- Projectiles are visible and dodgeable (not hitscan)
- All rotations use absolute headings for consistency
- LLM gets verbose context, DSL gets minimal fixed observations

### Testing Priorities
- Math correctness (heading normalization, target resolution)
- Occlusion edge cases (segment vs AABB)
- Deterministic replay verification  
- DSL interpreter vote resolution and carryover