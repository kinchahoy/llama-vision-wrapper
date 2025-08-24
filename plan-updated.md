# Updated Plan: Python Function-Based Bot Control

## Architecture Change

**Old Approach**: DSL-based bot programs with additive voting
**New Approach**: LLM-generated Python functions executed via Numba JIT

## Core Concept

Each bot runs a short Python function that:
- Takes structured input about visible objects and recent move history
- Has 10ms execution limit
- Returns a single action
- Runs via Numba JIT for performance

## Function Specification

### Input Parameters
```python
def bot_function(visible_objects: List[Dict], move_history: List[Dict]) -> Dict:
    """
    visible_objects: List of objects within bot's field of view
        - Each object: {"type": str, "x": float, "y": float, "distance": float, "angle": float, "hp": int?, "team": str?}
    
    move_history: Last 10 moves made by this bot
        - Each move: {"action": str, "target_x": float?, "target_y": float?, "timestamp": float}
    
    Returns: Single action dict
        - {"action": "move", "target_x": float, "target_y": float}
        - {"action": "fire", "target_x": float, "target_y": float} 
        - {"action": "rotate", "angle": float}
        - {"action": "dodge", "direction": float}
    """
    pass
```

## Technical Implementation

### Performance Requirements
- **Execution time**: ≤10ms per function call
- **Scale target**: 100k+ function calls/sec/core
- **JIT compilation**: Numba for hot path optimization

### Execution Environment
- Sandboxed Python execution
- Limited imports (numpy, math only)
- No file I/O, network access, or state persistence
- Function must be pure (deterministic given inputs)

### LLM Integration
- LLM generates Python function source code as string
- Function compiled once via Numba, cached per bot
- LLM can rewrite function every ~0.2s based on battle observations
- Rich text context provided to LLM for strategic reasoning

## Advantages Over DSL

1. **Flexibility**: Full Python expressiveness vs limited DSL grammar
2. **Familiarity**: LLMs are better at writing Python than custom DSL
3. **Debugging**: Standard Python tools and error messages
4. **Performance**: Numba JIT can optimize hot loops effectively
5. **Evolution**: Easier to implement genetic programming on Python AST

## Implementation Phases

### Phase 1: Core Function Runner
- Python function compilation and caching system
- Numba JIT integration with timeout handling
- Input/output validation and sanitization
- Basic action execution in physics engine

### Phase 2: LLM Integration  
- Function generation from battle context
- Hot-swapping compiled functions during battle
- Error handling for invalid generated code
- Performance monitoring and optimization

### Phase 3: Advanced Features
- Function versioning and rollback
- Genetic programming on function AST
- Multi-objective optimization (survival vs damage)
- Tournament evolution between function variants

## Migration from DSL

- Keep existing physics engine and perception systems
- Replace DSL parser with Python compiler
- Maintain same action output format
- Preserve deterministic replay capability