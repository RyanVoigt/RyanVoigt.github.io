# Drive Game - Architecture Reference

## Overview
Two-player driving/chase game. Single HTML file (`drive.html`), Canvas 2D, no external dependencies. Players control cars with WASD and arrow keys. Foundation for a chase game where one car pursues the other.

## Architecture

### Class Hierarchy
- **Entity** (base) — position, dimensions, angle, `getCorners()`, `getAxes()` for SAT collision
  - **Vehicle** (extends Entity) — car physics, input handling, rendering, collision response

### Core Systems
| System | Type | Role |
|--------|------|------|
| `CONFIG` | Object literal | All tunable values (physics, visuals, player config) |
| `gameState` | Object literal | Centralized state: vehicles, obstacles, collision events, scores, phase |
| `InputManager` | Class | Tracks pressed keys, maps bindings per player, `getInput(playerNum)` returns action object |
| `CollisionSystem` | Module object | SAT/OBB collision detection + resolution for rotated rectangles |
| `Renderer` | Module object | Background, HUD, future obstacle rendering |

### Game Loop Flow
```
requestAnimationFrame(gameLoop)
  -> deltaTime calculation (capped at 50ms)
  -> if playing:
       -> for each vehicle: InputManager.getInput() -> Vehicle.update(dt, input) -> Vehicle.clampToWorld()
       -> CollisionSystem.resolveVehicleCollisions() -> stores events in gameState.collisionEvents
  -> Renderer.drawBackground()
  -> for each vehicle: Vehicle.draw()
  -> Renderer.drawHUD()
```

## Key Design Decisions

### Car-Like Physics (not spaceship)
- **Speed is a scalar**, not a velocity vector. The car always moves in the direction it faces.
- Turn rate scales with speed (`minTurnSpeed` at rest, full `turnSpeed` at speed). Prevents spinning in place.
- Steering direction flips in reverse via `Math.sign(speed)`.
- Braking is stronger than friction. Pressing backward first decelerates, then reverses at 40% max speed.

### Input Decoupling
- Vehicles receive a plain `{ forward, backward, left, right }` object — they never reference the InputManager or specific keys.
- This means an AI controller can drive a vehicle by passing a synthetic input object with the same shape.

### SAT/OBB Collision
- Uses Separating Axis Theorem for Oriented Bounding Boxes (rotated rectangles).
- `Entity.getCorners()` returns 4 world-space corner points.
- `Entity.getAxes()` returns 2 edge normals (the separating axes to test).
- `CollisionSystem.testOBB(a, b)` returns `{ overlap, axis }` or `null`.
- Collision response: symmetric push (half overlap each), velocity bounce along collision normal.

### Collision Events
- `gameState.collisionEvents` is an array populated each frame. Each event contains `{ a, b, overlap, time }`.
- Any system can read this array to react to collisions without coupling to the collision code.
- This is the hook point for chase/tag scoring.

## Extensibility Points

### Chase Mode
- Add `chaseState: { chaser: playerNum, runner: playerNum, timer: 0 }` to gameState.
- In game loop: check `collisionEvents` for a chaser-runner pair. Increment score on contact.
- Swap roles on a timer or after a tag.

### Obstacles / Walls
- Add to `gameState.obstacles[]` as Entity instances (static, non-moving).
- `CollisionSystem.testOBB()` already works for any Entity pair — vehicle-obstacle collision uses the same SAT code.
- Add `CollisionSystem.resolveObstacleCollisions(vehicles, obstacles)` that iterates vehicles against static obstacles (push vehicle only, not the wall).

### World Editor (Drag & Drop)
- Future `EditorMode` class handles mouse events to place/resize/rotate wall entities.
- Serialize obstacle array to JSON for save/load.
- Toggle between play mode and edit mode.

### Camera / Larger World
- Wrap rendering in `ctx.save(); ctx.translate(-cameraX, -cameraY); ... ctx.restore();`
- Camera follows midpoint of both cars, or tracks the chaser.
- All game logic uses world coordinates; only rendering applies camera offset.
- `clampToWorld` would reference world bounds instead of canvas bounds.

### AI-Controlled Vehicle
- Create an AI module that returns `{ forward, backward, left, right }` based on game state.
- Pass AI output instead of `inputManager.getInput()` in the game loop.
- Vehicle class requires zero changes.

## Config-Driven Tuning
All physics and visual values live in the `CONFIG` object at the top of the script. Changing values there affects the entire game without touching any logic code:
- `vehicle.*` — acceleration, max speed, friction, turn speed, dimensions
- `collision.*` — bounce decay
- `world.*` — colors, grid spacing, boundary padding
- `players.*` — colors, names per player

## File Structure (within drive.html)
```
<style>        — CSS (viewport, canvas, pause overlay)
<canvas>       — game rendering surface
<script>
  CONFIG       — tunable parameters
  gameState    — centralized state
  InputManager — keyboard input
  Entity       — base class (position, collision geometry)
  Vehicle      — car physics, rendering, collision response
  CollisionSystem — SAT/OBB detection + resolution
  Renderer     — background, HUD drawing
  Canvas setup — sizing, resize handler
  Pause        — P key toggle
  gameLoop     — update/render cycle
  initGame     — create vehicles, start loop
```
