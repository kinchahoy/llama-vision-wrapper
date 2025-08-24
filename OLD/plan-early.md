# Evo‑LLM Battle Sim — Early Plan

**Purpose:** Build a simple, fast 2D battle simulator whose primary job is to **evolve improved small LLMs** that write short, reactive programs. Each bot runs a compact DSL (≤10 lines typical, ≤20 hard cap) at high tick rates; a small LLM rewrites that program at a lower cadence based on a **rich, human‑readable text observation**. We run thousands of episodes and train the LLM via **RLHF/DPO/BC**.

---

## 0) Executive summary

* **LLM‑first:** The sim exists to produce clean, regular supervision that teaches small models to produce good, reactive programs.
* **Two layers of control:** Bots execute a tiny, additive‑vote DSL **every controller tick**; an **LLM** periodically rewrites the DSL program, emits a short **plan**, and selects a **team signal**.
* **Continuous physics:** Moving and shooting are independent; speed/heading changes respect acceleration limits. To slow/stop, the DSL must issue speed setpoints explicitly.
* **Absolute heading:** All rotations target **absolute headings** (0°=North/+Y, 90°=East/+X, CCW positive). `ROTATE TO TARGET …` resolves to an **absolute** heading.
* **LLM gets rich context:** The LLM prompt is verbose and includes: game rules, a glossary, **full previous program**, **events since last LLM turn**, compact observations **plus an extended “LLM‑only extras” block** (more visible enemies, obstacle AABBs in FOV, etc.). The DSL program itself remains small and reactive.
* **Deterministic & exportable:** Fixed update order and seeded RNG produce deterministic replays. We export 240 Hz logs + glTF/JSON for silky animation in Unity/Unreal.

---

## 1) Design principles

1. **LLM‑first objective.** Create rich yet regular data for learning to write concise reactive programs.
2. **Simplicity → speed.** Minimal physics, small fixed observations for the DSL; branch‑light interpreter; Python reference, optimize hotspots only when measured (Numba/WASM/GPU). Target later migration to Madrona/GPUDrive for scale. We also plan to eventually animate simulation runs in Unreal Engine as part of the complete game / simulation tool.
3. **Reactive feel.** Bots react every controller tick; LLM only nudges by **replacing** the short program every \~0.2 s.
4. **Readable IO.** Human‑parsable observations and DSL; stable field names; fixed grammar and schema.
5. **Looks epic if animated.** Visible (but dodgeable) projectiles; flanks, collapses, and short team **signals**; smooth trajectories with accel/turn limits.

---

## 2) Clocks, scales, and performance

* **Units:** 1u = **1 meter**.
* **Rates:**

  * Physics `dt_p = 1/240 s` (240 Hz)
  * Controller/DSL `dt_c = 1/120 s` (120 Hz)
  * LLM update `dt_LLM = 25 × dt_c ≈ 0.208 s` (configurable 10–50 controller ticks)
* **Episode length:** 10–60 s (configurable). 10–100 bots per team.
* **Perf goals (reference Python):** ≥**100k bot‑ticks/sec/core**; ≥**5–20M/sec** with Numba/WASM/GPU.
* **Determinism:** Fixed update order, seeded RNG, SoA buffers.

---

## 3) World & physics (2D)

* **Space:** Rectangular arena (e.g., 100×100 m). Two map modes:

  * **OPEN** (no obstacles)
  * **OCCLUDED** (axis‑aligned rectangular obstacles; AABBs)
* **Bots:** Discs (radius **0.4 m**), heading `θ` (deg, absolute), forward‑facing.
* **Kinematics:** `v_max = 2 m/s`, `v_rev_max = 1 m/s`, `a_max = 8 m/s²`, `ω_max = 260°/s`, `α_max` (turn accel clamp), linear damping.
* **Collision with obstacles:** **Slide** along AABBs (tangential reflection); no interpenetration.
* **Projectiles:** speed `v_proj = 6 m/s`, TTL `5.0 s`, fire rate **8 Hz**, visible travel (no hitscan), no spread. Firing has risks!
* **HP & damage:** 100 HP max; projectile deals 25 damage; **friendly fire enabled** 

**Continuous locomotion + independent fire**

* Shooting is independent of locomotion. To slow/stop, the program issues `MOVE … SPEED 0/0.5/1` and deceleration occurs over multiple physics ticks.

---

## 4) Perception model

* **FOV:** 120° about heading; **sense range:** **30 m**.
* **Visibility & occlusion:** Segment vs AABB; `occ=1` if any obstacle intersects the line of sight.
* **Top‑K for DSL runtime:**

  * ENEMY: `FRONT=2`, `NEAR=2`
  * FRIEND: `NEAR=2`
  * PROJECTILE: `NEAR=2` (closing)
* **Sector summaries (8×45° bins):** Per class, report counts and mean distance per bin.
* **Derived wall features:** `GAP_DIR(bearing,width)` = largest obstacle‑free & enemy‑sparse angular gap inside FOV; `COVER_LEFT_DIST`, `COVER_RIGHT_DIST` (nearest obstacle edge tangentially left/right within FOV; `∞` if none).

> **Note:** The **DSL‑visible** state is limited to the Top‑K lists, sector summaries, and derived features above. The **LLM** receives these **plus an extended “LLM‑only extras” block** (see §7) to support deeper reasoning.

---

## 5) Signals, roles, and plan notes

* **Signal alphabet (dynamic):** Up to **16 active tokens** (ASCII ≤16 chars). The LLM may propose up to **3 new tokens** each LLM cycle; least‑used tokens auto‑expire (keeps budget stable).
* **Emission:** Each bot may publish **one signal** per LLM cycle (or `NONE`); visible only if the emitter is visible; TTL **1.0 s** after last emission.
* **Roles (optional):** The LLM may define a **role dictionary** (e.g., `[SCOUT, ANCHOR, WING]`), and each bot may adopt one; included in obs for coordination.
* **Plan text:** ≤4 lines free text. Provided back in the next observation as `PLAN_PREV`.

---

## 6) DSL — tiny, additive, absolute headings

**Idea:** Each line is an `IF … THEN …` rule that **adds votes** to actions. No branching flow; just additive hints. Only the **single action with highest total votes** is executed per timestep. Votes carry over using **A1 carryover** (up to **+2.5** into next tick) to provide continuity.

**Single Action Selection:** All actions compete - only ONE action (rotate, move, dodge, or fire) is selected per timestep based on highest vote total.

**Weights:** `{0,1,5}` per action; **5** = decisive, **1** = hint.

**Targets:** `ENEMY.FRONT#k`, `ENEMY.NEAR#k`, `FRIEND.NEAR#k`, `VISIBLE_ENEMYS_CENTROID`, `VISIBLE_FRIENDS_CENTROID`, `GAP_DIR`.

**Rotation semantics (absolute):** 0°=North (+Y), 90°=East (+X), CCW positive.

**Grammar (concise)**

```
RULE := IF <COND> : <ACT>
<COND> := <ATOM> { AND <ATOM> }{0..3}
<ATOM> := <SLOT>.<FIELD> <OP> <VALUE> | <COUNT> <OP> <INT> | <FLAG>
<SLOT> := ENEMY.FRONT#0..2 | ENEMY.NEAR#0..2 | FRIEND.NEAR#0..2 | PROJ.NEAR#0..1 | SELF
<FIELD> := DIST|BEARING|REL_TOWARDS|TTI|HP|VALID|SIGNAL|v|θ|occ
<OP> := < | ≤ | = | ≥ | >
<COUNT> := enemy_count_near | friend_count_near
<FLAG> := proj_imminent | ff_risk_front

<ACT> := { ROTATE TO HEADING <ANGLE> +<W> |
           ROTATE TO TARGET <TARGET> +<W> |
           MOVE <DIR> SPEED <S> +<W> |
           DODGE <DIR> +<W> |
           FIRE {ON|OFF} +<W> }{1..3}

<TARGET> := ENEMY.FRONT#k | ENEMY.NEAR#k | FRIEND.NEAR#k |
            VISIBLE_ENEMYS_CENTROID | VISIBLE_FRIENDS_CENTROID | GAP_DIR
<DIR>   := LEFT | RIGHT | FWD | BACK
<S>     := 0 | 0.5 | 1
<W>     := 1 | 5
<ANGLE> := integer degrees 0..359
```

**Examples**

```
1) IF PROJ.NEAR#0.TTI ≤ 0.25 : DODGE RIGHT +5 ; FIRE OFF +5
2) IF ENEMY.FRONT#0.DIST < 10 : ROTATE TO TARGET ENEMY.FRONT#0 +5 ; FIRE ON +5
3) IF enemy_count_near ≥ 5 : ROTATE TO TARGET GAP_DIR +5
4) IF FRIEND.NEAR#0.SIGNAL = ON_ME : MOVE FWD SPEED 1 +5
5) IF SELF.v > 1.0 AND ENEMY.FRONT#0.DIST < 6 : MOVE FWD SPEED 0.5 +1
6) IF ff_risk_front = 1 : FIRE OFF +5
7) IF SELF.hp ≤ 30 : ROTATE TO HEADING 180 +5 ; MOVE FWD SPEED 1 +5 ; FIRE OFF +5
```

---

## 7) LLM interface — cadence, prompt, and output contract

* **Cadence:** Every **10–50 controller ticks** (default 25; \~0.208 s).
* **Input:**

  * **Observation block** (compact, stable — see §8) **plus**
  * **LLM‑only extras**: rich lists that the DSL does not consume but which help the LLM reason.
  * **Full previous program** (as executed since last LLM turn).
  * **Events since last LLM turn**: human‑readable roll‑up (displacement, actions, shots, hits, min‑TTI, signals, HP delta, etc.).
* **Output (strict):** (a) ≤10 DSL rules; (b) an updated **plan** (≤4 lines); (c) one **signal** token (optionally including up to 3 **new** tokens).
* **Validation & fallback:** Engine parses/limits; invalid/bloated output → fall back to last valid program.

### 7.1 Prompt template (VERBOSE; copy‑paste)

```
You control ONE bot in a fast 2D battle simulator. The current simulation started with N bots (your allies) vs. M bots (your opponents).

== Game rules (always included) ==
• Time: physics 240 Hz; your program runs at 120 Hz; you will be invoked to rewrite it every ~0.2 s.
• Motion is continuous with acceleration and turn limits. New speed/heading take time to realize.
• You can SHOOT while moving. To slow/stop, explicitly command MOVE … SPEED 0/0.5/1.
• Projectiles: ~6 m/s, TTL 2.0 s, visible, dodgeable. Friendly fire is possible.
• Signals: you can emit ONE short token (≤16 chars) per LLM tick. Up to 16 active tokens team‑wide; at most 3 new tokens per tick (least‑used expire).

== Definitions ==
• Headings are absolute: 0°=North (+Y), 90°=East (+X), CCW positive.
• FOV: 120° centered on your heading; sense range 30 m.
• occ=1: target is geometrically occluded by a wall (axis‑aligned rectangle).
• SECTORS: 8 fixed 45° bins; we provide counts and mean distances by class.
• REL_TOWARDS: component of projectile velocity toward you (positive means closing).
• TTI: estimated time to collision in seconds (∞ if not closing).
• GAP_DIR_WALLS =(bearing,width): largest obstacle‑free angular gap within FOV (bearing is absolute; width is degrees). May not be meaningful if walls are not enabled
• GAP_DIR_ENEMY =(bearing,width): largest enemy‑sparse angular gap within FOV (bearing is absolute; width is degrees).

• COVER_LEFT_DIST / COVER_RIGHT_DIST: nearest obstacle edge tangentially left/right (meters; ∞ if none).

== Current observation ==
{OBSERVATION_TEXT_BLOCK}

== LLM‑only extras (rich context for better planning) ==
# These are more detailed than what your DSL program uses.
VISIBLE_ENEMIES_FULL (up to 16):
  - id, pos=(x,y), θ_abs, v, hp, bearing_abs, dist, vel=(vx,vy), occ, last_seen_ticks
VISIBLE_FRIENDS_FULL (up to 8):
  - id, pos, θ_abs, v, hp, bearing_abs, dist, signal, role
VISIBLE_PROJECTILES_FULL (up to 8):
  - id, pos, vel, rel_towards, tti, bearing_abs, shooter_id? (if known)
OBSTACLES_IN_FOV_AABBS (up to 16):
  - [xmin,ymin,xmax,ymax] per obstacle (absolute coordinates)
MAP_META:
  - bounds=[0..W, 0..H], spawn_zones (if any), objective_zones (if any)

== FULL previous program (as executed since your last turn) ==
DSL:
{LAST_PROGRAM_FULL_≤10_LINES}

== Events since your last LLM turn (~0.2 s) ==
• Displacement: moved Δ=(dx=…, dy=…); heading changed Δθ=…°
• Actions taken (controller ticks): ROTATE:…, MOVE:…, DODGE:… (LEFT/RIGHT)
• Firing: … shots; 
• Combat: dealt … dmg (target ids); took … dmg (from bearings …°)
• Projectiles dodged: … (min TTI … s at bearing …°)
• Signals seen: […]; MY signal last=…; ROLE=…
• Health: … → … HP
• PLAN_PREV lines:
- …

== What to output (STRICT) ==
DSL:
1) …
2) …
(≤10 lines)

PLAN:
- line 1
- line 2
- line 3
- line 4

SIGNAL:
TOKEN=UP_TO_16_ASCII_CHARS
OPTIONAL_NEW_TOKENS=[comma,separated,≤3]

OR JSON (engine compiles to DSL and echoes compiled DSL in logs):
{ "mode":"rules_v1", "dsl":[ … ], "plan":[…], "signal":{"token":"…","new_tokens":[…]} }

```

---

## 8) Observation text block (engine → LLM; compact + stable)

```
ARENA=0 TICK=184 DT=0.00833s
TEAM size=40 alive=33 ENEMY_ALIVE=37 SCORE=+3
SELF pos=(12.1,3.0) θ=94 v=3.2 hp=78 ROLE=SCOUT SIGNAL=NONE
ENEMY n=2: E0 d=8.4 bearing_abs=+32 vel=(2.0,0.5) hp=61 occ=0; E1 d=13.2 bearing_abs=-15 vel=(1.2,-0.1) hp=100 occ=1
FRIEND n=1: F0 d=6.3 bearing_abs=-40 signal=ON_ME
PROJ n=1: P0 d=3.1 bearing_abs=+5 rel_towards=+7.8 tti=0.22
SECTORS enemies.counts=[0,1,0,0,0,0,0,0] enemies.mean_d=[∞,13.2,∞,∞,∞,∞,∞,∞]
        friends.counts=[0,0,1,0,0,0,0,0] friends.mean_d=[∞,∞,6.3,∞,∞,∞,∞,∞]
        proj.counts   =[0,0,1,0,0,0,0,0] proj.mean_d   =[∞,∞,3.1,∞,∞,∞,∞,∞]
GAP_DIR bearing=+28 width=36 COVER_LEFT_DIST=5.2 COVER_RIGHT_DIST=∞
PLAN_PREV:
- flank right
- avoid mid
- collapse nearest low hp
```

---

## 9) Episodes, rewards & training

* **Episodes:** terminate on wipe or time T; randomize seeds, spawns, and map variants.
* **Logs:** (obs, plan, DSL, per‑tick actions, outcomes) for every bot; deterministic replays.
* **Rewards (initial weights):**

  * team **win** = +1
  * **damage** = +0.01 per HP dealt, −0.01 per HP taken
  * **friendly fire penalty** = −0.02 per HP
  * **cohesion** = −0.001 per meter (distance to nearest 2 friends, clamped)
  * optional: **objective captures**
* **Supervision:**

  * **DPO/RLHF:** (program, snapshot) preference pairs (human/heuristic).
  * **Behavior cloning:** top‑decile episodes.
  * **Curriculum:** OPEN 10v10 → MIXED 40v40 → OCCLUDED 80v80 (e.g., 30% OCCLUDED in stage 2).

---

## 10) Action resolution & gating

* **Single action arbitration:** sum votes `{0,1,5}` across ALL actions; single winner enacted per timestep; ties → stable priority (DODGE > ROTATE > MOVE > FIRE).
* **Carryover (A1):** winning action contributes up to **+2.5** votes next tick (caps continuity without lock‑in).
* **Dodge priority:** DODGE actions typically get high votes (+5) for immediate threat response.

---

## 11) Implementation plan

**Data layout** (SoA per arena)

* Arrays: `x[], y[], θ[], v[], ω[], hp[], role_id[], signal_id[], flags[]`
* Uniform hash grid for neighbor queries; cell size `max(sense_range/4, 2 m)`
* Obstacles: list of AABBs `[xmin,ymin,xmax,ymax]`

**Core systems**

1. **Physics:** Integrate v/ω with clamps; resolve obstacle sliding.
2. **LOS & occlusion:** segment vs AABB; populate `occ`.
3. **Neighborhood:** Top‑K selectors (front/near) & sector summaries.
4. **Derived features:** `GAP_DIR`, `COVER_*`.
5. **Interpreter:** parse DSL/JSON → rules; additive votes → winners; carryover; map `ROTATE TO TARGET` → absolute heading.
6. **Prompt packer:** emit observation, LLM‑only extras, full previous program, and events roll‑up.
7. **Validator:** strict parsers; fallback to last valid program.
8. **Harness:** batch arenas, seeding, replay, metrics, dataset writer for RLHF/DPO.
9. **Exporter:** 240 Hz logs → glTF 2.0 + JSON sidecar; spline compression (≤2 cm / 0.5°; configurable).

**Sprint 1 (engine MVP, 1–2 weeks)**

* SoA state, physics, obstacles (slide), projectiles, damage, FF gating.
* Neighborhood grid + Top‑K + sectors; occlusion.
* Observation serializer; prompt packer (minimal); DSL parser; action resolver with carryover.
* Deterministic replay logger; seed handling; unit tests (math, occlusion, parser round‑trip).

**Sprint 2 (LLM loop & export, 1–2 weeks)**

* LLM I/O adapter (prompt template, JSON acceptance, compiled‑DSL echo).
* Signals governance (≤16 active; ≤3 new; LRU expiry). Roles.
* Events roll‑up; LLM‑only extras; curriculum toggles (OPEN/OCCLUDED mix).
* glTF/JSON exporter + Unity/Unreal import scripts. Camera recipes.

**Perf micro‑benchmarks**

* Neighbor query throughput; occlusion cost per ray; interpreter ns per rule.
* Target ref: ≥100k bot‑ticks/sec/core on modern desktop CPU (Python); record before/after Numba/WASM.

---

## 12) Example outputs

**Example DSL (10 lines)**

```
1) IF PROJ.NEAR#0.TTI ≤ 0.25 : DODGE RIGHT +5 ; FIRE OFF +5
2) IF ENEMY.FRONT#0.DIST < 10 : ROTATE TO TARGET ENEMY.FRONT#0 +5 ; FIRE ON +5
3) IF enemy_count_near ≥ 5 : ROTATE TO TARGET GAP_DIR +5
4) IF FRIEND.NEAR#0.SIGNAL = ON_ME : MOVE FWD SPEED 1 +5
5) IF SELF.v > 1.0 AND ENEMY.FRONT#0.DIST < 6 : MOVE FWD SPEED 0.5 +1
6) IF ff_risk_front = 1 : FIRE OFF +5
7) IF SELF.hp ≤ 30 : ROTATE TO HEADING 180 +5 ; MOVE FWD SPEED 1 +5 ; FIRE OFF +5
8) IF ENEMY.NEAR#0.occ = 1 : ROTATE TO TARGET GAP_DIR +5
9) IF FRIEND.NEAR#0.DIST ≥ 8 : ROTATE TO TARGET VISIBLE_FRIENDS_CENTROID +1
10) IF ENEMY.FRONT#0.VALID = 0 : ROTATE TO HEADING 90 +1
```

---

## 13) Glossary (LLM‑facing)

* **ARENA:** integer id of the simultaneous arena in a batch.
* **TICK:** controller tick count (120 Hz).
* **θ (theta):** absolute heading in degrees (0°=North, 90°=East; CCW positive).
* **bearing\_abs:** absolute bearing from self to target in degrees.
* **FRONT vs NEAR (targets):** `FRONT` = smallest |bearing| first; `NEAR` = smallest distance first.
* **occ:** 1 if occluded by any AABB obstacle; else 0.
* **REL\_TOWARDS:** positive if projectile is moving toward self along the line of sight.
* **TTI:** time to impact (s) if current relative velocity persists; `∞` if diverging.
* **GAP\_DIR:** largest free & enemy‑sparse angular gap inside FOV; reported as (bearing\_abs,width\_deg).
* **COVER\_LEFT/RIGHT\_DIST:** nearest obstacle edge tangentially left/right within FOV (m; `∞` if none).

---

## 14) Testing checklist

* **Math:** heading normalization; `ROTATE TO TARGET` absolute angle.
* **Occlusion:** segment vs AABB correctness with corner cases.
* **Selectors:** Top‑K (front/near) invariants; sector binning.
* **Interpreter:** vote sums; tie‑break; carryover cap @ 2.5;
* **FF gating:** 1° cone ray test; regression for flicker.
* **Repro:** identical replays with same seed.
* **Exporter:** spline error ≤2 cm / 0.5°; event channel alignment.

---

## 15) Future extensions (non‑blocking)

* **Objectives & zones:** capture points; zone rewards.
* **Squad programs:** per‑squad DSL + per‑bot parameters to save tokens.
* **Stochastic weapons:** small spread; suppression values.
* **Terrain:** slow tiles; slippery tiles (changes damping).
* **Multi‑weapon types:** different ROF/TTK/muzzle speeds.
* **GPU engine:** Madrona/GPUDrive port once Python hotspots are profiled.

---

### TL;DR

A compact, deterministic 2D sim that feeds a **verbose, human‑readable prompt** to a small LLM every \~0.2 s. The LLM returns a **≤10‑line program** (plus a plan and a signal). Bots run that program at **120 Hz** under **continuous physics** and produce epic‑looking but simple behaviors. We batch thousands of episodes and train with **RLHF/DPO/BC**. Python first; accelerate only after profiling.

