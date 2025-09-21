from typing import Any, Deque, Dict, Iterable, List, Tuple

#--------------- Default example templates (for testing and endpointless dev) ---------------- #
def _get_default_templates() -> List[str]:
    return [
        _tpl_aggressive(),
        _tpl_defensive(),
        _tpl_balanced(),
        _tpl_sniper(),
    ]
def _tpl_aggressive() -> str:
    return '''
def bot_function(observation):
    """Aggressive bot with predictive aim, wall awareness, and friendly-fire avoidance."""
    import math
    visible = observation.get('visible_objects', [])
    allowed = observation.get('allowed_signals', [])
    self_state = observation.get('self', {})
    params = observation.get('params', {})
    memory = observation.get('memory', {})
    if not isinstance(memory, dict):
        memory = {}
    def choose(s):
        return s if s in allowed else 'none'
    def lead_point(sx, sy, tx, ty, tvx, tvy, proj_speed):
        rx, ry = tx - sx, ty - sy
        a = tvx * tvx + tvy * tvy - proj_speed * proj_speed
        b = 2.0 * (rx * tvx + ry * tvy)
        c = rx * rx + ry * ry
        t = 0.0
        if abs(a) < 1e-6:
            t = max(0.0, -c / b) if abs(b) > 1e-6 else 0.0
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                rdisc = math.sqrt(disc)
                t1 = (-b - rdisc) / (2.0 * a)
                t2 = (-b + rdisc) / (2.0 * a)
                cand = [t for t in (t1, t2) if t >= 0.0]
                t = min(cand) if cand else 0.0
        return (tx + tvx * t, ty + tvy * t)
    def friend_in_line_of_fire(sx, sy, ax, ay, friends, radius=0.8, cone_deg=12.0):
        vx, vy = ax - sx, ay - sy
        L = math.hypot(vx, vy)
        if L < 1e-6: return True
        ux, uy = vx / L, vy / L
        cos_thresh = math.cos(math.radians(cone_deg))
        for f in [o for o in visible if o.get('type') == 'friend']:
            fx, fy = f.get('x', 0.0), f.get('y', 0.0)
            rfx, rfy = fx - sx, fy - sy
            proj = rfx * ux + rfy * uy
            if proj < 0.0 or proj > L: continue
            mag = math.hypot(rfx, rfy)
            if mag > 1e-6 and ((rfx * ux + rfy * uy) / mag) < cos_thresh: continue
            px, py = ux * proj, uy * proj
            dx, dy = rfx - px, rfy - py
            if math.hypot(dx, dy) < radius: return True
        return False
    enemies = [o for o in visible if o.get('type') == 'enemy']
    friends = [o for o in visible if o.get('type') == 'friend']
    projectiles = [o for o in visible if o.get('type') == 'projectile']
    walls = [o for o in visible if o.get('type') == 'wall']
    sx, sy = self_state.get('x', 0.0), self_state.get('y', 0.0)
    can_fire = bool(self_state.get('can_fire', False))
    proj_speed = float(params.get('proj_speed', 12.0))
    for p in projectiles:
        px, py = p.get('x', 0.0), p.get('y', 0.0)
        vx, vy = p.get('velocity_x', 0.0), p.get('velocity_y', 0.0)
        rx, ry = sx - px, sy - py
        v2 = vx * vx + vy * vy
        if v2 <= 1e-5: continue
        tca = - (rx * vx + ry * vy) / v2
        if 0.0 <= tca <= 0.6:
            cx, cy = rx + vx * tca, ry + vy * tca
            if math.hypot(cx, cy) < 1.2:
                ang = math.degrees(math.atan2(vy, vx))
                dodge = (ang + 90.0) % 360.0
                return {'action': 'dodge','direction': dodge,'signal': choose('moving_to_cover'),'memory': memory}
    if not enemies:
        if walls:
            w = min(walls, key=lambda w: w.get('distance', 999.0))
            if w.get('distance', 999.0) < 2.0:
                away = (w.get('angle', 0.0) + 180.0) % 360.0
                return {'action': 'dodge','direction': away,'signal': choose('regrouping'),'memory': memory}
        return {'action': 'rotate','angle': 60.0,'signal': choose('ready'),'memory': memory}
    enemies.sort(key=lambda e: (1.0/max(0.1, e.get('distance', 999.0)) + 0.3*(1.0/max(1, e.get('hp', 100)))), reverse=True)
    t = enemies[0]
    if t.get('distance', 999.0) < 2.0:
        side = (t.get('angle', 0.0) + 90.0) % 360.0
        return {'action': 'dodge','direction': side,'signal': choose('retreating'),'memory': memory}
    tx, ty = float(t.get('x', 0.0)), float(t.get('y', 0.0))
    tvx, tvy = float(t.get('velocity_x', 0.0)), float(t.get('velocity_y', 0.0))
    ax, ay = lead_point(sx, sy, tx, ty, tvx, tvy, proj_speed)
    if can_fire and not friend_in_line_of_fire(sx, sy, ax, ay, friends) and t.get('distance', 999.0) <= 12.0:
        return {'action': 'fire','target_x': ax,'target_y': ay,'signal': choose('firing'),'memory': memory}
    flank = math.atan2(tvy, tvx) + math.pi/2.0 if (tvx*tvx+tvy*tvy)>0.01 else math.radians(t.get('angle', 0.0)+90.0)
    d = max(3.0, min(8.0, t.get('distance', 8.0)-1.5))
    return {'action':'move','target_x': tx - d*math.cos(flank),'target_y': ty - d*math.sin(flank),'signal': choose('attacking'),'memory': memory}
'''
def _tpl_defensive() -> str:
    return '''
def bot_function(observation):
    """Defensive bot: kites enemies, uses predictive aim, and avoids friendly fire."""
    import math
    visible = observation.get('visible_objects', [])
    allowed = observation.get('allowed_signals', [])
    self_state = observation.get('self', {})
    params = observation.get('params', {})
    def choose(s): return s if s in allowed else 'none'
    def lead_point(sx, sy, tx, ty, tvx, tvy, ps):
        rx, ry = tx - sx, ty - sy
        a = tvx*tvx + tvy*tvy - ps*ps
        b = 2.0 * (rx*tvx + ry*tvy)
        c = rx*rx + ry*ry
        t = 0.0
        if abs(a) < 1e-6: t = max(0.0, -c/b) if abs(b)>1e-6 else 0.0
        else:
            d = b*b - 4.0*a*c
            if d>=0.0:
                rd = math.sqrt(d)
                t1 = (-b-rd)/(2.0*a)
                t2 = (-b+rd)/(2.0*a)
                ts = [t for t in (t1,t2) if t>=0.0]
                t = min(ts) if ts else 0.0
        return (tx + tvx*t, ty + tvy*t)
    enemies = [o for o in visible if o.get('type')=='enemy']
    friends = [o for o in visible if o.get('type')=='friend']
    walls = [o for o in visible if o.get('type')=='wall']
    sx, sy = self_state.get('x',0.0), self_state.get('y',0.0)
    ps = float(params.get('proj_speed',12.0))
    if walls:
        w = min(walls, key=lambda w:w.get('distance',999.0))
        if w.get('distance',999.0) < 1.5:
            away = (w.get('angle',0.0)+180.0)%360.0
            return {'action':'dodge','direction':away,'signal':choose('moving_to_cover'),'memory':{}}
    if not enemies:
        return {'action':'rotate','angle':0.0,'signal':choose('ready'),'memory':{}}
    enemies.sort(key=lambda e:(0.7*(1.0/max(1,e.get('hp',100)))+0.3*(1.0/max(0.1,e.get('distance',999.0)))), reverse=True)
    t = enemies[0]
    d = float(t.get('distance',999.0))
    if d < 6.0:
        away = (t.get('angle',0.0)+180.0)%360.0
        rad = math.radians(away)
        step = max(4.0, 10.0 - d)
        return {'action':'move','target_x': sx+step*math.cos(rad),'target_y': sy+step*math.sin(rad),'signal':choose('retreating'),'memory':{}}
    if self_state.get('can_fire',False) and d<=14.0:
        ax, ay = lead_point(sx, sy, t.get('x',0.0), t.get('y',0.0), t.get('velocity_x',0.0), t.get('velocity_y',0.0), ps)
        return {'action':'fire','target_x':ax,'target_y':ay,'signal':choose('cover_fire'),'memory':{}}
    theta = math.radians(t.get('angle',0.0) + 90.0)
    r = max(0.0, d - 10.0)
    return {'action':'move','target_x': t.get('x',0.0)-r*math.cos(theta),'target_y': t.get('y',0.0)-r*math.sin(theta),'signal':choose('advancing'),'memory':{}}
'''
def _tpl_balanced() -> str:
    return '''
def bot_function(observation):
    """Balanced bot: coordinates via signals, predicts aim, avoids friendly fire."""
    import math
    visible = observation.get('visible_objects', [])
    allowed = observation.get('allowed_signals', [])
    self_state = observation.get('self', {})
    params = observation.get('params', {})
    def choose(s): return s if s in allowed else 'none'
    def lead_point(sx, sy, tx, ty, tvx, tvy, ps):
        rx, ry = tx - sx, ty - sy
        a = tvx*tvx + tvy*tvy - ps*ps
        b = 2.0 * (rx*tvx + ry*tvy)
        c = rx*rx + ry*ry
        t = 0.0
        if abs(a) < 1e-6: t = max(0.0, -c/b) if abs(b)>1.0e-6 else 0.0
        else:
            d = b*b - 4.0*a*c
            if d>=0.0:
                rd = math.sqrt(d)
                t1 = (-b-rd)/(2.0*a)
                t2 = (-b+rd)/(2.0*a)
                ts = [t for t in (t1,t2) if t>=0.0]
                t = min(ts) if ts else 0.0
        return (tx + tvx*t, ty + tvy*t)
    enemies = [o for o in visible if o.get('type')=='enemy']
    friends = [o for o in visible if o.get('type')=='friend']
    sx, sy = self_state.get('x',0.0), self_state.get('y',0.0)
    ps = float(params.get('proj_speed',12.0))
    if not enemies:
        return {'action':'rotate','angle':315.0,'signal':choose('ready'),'memory':{}}
    focus = None
    for f in friends:
        if f.get('signal') in ('focus_fire','enemy_spotted'): focus = f; break
    enemies.sort(key=lambda e:(1.0/max(0.1,e.get('distance',999.0)) + 0.5*(1.0/max(1,e.get('hp',100)))), reverse=True)
    t = enemies[0]
    if focus is not None:
        fx, fy = focus.get('x',0.0), focus.get('y',0.0)
        t = min(enemies, key=lambda e:(e.get('x',0.0)-fx)**2 + (e.get('y',0.0)-fy)**2)
    d = float(t.get('distance',999.0))
    ax, ay = lead_point(sx, sy, t.get('x',0.0), t.get('y',0.0), t.get('velocity_x',0.0), t.get('velocity_y',0.0), ps)
    if self_state.get('can_fire',False) and d<=12.0:
        return {'action':'fire','target_x':ax,'target_y':ay,'signal':choose('firing'),'memory':{}}
    flank = math.radians(t.get('angle',0.0) + 90.0)
    off = min(max(d - 8.0, 2.0), 6.0)
    return {'action':'move','target_x': t.get('x',0.0)-off*math.cos(flank),'target_y': t.get('y',0.0)-off*math.sin(flank),'signal':choose('attacking'),'memory':{}}
'''
def _tpl_sniper() -> str:
    return '''
def bot_function(observation):
    """Sniper bot that prefers long-range engagement and overwatch."""
    import math
    visible = observation.get('visible_objects', [])
    allowed = observation.get('allowed_signals', [])
    self_state = observation.get('self', {})
    def choose(s): return s if s in allowed else 'none'
    projectiles = [o for o in visible if o.get('type')=='projectile']
    friends = [o for o in visible if o.get('type')=='friend']
    for p in projectiles:
        if p.get('distance',999.0) < 1.8:
            dodge = (p.get('angle',0.0) + 90.0) % 360.0
            return {'action':'dodge','direction':dodge,'signal':choose('moving_to_cover'),'memory':{}}
    enemies = [o for o in visible if o.get('type')=='enemy']
    if not enemies:
        trouble = any(f.get('signal') in ('need_backup','retreating') for f in friends)
        return {'action':'rotate','angle':270.0,'signal': choose('watching_flank' if trouble else 'ready'),'memory':{}}
    enemies.sort(key=lambda e:e.get('distance',999.0))
    long = [e for e in enemies if 8.0 <= e.get('distance',0.0) <= 15.0]
    t = long[0] if long else enemies[0]
    if self_state.get('can_fire',False):
        return {'action':'fire','target_x': t.get('x'),'target_y': t.get('y'),'signal': choose('cover_fire'),'memory':{}}
    d = t.get('distance',999.0)
    if d < 8.0:
        away = math.radians((t.get('angle',0.0) + 180.0) % 360.0)
        ret = max(0.0, 12.0 - d)
        return {'action':'move','target_x': self_state.get('x',0.0) + ret*math.cos(away),'target_y': self_state.get('y',0.0) + ret*math.sin(away),'signal': choose('retreating'),'memory':{}}
    opt = 10.0
    ang = math.radians(t.get('angle',0.0))
    return {'action':'move','target_x': t.get('x',0.0)-opt*math.cos(ang),'target_y': t.get('y',0.0)-opt*math.sin(ang),'signal': choose('advancing'),'memory':{}}
'''