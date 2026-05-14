import numpy as np


class CriticalObstacleManager:
    """
    Manages critical dynamic obstacle scenarios for C3 / C4.

    Each scenario defines:
      - def_name:         DEF name of the Webots node (must exist in the .wbt world)
      - trigger_distance: ego–obstacle 2D distance (m) that activates movement
      - move_direction:   3-D vector (will be normalised internally)
      - move_distance:    total displacement (m)
      - speed:            displacement per simulation step (m/step)

    NOTE: distance is computed in the X-Z plane (indices 0 and 2),
    which is the ground plane in Webots (Y is vertical).

    Usage:
        manager = CriticalObstacleManager(supervisor, translation_field)
        # inside env.reset():  manager.reset()
        # inside env.step():   manager.step()  — call BEFORE robot.step()
    """

    SCENARIOS = [
        {
            "def_name": "PEDESTRIAN_1",
            "label": "Pedestre",

            # Do not trigger immediately at episode start
            "min_step": 350,

            # Spawn dynamically near the ego vehicle
            "spawn_in_front": True,

            # Higher value = pedestrian appears further ahead of the car
            "front_offset": 15.0,

            # Negative/positive controls the side of the road
            "side_offset": -5.0,

            # Pedestrian crosses laterally
            "move_direction": [0.0, 1.0, 0.0],
            "move_distance": 12.0,
            "speed": 0.08,
        },
        {
            "def_name": "VEHICLE_1",
            "label": "Automóvel",

            # Trigger after the pedestrian scenario
            "min_step": 300,

            "move_direction": [-1.0, 0.0, 0.0],
            "move_distance": 18.0,
            "speed": 0.10,
        },
    ]

    def __init__(self, supervisor, vehicle_translation_field):
        self.supervisor    = supervisor
        self.vehicle_field = vehicle_translation_field
        self.obstacles     = []
        self.current_step = 0
        self._load()

    def _load(self):
        for s in self.SCENARIOS:
            node = self.supervisor.getFromDef(s["def_name"])
            if node is None:
                print(f"[CriticalObstacles] WARNING: '{s['def_name']}' not found in world — skipped.")
                print(f"  Make sure you opened worlds/city_obstacles.wbt (not city_default.wbt).")
                continue

            tf  = node.getField("translation")
            pos = np.array(tf.getSFVec3f(), dtype=np.float32)
            d   = np.array(s["move_direction"], dtype=np.float32)
            d  /= np.linalg.norm(d) + 1e-8

            self.obstacles.append({
                **s,
                "node":        node,
                "tf":          tf,
                "start":       pos.copy(),
                "end":         pos + d * s["move_distance"],
                "trigger_pos": pos.copy(),
                "direction":   d,
                "active":      False,
                "completed":   False,
            })
            print(f"[CriticalObstacles] Loaded: {s['label']} ({s['def_name']}) at {pos}")

    def reset(self):
        self.current_step = 0
        """Call from env.reset() — restores all obstacles to their start positions."""
        for o in self.obstacles:
            o["tf"].setSFVec3f(o["start"].tolist())
            o["node"].resetPhysics()
            o["active"]    = False
            o["completed"] = False

    def step(self):
        """
        Call from env.step() BEFORE robot.step().
        Checks trigger distances and moves active obstacles.
        """
        self.current_step += 1

        ego = np.array(self.vehicle_field.getSFVec3f(), dtype=np.float32)

        for o in self.obstacles:
            if o["completed"]:
                continue

            if not o["active"]:
                triggered = False

                # Time-based trigger: useful for spawning the pedestrian later
                if "min_step" in o:
                    triggered = self.current_step >= o["min_step"]

                # Distance-based trigger: useful for fixed obstacles
                elif "trigger_distance" in o:
                    dist = np.linalg.norm(ego[[0, 1]] - o["trigger_pos"][[0, 1]])
                    triggered = dist <= o["trigger_distance"]

                if triggered:
                    if o.get("spawn_in_front", False):
                        spawn = ego.copy()

                        # More negative/positive depends on the direction of the car.
                        # Since this currently places the pedestrian in front, keep the minus.
                        spawn[0] = ego[0] - o["front_offset"]
                        spawn[1] = ego[1] + o["side_offset"]
                        spawn[2] = o["start"][2]

                        direction = o["direction"]
                        o["start"] = spawn.copy()
                        o["end"] = spawn + direction * o["move_distance"]
                        o["trigger_pos"] = spawn.copy()

                        o["tf"].setSFVec3f(spawn.tolist())
                        o["node"].resetPhysics()

                        print(
                            f"[CriticalObstacles] Spawned {o['label']} "
                            f"near ego at step {self.current_step}: {spawn}"
                        )

                    print(f"[CriticalObstacles] TRIGGERED: {o['label']} at step {self.current_step}")
                    o["active"] = True

            if o["active"]:
                self._move(o)

    def _move(self, o):
        cur       = np.array(o["tf"].getSFVec3f(), dtype=np.float32)
        remaining = np.linalg.norm(o["end"] - cur)

        if remaining < 0.05:
            o["tf"].setSFVec3f(o["end"].tolist())
            o["node"].resetPhysics()
            o["active"]    = False
            o["completed"] = True
            print(f"[CriticalObstacles] Finished: {o['label']}")
            return

        new_pos = cur + o["direction"] * o["speed"]
        o["tf"].setSFVec3f(new_pos.tolist())
        o["node"].resetPhysics()
