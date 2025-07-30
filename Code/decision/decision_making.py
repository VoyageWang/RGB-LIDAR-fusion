import time
import numpy as np
from collections import deque

class VPStateTracker:
    def __init__(self):
        self.distance_history = deque(maxlen=5)
        self.angle_history = deque(maxlen=5)
        self.last_seen = time.time()

    def update(self, distance, angle):
        self.distance_history.append(distance)
        self.angle_history.append(angle)
        self.last_seen = time.time()

    def is_stale(self, timeout_sec=0.3):
        return time.time() - self.last_seen > timeout_sec

    def get_filtered(self):
        return {
            "distance": self.kalman_filter(self.distance_history),
            "angle": self.kalman_filter(self.angle_history)
        }

    @staticmethod
    def kalman_filter(values):
        if not values:
            return None
        x = np.mean(values)
        p = 1.0
        q = 0.1
        r = 1.0
        for z in values:
            p += q
            k = p / (p + r)
            x += k * (z - x)
            p *= (1 - k)
        return x

class SpeedTracker:
    def __init__(self):
        self.history = deque(maxlen=5)
        self.last_seen = time.time()

    def update(self, speed):
        self.history.append(speed)
        self.last_seen = time.time()

    def is_stale(self, timeout_sec=0.3):
        return time.time() - self.last_seen > timeout_sec

    def get_filtered(self):
        return VPStateTracker.kalman_filter(self.history)

class DecisionEngine:
    def __init__(self):
        self.vp_trackers = {}  # {vehicle_id: {person_id: VPStateTracker}}
        self.vehicle_speed_trackers = {}  # {vehicle_id: SpeedTracker}
        self.last_stop_sent = {}  # {vehicle_id: bool}
        self.hysteresis_flag = False

    def run_decision(self, frame_dict:dict,bsm_data:dict):
        self.update_trackers_from_frame(frame_dict, bsm_data)
        v_id = self.get_vehicle_id_from_bsm(bsm_data)
        if v_id is None:
            return self.build_none_event()
        return self.decide_from_all_pairs(v_id)
    
    def get_vehicle_id_from_bsm(self, bsm_data):
        if bsm_data is None:
            return None
        return bsm_data.get("vehicle_id", None)

    def get_vehicle_speed_from_bsm(self, bsm_data):
        if bsm_data is None:
            return 0.0
        return bsm_data.get("speed_bsm_kmh", 0.0)

    def get_first_vehicle_id(self, frame_dict):
        for view_key in ["infrastructure_view1", "infrastructure_view2"]:
            views = frame_dict.get(view_key, [])
            if not views:
                continue
            vehicles = views[0].get("vehicles", [])
            if not vehicles:
                continue
            return vehicles[0].get("id", None)
        return None

    def update_trackers_from_frame(self, frame_dict, bsm_data):
        current_vp_pairs = set()
        v_id = self.get_vehicle_id_from_bsm(bsm_data)
        if v_id is None:
            return
        speed = self.get_vehicle_speed_from_bsm(bsm_data)       
        if v_id not in self.vehicle_speed_trackers:
            self.vehicle_speed_trackers[v_id] = SpeedTracker()
        self.vehicle_speed_trackers[v_id].update(speed)
        print(f"bsm_data:speed{speed}, v_id:{v_id}")

        if frame_dict is None:
            return

        for view_key in ["infrastructure_view1", "infrastructure_view2"]:
            views = frame_dict.get(view_key, [])
            if not views:
                continue

            vehicles = views[-1].get("vehicles", [])
            if not vehicles:
                continue

            vehicle = vehicles[0]
            # speed = bsm_data.get("speed_bsm_kmh", vehicle.get("speed_kmh", 0.0))

            for p in vehicle.get("distances_to_persons", []):
                p_id = p.get("person_id", "unknown_person")
                distance = p.get("distance_to_person", float('inf'))
                angle = p.get("angle_degrees", float('inf'))

                if v_id not in self.vp_trackers:
                    self.vp_trackers[v_id] = {}

                if p_id not in self.vp_trackers[v_id]:
                    self.vp_trackers[v_id][p_id] = VPStateTracker()

                self.vp_trackers[v_id][p_id].update(distance, angle)
                current_vp_pairs.add((v_id, p_id))

        self.cleanup_stale_pairs(current_vp_pairs)

    def cleanup_stale_pairs(self, current_vp_pairs):
        for v_id in list(self.vp_trackers.keys()):
            for p_id in list(self.vp_trackers[v_id].keys()):
                if (v_id, p_id) not in current_vp_pairs:
                    tracker = self.vp_trackers[v_id][p_id]
                    if tracker.is_stale():
                        del self.vp_trackers[v_id][p_id]
            if not self.vp_trackers[v_id]:
                del self.vp_trackers[v_id]

        for v_id in list(self.vehicle_speed_trackers.keys()):
            if self.vehicle_speed_trackers[v_id].is_stale():
                del self.vehicle_speed_trackers[v_id]

    def decide_from_all_pairs(self, vehicle_id):
        # if vehicle_id not in self.vp_trackers:
        #     return self.build_none_event()

        speed = 0.0
        if vehicle_id in self.vehicle_speed_trackers:
            speed = self.vehicle_speed_trackers[vehicle_id].get_filtered() or 0.0

        candidates = []
        if vehicle_id in self.vp_trackers:
            for p_id, tracker in self.vp_trackers[vehicle_id].items():
                filtered = tracker.get_filtered()
                if filtered["angle"] is not None and filtered["distance"] is not None:
                    candidates.append(filtered)

        toward_people = [c for c in candidates if c["angle"] < 8]
        if not toward_people: # 如果没有朝向的行人，则距离赋值为inf，角度赋值为90.0
            toward_people = [{"distance": float('inf'), "angle": 90.0}]

        target = min(toward_people, key=lambda c: c["distance"])
        return self.make_decision_logic(vehicle_id, target["distance"], speed)

    def make_decision_logic(self, vehicle_id, dist, speed):
        if self.hysteresis_flag:
            if speed < 30:
                event_status = 0
                event_type = 0
                event_desc = ''
                self.hysteresis_flag = False
            else:
                event_status = 3
                event_type = 913
                event_desc = ''
        else:
            if speed > 40:
                event_status = 3
                event_type = 913
                event_desc = ''
                self.hysteresis_flag = True
            else:
                event_status = 0
                event_type = 0
                event_desc = ''

        if dist <= 30 and dist > 15:
            event_status = 3
            event_type = 912
            event_desc = '20'
        elif dist <= 15:
            event_status = 3
            event_type = 911
            event_desc = ''

        return event_status, event_type, event_desc

    def build_none_event(self):
        event_status = 0
        event_type = 0
        event_desc = ''
        return event_status, event_type, event_desc
