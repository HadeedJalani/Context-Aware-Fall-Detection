from __future__ import annotations
from pathlib import Path
from .behavior import State
from .reporting.incident import build_incident, save_incident
from .risk import calculate_risk

class IncidentManager:
    def __init__(self, report_dir="outputs/incidents"):
        self.report_dir=Path(report_dir); self.active={}; self.closed=set()
    def update(self, person, confidence, movement, now=None):
        risk=calculate_risk(person.state.value,confidence,person.ground_duration,person.peak_velocity,person.recovery_attempts)
        if person.state in (State.FALLING,State.FALLEN,State.RECOVERING):
            self.active[person.track_id]=risk
        if person.state==State.RECOVERED and person.track_id not in self.closed:
            incident=build_incident(person.track_id,"FALL",risk,person)
            save_incident(self.report_dir,incident); self.closed.add(person.track_id)
        return risk
    def highest_priority(self):
        if not self.active: return None
        return max(self.active.items(),key=lambda item:item[1].score)
