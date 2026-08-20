from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time

class State(str, Enum):
    NORMAL='NORMAL'; FALLING='FALLING'; FALLEN='FALLEN'; RECOVERING='RECOVERING'; RECOVERED='RECOVERED'

@dataclass
class PersonIncident:
    track_id:int
    state:State=State.NORMAL
    fall_start:float|None=None
    ground_start:float|None=None
    recovery_attempts:int=0
    peak_velocity:float=0.0
    peak_confidence:float=0.0
    last_movement:float=field(default_factory=time.time)
    recovery_frames:int=0

    def update(self,label,confidence,velocity,movement,now=None):
        now=time.time() if now is None else now
        self.peak_velocity=max(self.peak_velocity,abs(float(velocity)))
        self.peak_confidence=max(self.peak_confidence,float(confidence))
        if movement>.01: self.last_movement=now
        if label=='FALLING': self.fall_start=self.fall_start or now; self.state=State.FALLING
        elif label=='FALLEN':
            self.fall_start=self.fall_start or now; self.ground_start=self.ground_start or now; self.state=State.FALLEN; self.recovery_frames=0
        elif label=='RECOVERING':
            if self.state==State.FALLEN: self.recovery_attempts+=1
            self.state=State.RECOVERING; self.recovery_frames=0
        elif label=='NORMAL':
            if self.state in (State.FALLEN,State.RECOVERING):
                self.recovery_frames+=1
                if self.recovery_frames>=8: self.state=State.RECOVERED
            else: self.state=State.NORMAL

    @property
    def ground_duration(self): return 0.0 if self.ground_start is None else time.time()-self.ground_start
    @property
    def fall_duration(self): return 0.0 if self.fall_start is None or self.ground_start is None else self.ground_start-self.fall_start
