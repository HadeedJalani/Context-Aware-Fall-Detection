from dataclasses import dataclass

@dataclass
class RiskResult:
    score:int
    level:str
    reasons:list[str]

def calculate_risk(state,confidence,ground_seconds,peak_velocity,recovery_attempts,immobile_seconds=0):
    score=0; reasons=[]
    if state=='FALLING': score+=45; reasons.append('active fall')
    elif state=='FALLEN': score+=60; reasons.append('person remains on ground')
    elif state=='RECOVERING': score+=30; reasons.append('recovery in progress')
    if confidence>=.9: score+=10; reasons.append('high detection confidence')
    if ground_seconds>=10: score+=min(20,int(ground_seconds/5)); reasons.append('extended ground duration')
    if peak_velocity>=.02: score+=10; reasons.append('rapid body displacement')
    if recovery_attempts>=2: score+=5; reasons.append('repeated recovery attempts')
    if immobile_seconds>=5: score+=10; reasons.append('prolonged immobility')
    score=max(0,min(100,score))
    level='CRITICAL' if score>=85 else 'HIGH' if score>=70 else 'MODERATE' if score>=40 else 'LOW'
    return RiskResult(score,level,reasons)
