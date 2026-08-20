from pathlib import Path
import json
from .pdf_report import create_pdf_report

def build_incident(person_id,event,risk,behavior):
    recovered=behavior.state.value in ('RECOVERED','NORMAL')
    action='IMMEDIATE ASSISTANCE' if risk.level=='CRITICAL' else 'CHECK PERSON' if risk.level=='HIGH' else 'MONITOR'
    return {'person_id':person_id,'event':event,'risk_score':risk.score,'risk_level':risk.level,'confidence':behavior.peak_confidence,'fall_duration':behavior.fall_duration,'ground_duration':behavior.ground_duration,'peak_velocity':behavior.peak_velocity,'recovery_attempts':behavior.recovery_attempts,'recovered':recovered,'recommended_action':action,'reasons':risk.reasons}

def save_incident(report_dir,incident):
    report_dir=Path(report_dir); report_dir.mkdir(parents=True,exist_ok=True); stem=f"incident_person_{incident['person_id']}"
    (report_dir/f'{stem}.json').write_text(json.dumps(incident,indent=2))
    (report_dir/f'{stem}.txt').write_text('FALL INCIDENT REPORT\n\n'+'\n'.join(f'{k}: {v}' for k,v in incident.items()))
    create_pdf_report(report_dir/f'{stem}.pdf',incident)
