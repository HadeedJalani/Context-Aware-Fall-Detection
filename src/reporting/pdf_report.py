from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf_report(path,incident):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet(); doc=SimpleDocTemplate(str(path),pagesize=A4)
    story=[Paragraph('Fall Incident Report',styles['Title']),Spacer(1,12)]
    rows=[['Person ID',incident['person_id']],['Event',incident['event']],['Risk',f"{incident['risk_level']} ({incident['risk_score']}/100)"],['Confidence',f"{incident['confidence']:.1%}"],['Fall duration',f"{incident['fall_duration']:.2f} sec"],['Ground duration',f"{incident['ground_duration']:.2f} sec"],['Peak fall velocity',f"{incident['peak_velocity']:.4f}"],['Recovery attempts',incident['recovery_attempts']],['Recovered','Yes' if incident['recovered'] else 'No'],['Recommended action',incident['recommended_action']]]
    t=Table(rows,colWidths=[180,320]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.grey),('BACKGROUND',(0,0),(0,-1),colors.whitesmoke),('PADDING',(0,0),(-1,-1),7)]))
    story += [t,Spacer(1,16),Paragraph('Research prototype only — not a certified medical or emergency-response device.',styles['Normal'])]
    doc.build(story)
