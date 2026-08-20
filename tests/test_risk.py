from src.risk import calculate_risk

def test_critical_risk():
    r=calculate_risk('FALLEN',.97,30,.04,2,10)
    assert r.level=='CRITICAL'
    assert r.score>=85
