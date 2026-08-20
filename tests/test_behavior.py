from src.behavior import PersonIncident,State

def test_post_fall_recovery_transition():
    p=PersonIncident(1)
    p.update('FALLING',.9,.02,.02,now=1.0)
    p.update('FALLEN',.95,.01,.01,now=2.0)
    assert p.state==State.FALLEN
    p.update('FALLEN',.95,.04,.04,now=3.0)
    assert p.state==State.RECOVERING
    for i in range(8): p.update('NORMAL',.95,.0,.0,now=4+i)
    assert p.state==State.RECOVERED
