from __future__ import annotations
import math
import numpy as np

def angle_from_vertical(a, b):
    v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return float(abs(math.degrees(math.atan2(v[0], -v[1]))))

def joint_angle(a, b, c):
    a, b, c = map(lambda x: np.asarray(x, dtype=float), (a, b, c))
    u, v = a - b, c - b
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den < 1e-8: return 0.0
    return float(math.degrees(math.acos(np.clip(np.dot(u, v) / den, -1, 1))))

def extract_features(keypoints, previous=None, previous_previous=None, dt=1/30):
    """Return the 9 context-aware temporal features for a COCO-17 pose."""
    kp = np.asarray(keypoints, dtype=float)
    if kp.shape[0] < 17: return np.zeros(9, dtype=np.float32)
    ls, rs = kp[5], kp[6]; lh, rh = kp[11], kp[12]
    lk, rk, la, ra = kp[13], kp[14], kp[15], kp[16]
    hip = (lh[:2] + rh[:2]) / 2; shoulder = (ls[:2] + rs[:2]) / 2
    center = (hip + shoulder) / 2
    visible = kp[kp[:,2] > .2]
    w = float(visible[:,0].max()-visible[:,0].min()) if len(visible) else 0
    h = float(visible[:,1].max()-visible[:,1].min()) if len(visible) else 0
    shape = w / max(h, 1e-6)
    angles = [joint_angle(ls[:2],lh[:2],lk[:2]), joint_angle(rs[:2],rh[:2],rk[:2]), joint_angle(lh[:2],lk[:2],la[:2]), joint_angle(rh[:2],rk[:2],ra[:2])]
    if previous is None:
        vel=acc=immobility=0.0
    else:
        p=np.asarray(previous,float); pc=((p[11,:2]+p[12,:2])/2+(p[5,:2]+p[6,:2])/2)/2
        vel=float((center[1]-pc[1])/max(dt,1e-6)); immobility=float(np.linalg.norm(center-pc)<.01)
        acc=0.0
        if previous_previous is not None:
            pp=np.asarray(previous_previous,float); ppc=((pp[11,:2]+pp[12,:2])/2+(pp[5,:2]+pp[6,:2])/2)/2
            prev_vel=float((pc[1]-ppc[1])/max(dt,1e-6))
            acc=float((vel-prev_vel)/max(dt,1e-6))
    return np.array([angle_from_vertical(shoulder,hip),center[1],shape,vel,acc,float(np.mean(angles)),float(np.std(angles)),immobility,float(np.mean(kp[:,2]))],dtype=np.float32)
