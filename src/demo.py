import argparse
import cv2
from ultralytics import YOLO
from .features import extract_features
from .behavior import PersonIncident
from .risk import calculate_risk
from .model import FallLSTM
from .sequence import TemporalBuffer, predict

def run(input_path, output_path=None, pose_model='yolo11n-pose.pt', classifier_path=None):
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError(f'Could not open {input_path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    writer=None
    if output_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'); w,h=int(cap.get(3)),int(cap.get(4)); writer=cv2.VideoWriter(output_path,fourcc,fps,(w,h))
    model=YOLO(pose_model); incidents={}; previous={}; previous_previous={}; temporal=TemporalBuffer(30,9)
    classifier=None; mean=std=None
    if classifier_path: classifier,ckpt=FallLSTM.from_checkpoint(classifier_path); mean=ckpt.get('feature_mean'); std=ckpt.get('feature_std')
    while True:
        ok,frame=cap.read()
        if not ok: break
        result=model.track(frame,persist=True,tracker='botsort.yaml',verbose=False)[0]
        if result.keypoints is not None and result.boxes is not None:
            ids=result.boxes.id; ids=ids.int().cpu().tolist() if ids is not None else list(range(len(result.keypoints.data)))
            for i,track_id in enumerate(ids):
                kp=result.keypoints.data[i].cpu().numpy(); feat=extract_features(kp,previous.get(track_id),previous_previous.get(track_id),1/max(fps,1))
                previous_previous[track_id]=previous.get(track_id,kp); previous[track_id]=kp
                ready=temporal.append(track_id,feat)
                if classifier and ready: label,confidence,_=predict(classifier,temporal,track_id,mean,std)
                else: label,confidence='NORMAL',float(feat[-1])
                person=incidents.setdefault(track_id,PersonIncident(track_id)); person.update(label,confidence,feat[3],abs(float(feat[3])))
                risk=calculate_risk(person.state.value,confidence,person.ground_duration,person.peak_velocity,person.recovery_attempts)
                x1,y1,x2,y2=result.boxes.xyxy[i].cpu().numpy().astype(int)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
                cv2.putText(frame,f'ID {track_id} {person.state.value} RISK {risk.score}',(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        if writer: writer.write(frame)
    cap.release()
    if writer: writer.release()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output'); p.add_argument('--pose-model',default='yolo11n-pose.pt'); p.add_argument('--classifier')
    a=p.parse_args(); run(a.input,a.output,a.pose_model,a.classifier)
