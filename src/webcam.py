import argparse
from .demo import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Context-aware fall detection from a live camera')
    p.add_argument('--camera',type=int,default=0); p.add_argument('--pose-model',default='yolo11n-pose.pt'); p.add_argument('--classifier'); p.add_argument('--report-dir',default='outputs/incidents'); a=p.parse_args()
    # OpenCV camera index is supported directly by the same inference loop.
    run(a.camera,None,a.pose_model,a.classifier,a.report_dir)
