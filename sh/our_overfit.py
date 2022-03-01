import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('machine', type=str)
args = parser.parse_args()

with open('sh/dir.json') as f:
    d = json.load(f)
d = d[args.machine]
cmd = f"""{d['train_environ']}
    python train.py
    --num_workers 4
    --data_root /sk3d
    --list_dir /workspace/list_files
    --dataset_name our
    --model_name model_cas
    --num_src 3
    --max_d 128
    --interval_scale 1
    --cas_depth_num 32,16,8
    --cas_interv_scale 4,2,1
    --resize 853,683
    --crop 640,512
    --mode soft
    --num_samples 5000
    --batch_size 2
    --job_name temp
    --save_dir /workspace/Vis-MVSNet/pretrained_model/overfit
    
"""
cmd = ' '.join(cmd.strip().split())
print(cmd)
os.system(cmd)
# --load_path /workspace/Vis-MVSNet/pretrained_model/dragon/temp