import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('machine', type=str)
parser.add_argument('--wandb_name', type=str, help=' ')
parser.add_argument('--load_step', type=int, default=-1, help='The step to load. -1 for the latest one.')
args = parser.parse_args()

with open('sh/dir.json') as f:
    d = json.load(f)
d = d[args.machine]
for m in ['temp']:
    for ns in range(3, 3+1):
        cmd = f"""{d['val_environ']}
            python val.py
            --data_root /sk3d
            --list_dir /workspace/list_files
            --dataset_name our_dataval
            --model_name model_cas
            --num_src {ns}
            --max_d 128
            --interval_scale 1
            --cas_depth_num 32,16,8
            --cas_interv_scale 4,2,1
            --resize 853,683
            --crop 640,512 
            --mode soft
            --result_dir /workspace/mvg_val
            --load_path /workspace/Vis-MVSNet/pretrained_model/sk3d3/{m}
            --wandb_name {args.wandb_name}
            --load_step {args.load_step}
        """
        cmd = ' '.join(cmd.strip().split())
        print(cmd)
        os.system(cmd)
        
#         --load_path {d['save_dir']}/{m}
            # --show_result
            # --write_result