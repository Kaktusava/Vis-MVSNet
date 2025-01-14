import argparse
import gc
import os
import time
import random
import sys
import importlib
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
# from apex import amp
from apex import parallel as apex_parallel

# import data.dtu as dtu, data.sceneflow as sceneflow, data.blended as bld
# from core.model_cas import Model, Loss
from utils.io_utils import load_model, save_model
from utils.preproc import recursive_apply
from utils.utils import NanError

from matplotlib import cm as cm
from PIL import Image

import wandb

# wandb.init(project="Vis-MVSNet_MVG", entity="kaktusava")
wandb.init(project="vis-mvsnet-mvg_sk3dSetup", entity="kaktusava")

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=8, help='The number of workers for the dataloader. 0 to disable the async loading.')
# parser.add_argument('--num_gpus', type=int, default=1)

parser.add_argument('--data_root', type=str, help='The root dir of the data.')
parser.add_argument('--list_dir', type=str, help='The list dir of the data.')
parser.add_argument('--dataset_name', type=str, default='blended', help='The name of the dataset. Should be identical to the dataloader source file. e.g. blended refers to data/blended.py.')
parser.add_argument('--model_name', type=str, default='model_cas', help='The name of the model. Should be identical to the model source file. e.g. model_cas refers to core/model_cas.py.')

parser.add_argument('--num_src', type=int, default=3, help='The number of source views.')
parser.add_argument('--max_d', type=int, default=128, help='The standard max depth number.')
parser.add_argument('--interval_scale', type=float, default=1., help='The standard interval scale.')
parser.add_argument('--cas_depth_num', type=str, default='32,16,8', help='The depth number for each stage.')
parser.add_argument('--cas_interv_scale', type=str, default='4,2,1', help='The interval scale for each stage.')
parser.add_argument('--resize', type=str, default='768,576', help='The size of the preprocessed input resized from the original one.')
parser.add_argument('--crop', type=str, default='640,512', help='The size of the preprocessed input cropped from the resized one.')

parser.add_argument('--mode', type=str, default='soft', choices=['soft', 'hard', 'uwta', 'maxpool', 'average'], help='The fusion strategy.')
parser.add_argument('--occ_guide', action='store_true', default=False, help='Deprecated')

# parser.add_argument('--lr', type=str, default='1e-3,.5e-3,.25e-3,.125e-3,.0625e-3,.03125e-3,.015625e-3', help='Learning rate under piecewise constant scheme.')
# parser.add_argument('--boundaries', type=str, default='.3125,.375,.4375,.5625,.6875,.74', help='Boundary percentage for changing the learning rate.')

parser.add_argument('--lr', type=str, default='1e-3,.5e-3,.25e-3,.125e-3,.0625e-3,.03125e-3,.015625e-3', help='Learning rate under piecewise constant scheme.')
parser.add_argument('--boundaries', type=str, default='.3125,.375,.4375,.5625,.6875,.74', help='Boundary percentage for changing the learning rate.')


parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay factor.')
parser.add_argument('--num_samples', type=int, default=160000, help='Total number =total_step*batch_size of samples for training.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')

parser.add_argument('--load_path', type=str, default=None, help='The dir of the folder containing the pretrained checkpoints.')
parser.add_argument('--load_step', type=int, default=-1, help='The step to load. -1 for the latest one.')
parser.add_argument('--reset_step', action='store_true', help='Set to reset the global step. Otherwise resume from the step of the checkpoint.')

parser.add_argument('--job_name', type=str, default='temp', help='Job name for the name of the saved checkpoint.')

parser.add_argument('--save_dir', type=str, help='The dir for saving the checkpoints.')

parser.add_argument('--snapshot', type=int, default=100, help='Step interval to save a checkpoint.')

parser.add_argument('--depth_save', type=int, default=1, help='Step interval to save a depth.')

parser.add_argument('--max_keep', type=int, default=1000, help='Max number of checkpoints kept.')

args = parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    seed = 0
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_steps = args.num_samples // args.batch_size
    [resize_width, resize_height], [crop_width, crop_height] = [[int(v) for v in arg_str.split(',')] for arg_str in [args.resize, args.crop]]
    cas_depth_num = [int(v) for v in args.cas_depth_num.split(',')]
    cas_interv_scale = [float(v) for v in args.cas_interv_scale.split(',')]

    Model = importlib.import_module(f'core.{args.model_name}').Model
    Loss = importlib.import_module(f'core.{args.model_name}').Loss
    get_train_loader = importlib.import_module(f'data.{args.dataset_name}').get_train_loader

    dataset, loader = get_train_loader(
        args.data_root, args.num_src, total_steps, args.batch_size,
        {
            'interval_scale': args.interval_scale,
            'max_d': args.max_d,
            'resize_width': resize_width,
            'resize_height': resize_height,
            'crop_width': crop_width,
            'crop_height': crop_height
        },
        num_workers=args.num_workers
    )

    model = Model()
    model.cuda()
    model = apex_parallel.convert_syncbn_model(model)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters() if p.requires_grad])))
    compute_loss = Loss()

    model = nn.DataParallel(model)
    
    
    # lr = [0.001, 0.0005, 0.00025, 0.000125,  0.0000625, 0.00003125, 0.000015625]
    # boundaries = [100000, 120000, 140000, 180000, 220000, 260000]
    
    lr = [float(v) for v in args.lr.split(',')]
    boundaries = args.boundaries
    if boundaries is not None:
        boundaries = [int(total_steps * float(b)) for b in boundaries.split(',')]
    
    optimizer = optim.Adam(model.parameters(), lr=lr[0], weight_decay=args.weight_decay)
    
    
    if args.load_path is None:
        for m in model.modules():
            if any([isinstance(m, T) for T in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight)
            elif any([isinstance(m, T) for T in [nn.BatchNorm2d, nn.BatchNorm3d]]):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        global_step = 0
    else:
        global_step = load_model(model, optimizer, args.load_path, args.load_step, val=False)
        if args.reset_step: global_step = 0
        print(f'load {os.path.join(args.load_path, str(args.load_step))}')

    # lr = [float(v) for v in args.lr.split(',')]
    # boundaries = args.boundaries
    # if boundaries is not None:
    #     boundaries = [int(total_steps * float(b)) for b in boundaries.split(',')]
    

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

    def piecewise_constant():
        if boundaries is None: return lr[0]
        i = 0
        for b in boundaries:
            if global_step < b: break
            i += 1
        curr_lr = lr[i]
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        return curr_lr

    model.train()

    pbar = tqdm.tqdm(loader, dynamic_ncols=True)
    if global_step != 0: pbar.update(global_step)
    for sample in pbar:
        if global_step >= total_steps: break
        if sample.get('skip') is not None and np.any(sample['skip']): continue
        curr_lr = piecewise_constant()
        recursive_apply(sample, lambda x: torch.from_numpy(x).float().cuda())
        ref, ref_cam, srcs, srcs_cam, gt, masks = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]

        loss, uncert_loss, less1, less3, l1, losses, outputs, refined_depth, prob_maps = None, None, None, None, None, None, None, None, None
        try:
            # est_depth, prob_map, pair_results = model([ref, ref_cam, srcs, srcs_cam], args.max_d, mode=args.mode)
            outputs, refined_depth, prob_maps = model(sample, cas_depth_num, cas_interv_scale, mode=args.mode)

            # losses = compute_loss([est_depth, pair_results], gt, masks, ref_cam, args.max_d, occ_guide=args.occ_guide, mode=args.mode)
            # print('gt: ',type(gt))
            # print('masks: ',type(masks))
            # print('ref_cam: ',type(ref_cam))
            # print('outputs: ',type(outputs))
            # print('refined_depth: ',type(refined_depth))
            losses = compute_loss([outputs, refined_depth], gt, masks, ref_cam, args.max_d, occ_guide=args.occ_guide, mode=args.mode)
            
            
            loss, uncert_loss, less1, less3, l1 = losses[:5]  #MVS
            # print("loss", loss)
            # print("loss",loss)
            # print("uncert_loss",uncert_loss)
            # print("less1",less1)
            # print("less3",less3)
            # print("l1",l1)
            # loss, less1, less3, l1 = losses[:4]

            if np.isnan(loss.item()):
                raise NanError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_np = [v.item() for v in losses[:5]]  #MVS
            loss, uncert_loss, less1, less3, l1 = losses_np  #MVS
            # loss, less1, less3, l1 = losses_np

            stats = losses[5]
            stats_np = [(l1.item(), less1.item(), less3.item()) for l1, less1, less3 in stats]
            stats_str = ''.join([f'({l1:.3f} {less1*100:.2f} {less3*100:.2f})' for l1, less1, less3 in stats_np])

            pbar.set_description(f'{loss:.3f}{stats_str}{l1:.3f}')
            # pbar.set_description(f'{loss:.4f} {less1:.3f} {less3:.3f} {l1:.4f}')  #MVS
            # pbar.set_description(f'{less1:.3f} {less3:.3f} {l1:.4f}')
            
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            if len(parameters) == 0:
                total_norm = 0.0
            else:
                device = parameters[0].grad.device
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]), 2.0).item()
            
            #Add logging depths
            
            if global_step != 0 and global_step % args.depth_save == 0:
                [[est_depth_1, pair_results_1], [est_depth_2, pair_results_2], [est_depth_3, pair_results]] = outputs
                d = np.flipud(est_depth_3.cpu().detach().numpy())
                dmin = gt[gt != 0].cpu().detach().numpy().min()
                dmax = gt[gt != 0].cpu().detach().numpy().max()
                d = (d - dmin) / (dmax - dmin)
                d = cm.plasma_r(d)[..., :3]
                d = np.clip(d * 255, 0, 255).astype(np.uint8)
                d_image = Image.fromarray(d[0,0])
                wandb.log({b
                       "samples": wandb.Image(d_image, caption=f'{global_step} iteration')
                      })
                del est_depth_1, pair_results_1, est_depth_2, pair_results_2, est_depth_3, pair_results, d, dmin, dmax, d_image
                
            wandb.log({"loss": loss,
                       "loss":loss,
                       "uncert_loss":uncert_loss,
                       "less1":less1,
                       "less3":less3,
                       "l1":l1,
                       "curr_lr":curr_lr,
                      "total_norm":total_norm
                      })
            
        except NanError:
            print(f'nan: {global_step}/{total_steps}')
            gc.collect()
            torch.cuda.empty_cache()
            # optimizer.zero_grad()
            # optimizer.step()

        if global_step != 0 and global_step % args.snapshot == 0:
            save_model({
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.save_dir, args.job_name, global_step, args.max_keep)

        global_step += 1
        del loss, uncert_loss, less1, less3, l1, losses, outputs, refined_depth, prob_maps

    save_model({
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, args.save_dir, args.job_name, global_step, args.max_keep)
