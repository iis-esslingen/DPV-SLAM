import cv2
import numpy as np
import glob
import os
import sys
import time
from pathlib import Path

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg

import torch
from multiprocessing import Process, Queue
from itertools import chain

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
from pathlib import Path
import json


SKIP = 0

def show_image(image, t=0):
    image = image.cpu().numpy()
    cv2.imshow('image', np.array(image) / 255.0)
    cv2.waitKey(t)

def image_stream(queue, imagedir, camera, stride, skip=0):
    """ image generator """

    if camera == "t265":
        imagedir = os.path.join(imagedir, "cam_left")
    else:
        imagedir = os.path.join(imagedir, "rgb")

    calib = os.path.join("calib", f"{camera}.txt")

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        image = image.transpose(2,0,1)

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))

@torch.no_grad()
def run(cfg, network, datapath, camera, stride=1, viz=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, datapath, camera, stride, 0))
    reader.start()

    for step in range(sys.maxsize):
        (t, images, intrinsics) = queue.get()
        if t < 0: break

        images = torch.as_tensor(images, device='cuda')
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        if viz:
            show_image(images[0], 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=images.shape[-2], wd=images.shape[-1], viz=viz)

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, images, intrinsics)

    for _ in range(12):
        slam.update()

    reader.join()

    return slam.terminate()

def evaluate(traj_est, datapath, outputpath, ground_truth_path, camera, stride):

    if camera == "t265":
        image_dir = os.path.join(datapath, "cam_left")
    else:
        image_dir = os.path.join(datapath, "rgb")

    images_list = sorted(glob.glob(os.path.join(image_dir, '*.png')))[::stride]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est_fused = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    Path(outputpath).mkdir(parents=True, exist_ok=True)
    result_file_name = f"{camera}_slam_trajectory.txt"
    file_interface.write_tum_trajectory_file(os.path.join(outputpath, f"{camera}_slam_trajectory.txt"), traj_est_fused)
    
    traj_ref = file_interface.read_tum_trajectory_file(ground_truth_path)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est_fused)

    result_ape = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    with open(os.path.join(outputpath, result_file_name.replace("slam_trajectory", "ape_results")), "w") as file:
        file.write(json.dumps(result_ape.stats))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=Path)
    parser.add_argument("--ground_truth_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--cameras", type=str, choices=["d435i", "t265", "pi_cam"])
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    # parser.add_argument('--viz', action="store_true")
    # parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")
    
    for trial in range(args.trials):
        print(f"Trial: {trial}")
        main_seed = int(time.time())
        torch.manual_seed(main_seed)
        
        datapath = os.path.join(args.base_data_path, args.camera)

        traj_est, timestamps = run(cfg, args.network, datapath, args.camera, args.stride)

        outputpath = os.path.join(args.output_path, args.camera, str(trial))
        evaluate(traj_est, datapath, outputpath, args.ground_truth_path, args.camera, args.stride)
