# Stereo Depth & Stereo Visual Odometry - Final Project

## Dependencies

Install requirements:

pip install -r requirements.txt

---

# Part A – Stereo Depth

## A1: Compute disparity (SAD / NCC)

python scripts/partA_1_disparity_blockmatching.py --root "E:\CV_project3" --seq 00 --n 10 --cost SAD --window 9 --max-disp 96

## A5: Quantitative evaluation on KITTI Stereo 2015

python scripts/partA_5_eval_kitti2015_disparity.py --root "E:\CV_project3" --kitti2015 "data/kitti_stereo_2015/training" --n 10 --max-disp 192 --bad-thresh 3.0 --costs SAD NCC --windows 7 11

Outputs:
- eval_table.csv
- eval_summary.json
- qualitative disparity & error maps

Metrics:
- MAE (px)
- Bad-pixel rate (>3px)

---

# Part B – Stereo Visual Odometry

## B1: SIFT feature matching

python scripts/partB_1_sift_matching.py --root "E:\CV_project3" --seq 00 --n 10

## B2: Essential matrix + RANSAC

python scripts/partB_2_essential_pose.py --root "E:\CV_project3" --seq 00

## B3: Metric scale using stereo depth (PnP + RANSAC)

python scripts/partB_3_pnp_scale.py --root "E:\CV_project3" --seq 00 --depthtag "SAD_w9_d96_lr2.0_med5"

## B4: Trajectory evaluation (ATE + RPE)

python scripts/partB_4_traj_eval_FIX.py --root "E:\CV_project3" --seq 00 --depthtag "SAD_w9_d96_lr2.0_med5"

Metrics:
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error)

## B5: VO Ablation

python scripts/partB_5_vo_ablation.py --root "E:\CV_project3" --seq 00 --depthtag "SAD_w9_d96_lr2.0_med5"

Comparison:
- Essential w/ RANSAC
- Essential without RANSAC
- PnP metric scale

---

# Calibration

Camera intrinsics (K), focal length (f), and baseline (B) are extracted from KITTI calib.txt.

Depth is computed as:

Z = f * B / disparity

---

# Notes

- Depth evaluation performed on KITTI Stereo 2015 training set.
- VO evaluation performed on KITTI Odometry sequences.
- All experiments use 10 frames for runtime efficiency.

