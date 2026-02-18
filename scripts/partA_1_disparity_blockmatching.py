import os
import argparse
from pathlib import Path
import numpy as np
import cv2

def read_kitti_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def box_sum(img: np.ndarray, ksize: int) -> np.ndarray:
    # Sum over kxk window using boxFilter (fast)
    return cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize), normalize=False, borderType=cv2.BORDER_CONSTANT)

def disparity_sad(left: np.ndarray, right: np.ndarray, max_disp: int, window: int) -> np.ndarray:
    """
    Classic block matching with SAD:
    For each disparity d, compute |L - R_shifted(d)| and sum over window -> cost volume.
    Choose argmin cost per pixel.
    """
    if window % 2 == 0:
        raise ValueError("window size must be odd")

    H, W = left.shape
    L = left.astype(np.float32)
    R = right.astype(np.float32)

    # Cost volume: H x W x D
    costs = np.full((H, W, max_disp + 1), np.inf, dtype=np.float32)

    for d in range(0, max_disp + 1):
        # shift right image to the right by d (so pixel x in left compares with x-d in right)
        R_shift = np.zeros_like(R)
        if d == 0:
            R_shift[:] = R
        else:
            R_shift[:, d:] = R[:, :-d]

        diff = np.abs(L - R_shift).astype(np.float32)
        sad = box_sum(diff, window)

        # invalid region: first d columns are invalid for disparity d
        sad[:, :d] = np.inf
        costs[:, :, d] = sad

    disp = np.argmin(costs, axis=2).astype(np.float32)

    # Remove borders where window is incomplete (optional, but makes it cleaner/defensible)
    pad = window // 2
    disp[:pad, :] = 0
    disp[-pad:, :] = 0
    disp[:, :pad] = 0
    disp[:, -pad:] = 0
    return disp

def disparity_ncc(left: np.ndarray, right: np.ndarray, max_disp: int, window: int) -> np.ndarray:
    """
    Normalized Cross-Correlation (NCC) block matching:
    NCC = (sum((L-meanL)(R-meanR))) / sqrt(sum((L-meanL)^2) * sum((R-meanR)^2))
    Choose argmax NCC per pixel.
    """
    if window % 2 == 0:
        raise ValueError("window size must be odd")

    H, W = left.shape
    L = left.astype(np.float32)
    R = right.astype(np.float32)

    # Precompute sums for L (independent of disparity)
    sumL = box_sum(L, window)
    sumL2 = box_sum(L * L, window)

    # NCC volume: higher is better
    scores = np.full((H, W, max_disp + 1), -np.inf, dtype=np.float32)
    eps = 1e-6

    for d in range(0, max_disp + 1):
        R_shift = np.zeros_like(R)
        if d == 0:
            R_shift[:] = R
        else:
            R_shift[:, d:] = R[:, :-d]

        sumR  = box_sum(R_shift, window)
        sumR2 = box_sum(R_shift * R_shift, window)
        sumLR = box_sum(L * R_shift, window)

        # Compute NCC numerator/denominator per pixel within window
        N = float(window * window)
        meanL = sumL / N
        meanR = sumR / N

        num = sumLR - N * meanL * meanR
        denL = sumL2 - N * meanL * meanL
        denR = sumR2 - N * meanR * meanR
        den = np.sqrt(np.maximum(denL, 0) * np.maximum(denR, 0)) + eps

        ncc = num / den

        # invalid region for disparity d
        ncc[:, :d] = -np.inf
        scores[:, :, d] = ncc

    disp = np.argmax(scores, axis=2).astype(np.float32)

    pad = window // 2
    disp[:pad, :] = 0
    disp[-pad:, :] = 0
    disp[:, :pad] = 0
    disp[:, -pad:] = 0
    return disp

def save_disparity_outputs(disp: np.ndarray, out_prefix: Path):
    """
    Saves:
    - raw disparity as .npy
    - normalized 8-bit visualization as .png (NO DISPLAY)
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_disp.npy", disp)

    # Visualization: scale to 0..255 for saving
    dmax = np.max(disp)
    if dmax <= 0:
        vis = np.zeros_like(disp, dtype=np.uint8)
    else:
        vis = (disp / dmax * 255.0).clip(0, 255).astype(np.uint8)

    # Optional colormap (still just file output)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_prefix) + "_disp_vis.png", vis_color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Project root path (e.g., E:\\CV_project3)")
    parser.add_argument("--seq", type=str, default="00", help="KITTI odometry sequence (00..10)")
    parser.add_argument("--n", type=int, default=10, help="How many frames to process")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--max-disp", type=int, default=96, help="Maximum disparity (search range)")
    parser.add_argument("--window", type=int, default=9, help="Odd window size (e.g., 7, 9, 11)")
    parser.add_argument("--cost", type=str, choices=["SAD", "NCC"], default="SAD", help="Matching cost")
    parser.add_argument("--outdir", type=str, default="outputs/partA_1", help="Output directory (relative to root)")
    args = parser.parse_args()

    root = Path(args.root)
    left_dir  = root / "data" / "kitti_odometry" / args.seq / "image_0"
    right_dir = root / "data" / "kitti_odometry" / args.seq / "image_1"

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Could not find image_0/image_1 in: {root / 'data' / 'kitti_odometry' / args.seq}")

    outdir = root / args.outdir / f"seq_{args.seq}" / f"{args.cost}_w{args.window}_d{args.max_disp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect frame file names (KITTI uses 000000.png ...)
    left_files = sorted([p for p in left_dir.glob("*.png")])
    right_files = sorted([p for p in right_dir.glob("*.png")])

    if len(left_files) == 0 or len(right_files) == 0:
        raise RuntimeError("No PNG images found in image_0/image_1")

    end = min(args.start + args.n, len(left_files), len(right_files))
    print(f"[INFO] Processing seq={args.seq} frames {args.start}..{end-1} (total {end-args.start})")
    print(f"[INFO] Cost={args.cost}, window={args.window}, max_disp={args.max_disp}")
    print(f"[INFO] Output dir: {outdir}")

    for i in range(args.start, end):
        L = read_kitti_image(left_files[i])
        R = read_kitti_image(right_files[i])

        if args.cost == "SAD":
            disp = disparity_sad(L, R, max_disp=args.max_disp, window=args.window)
        else:
            disp = disparity_ncc(L, R, max_disp=args.max_disp, window=args.window)

        frame_id = left_files[i].stem  # e.g., 000123
        save_disparity_outputs(disp, outdir / frame_id)
        print(f"[OK] Frame {frame_id}: saved")

    print("[DONE] Part A_1 finished (raw disparity for 10 frames).")

if __name__ == "__main__":
    main()
