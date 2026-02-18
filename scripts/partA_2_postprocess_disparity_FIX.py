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
    return cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize), normalize=False, borderType=cv2.BORDER_CONSTANT)

def disparity_sad_leftref(L: np.ndarray, R: np.ndarray, max_disp: int, window: int) -> np.ndarray:
    # L(x) vs R(x-d)
    H, W = L.shape
    Lf = L.astype(np.float32)
    Rf = R.astype(np.float32)
    costs = np.full((H, W, max_disp + 1), np.inf, dtype=np.float32)

    for d in range(max_disp + 1):
        R_shift = np.zeros_like(Rf)
        if d == 0:
            R_shift[:] = Rf
        else:
            R_shift[:, d:] = Rf[:, :-d]  # R(x-d) aligned to x

        diff = np.abs(Lf - R_shift)
        sad = box_sum(diff, window)
        sad[:, :d] = np.inf
        costs[:, :, d] = sad

    disp = np.argmin(costs, axis=2).astype(np.float32)
    pad = window // 2
    disp[:pad, :] = 0; disp[-pad:, :] = 0; disp[:, :pad] = 0; disp[:, -pad:] = 0
    return disp

def disparity_sad_rightref(R: np.ndarray, L: np.ndarray, max_disp: int, window: int) -> np.ndarray:
    # R(x) vs L(x+d)  -> shift L to the LEFT by d so that L(x+d) aligns to x
    H, W = R.shape
    Rf = R.astype(np.float32)
    Lf = L.astype(np.float32)
    costs = np.full((H, W, max_disp + 1), np.inf, dtype=np.float32)

    for d in range(max_disp + 1):
        L_shift = np.zeros_like(Lf)
        if d == 0:
            L_shift[:] = Lf
        else:
            L_shift[:, :-d] = Lf[:, d:]  # L(x+d) aligned to x

        diff = np.abs(Rf - L_shift)
        sad = box_sum(diff, window)
        sad[:, W-d:] = np.inf  # last d columns invalid for right-ref
        costs[:, :, d] = sad

    disp = np.argmin(costs, axis=2).astype(np.float32)
    pad = window // 2
    disp[:pad, :] = 0; disp[-pad:, :] = 0; disp[:, :pad] = 0; disp[:, -pad:] = 0
    return disp

def lr_consistency_mask(dispL: np.ndarray, dispR: np.ndarray, thresh: float) -> np.ndarray:
    # xR = xL - dL ; compare dR at xR
    H, W = dispL.shape
    xs = np.arange(W, dtype=np.int32)[None, :].repeat(H, axis=0)
    xR = (xs - dispL).round().astype(np.int32)

    invalid = np.zeros((H, W), dtype=np.uint8)
    oob = (xR < 0) | (xR >= W)
    invalid[oob] = 1

    yy, xx = np.where(~oob)
    xr = xR[yy, xx]
    diff = np.abs(dispL[yy, xx] - dispR[yy, xr])
    invalid[yy[diff > thresh], xx[diff > thresh]] = 1
    return invalid

def median_filter_disparity(disp: np.ndarray, k: int) -> np.ndarray:
    return cv2.medianBlur(disp.astype(np.float32), k)

def hole_fill_inpaint(disp: np.ndarray, invalid_mask: np.ndarray, max_disp: int, inpaint_radius: int = 3) -> np.ndarray:
    d = np.clip(disp, 0, max_disp)
    disp8 = (d / float(max_disp) * 255.0).astype(np.uint8)
    mask8 = (invalid_mask * 255).astype(np.uint8)
    filled8 = cv2.inpaint(disp8, mask8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return (filled8.astype(np.float32) / 255.0) * float(max_disp)

def save_outputs(disp_raw: np.ndarray, disp_pp: np.ndarray, invalid: np.ndarray, out_prefix: Path, max_disp: int):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_disp_raw.npy", disp_raw)
    np.save(str(out_prefix) + "_disp_pp.npy", disp_pp)
    cv2.imwrite(str(out_prefix) + "_invalid_mask.png", (invalid * 255).astype(np.uint8))

    def save_vis(disp: np.ndarray, suffix: str):
        d = np.clip(disp, 0, max_disp)
        vis = (d / float(max_disp) * 255.0).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_prefix) + f"_{suffix}_vis.png", vis_color)

    save_vis(disp_raw, "raw")
    save_vis(disp_pp, "pp")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seq", default="00")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-disp", type=int, default=96)
    p.add_argument("--window", type=int, default=9)
    p.add_argument("--lr-thresh", type=float, default=2.0)
    p.add_argument("--median-k", type=int, default=5)
    p.add_argument("--outdir", default="outputs/partA_2_fix")
    args = p.parse_args()

    root = Path(args.root)
    left_dir  = root / "data" / "kitti_odometry" / args.seq / "image_0"
    right_dir = root / "data" / "kitti_odometry" / args.seq / "image_1"

    left_files = sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.png"))
    end = min(args.start + args.n, len(left_files), len(right_files))

    outdir = root / args.outdir / f"seq_{args.seq}" / f"SAD_w{args.window}_d{args.max_disp}_lr{args.lr_thresh}_med{args.median_k}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Part A_2 (FIX) seq={args.seq} frames {args.start}..{end-1}")
    print(f"[INFO] Steps: LR-consistency, median filter, hole filling")
    print(f"[INFO] Output: {outdir}")

    for i in range(args.start, end):
        L = read_kitti_image(left_files[i])
        R = read_kitti_image(right_files[i])

        dispL_raw = disparity_sad_leftref(L, R, args.max_disp, args.window)
        dispR_raw = disparity_sad_rightref(R, L, args.max_disp, args.window)

        invalid = lr_consistency_mask(dispL_raw, dispR_raw, args.lr_thresh)

        disp_med = median_filter_disparity(dispL_raw, args.median_k)
        disp_med[invalid == 1] = 0

        disp_pp = hole_fill_inpaint(disp_med, invalid, args.max_disp, inpaint_radius=3)

        frame_id = left_files[i].stem
        inv_pct = float(invalid.mean()) * 100.0
        save_outputs(dispL_raw, disp_pp, invalid, outdir / frame_id, args.max_disp)
        print(f"[OK] Frame {frame_id}: invalid={inv_pct:.2f}%")

    print("[DONE] Part A_2 fix finished.")

if __name__ == "__main__":
    main()
