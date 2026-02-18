import argparse
from pathlib import Path
import numpy as np
import cv2
import json
import csv

def read_img_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def read_kitti2015_disp_png(p: Path) -> np.ndarray:
    """
    KITTI Stereo 2015 disparity GT is 16-bit PNG.
    Stored as disparity * 256. Invalid pixels are 0.
    """
    disp_u16 = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if disp_u16 is None:
        raise FileNotFoundError(p)
    disp = disp_u16.astype(np.float32) / 256.0
    disp[disp_u16 == 0] = np.nan
    return disp

def box_sum(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)

def disparity_sad(left: np.ndarray, right: np.ndarray, max_disp: int, window: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("window must be odd")
    H, W = left.shape
    L = left.astype(np.float32)
    R = right.astype(np.float32)

    costs = np.full((H, W, max_disp + 1), np.inf, dtype=np.float32)
    for d in range(max_disp + 1):
        R_shift = np.zeros_like(R)
        if d == 0:
            R_shift[:] = R
        else:
            R_shift[:, d:] = R[:, :-d]
        diff = np.abs(L - R_shift).astype(np.float32)
        sad = box_sum(diff, window)
        sad[:, :d] = np.inf
        costs[:, :, d] = sad

    disp = np.argmin(costs, axis=2).astype(np.float32)

    pad = window // 2
    disp[:pad, :] = 0
    disp[-pad:, :] = 0
    disp[:, :pad] = 0
    disp[:, -pad:] = 0
    return disp

def disparity_ncc(left: np.ndarray, right: np.ndarray, max_disp: int, window: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("window must be odd")
    H, W = left.shape
    L = left.astype(np.float32)
    R = right.astype(np.float32)

    sumL = box_sum(L, window)
    sumL2 = box_sum(L * L, window)

    scores = np.full((H, W, max_disp + 1), -np.inf, dtype=np.float32)
    eps = 1e-6
    N = float(window * window)

    for d in range(max_disp + 1):
        R_shift = np.zeros_like(R)
        if d == 0:
            R_shift[:] = R
        else:
            R_shift[:, d:] = R[:, :-d]

        sumR = box_sum(R_shift, window)
        sumR2 = box_sum(R_shift * R_shift, window)
        sumLR = box_sum(L * R_shift, window)

        meanL = sumL / N
        meanR = sumR / N

        num = sumLR - N * meanL * meanR
        denL = sumL2 - N * meanL * meanL
        denR = sumR2 - N * meanR * meanR
        den = np.sqrt(np.maximum(denL, 0) * np.maximum(denR, 0)) + eps

        ncc = num / den
        ncc[:, :d] = -np.inf
        scores[:, :, d] = ncc

    disp = np.argmax(scores, axis=2).astype(np.float32)

    pad = window // 2
    disp[:pad, :] = 0
    disp[-pad:, :] = 0
    disp[:, :pad] = 0
    disp[:, -pad:] = 0
    return disp

def save_vis_disparity(disp: np.ndarray, out_png: Path, max_disp: int):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    d = disp.copy()
    d = np.clip(d, 0, max_disp)
    vis = (d / float(max_disp) * 255.0).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_png), vis_color)

def save_vis_error(err: np.ndarray, out_png: Path, clip: float = 10.0):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    e = np.clip(err, 0, clip)
    vis = (e / clip * 255.0).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_png), vis_color)

def eval_metrics(disp_est: np.ndarray, disp_gt: np.ndarray, bad_thresh: float):
    """
    Compute over valid GT pixels (GT != NaN).
    Returns: mae, bad_rate(%), valid_count
    """
    m = np.isfinite(disp_gt)
    valid = int(m.sum())
    if valid == 0:
        return None, None, 0

    err = np.abs(disp_est[m] - disp_gt[m])
    mae = float(np.mean(err))
    bad = float(np.mean(err > bad_thresh) * 100.0)
    return mae, bad, valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Project root (E:\\CV_project3)")
    ap.add_argument("--kitti2015", default="data/kitti_stereo_2015/training", help="relative to root")
    ap.add_argument("--n", type=int, default=10, help="how many image pairs")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--bad-thresh", type=float, default=3.0)

    # Ablation settings required by project: 2 costs + 2 windows
    ap.add_argument("--costs", nargs="+", default=["SAD", "NCC"])
    ap.add_argument("--windows", nargs="+", type=int, default=[7, 11])

    ap.add_argument("--outdir", default="outputs/partA_5_eval_kitti2015")
    args = ap.parse_args()

    root = Path(args.root)
    base = root / args.kitti2015

    imgL_dir = base / "image_2"
    imgR_dir = base / "image_3"
    gt_dir   = base / "disp_occ_0"

    if not imgL_dir.exists() or not imgR_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError("Expected image_2, image_3, disp_occ_0 under training folder.")

    L_files = sorted(imgL_dir.glob("*.png"))
    R_files = sorted(imgR_dir.glob("*.png"))
    GT_files = sorted(gt_dir.glob("*.png"))

    end = min(args.start + args.n, len(L_files), len(R_files), len(GT_files))
    if end - args.start <= 0:
        raise RuntimeError("No images found or start/n out of range.")

    out_base = root / args.outdir
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Part A_5: Quantitative disparity eval on KITTI Stereo 2015 TRAINING")
    print(f"[INFO] Frames {args.start}..{end-1} (total {end-args.start})")
    print(f"[INFO] Metrics: MAE + Bad-pixel-rate (>{args.bad_thresh}px)")
    print(f"[INFO] Ablation: costs={args.costs}, windows={args.windows}, max_disp={args.max_disp}")
    print(f"[INFO] Output: {out_base}")

    summary = []
    # CSV for easy report table
    csv_path = out_base / "eval_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["cost", "window", "max_disp", "n_images", "mean_MAE_px", "mean_BadRate_%", "mean_validGT_pixels"])

        for cost in args.costs:
            for win in args.windows:
                maes=[]
                bads=[]
                valids=[]
                setting_tag = f"{cost}_w{win}_d{args.max_disp}"
                setting_dir = out_base / setting_tag
                setting_dir.mkdir(parents=True, exist_ok=True)

                disp_fn = disparity_sad if cost.upper()=="SAD" else disparity_ncc

                for i in range(args.start, end):
                    L = read_img_gray(L_files[i])
                    R = read_img_gray(R_files[i])
                    gt = read_kitti2015_disp_png(GT_files[i])

                    disp = disp_fn(L, R, max_disp=args.max_disp, window=win)

                    mae, bad, valid = eval_metrics(disp, gt, bad_thresh=args.bad_thresh)
                    if mae is None:
                        continue

                    maes.append(mae); bads.append(bad); valids.append(valid)

                    # Save a few qualitative outputs (first 3 frames only to keep it light)
                    if (i - args.start) < 3:
                        stem = L_files[i].stem
                        save_vis_disparity(disp, setting_dir / f"{stem}_disp_vis.png", max_disp=args.max_disp)

                        # error map vis
                        m = np.isfinite(gt)
                        err = np.zeros_like(disp, dtype=np.float32)
                        err[m] = np.abs(disp[m] - gt[m])
                        save_vis_error(err, setting_dir / f"{stem}_err_vis.png", clip=10.0)

                mean_mae = float(np.mean(maes)) if len(maes)>0 else None
                mean_bad = float(np.mean(bads)) if len(bads)>0 else None
                mean_valid = float(np.mean(valids)) if len(valids)>0 else None

                print(f"[OK] {setting_tag}: MAE={mean_mae:.4f}px, Bad>{args.bad_thresh}px={mean_bad:.2f}%, validGT={mean_valid:.0f}")

                summary.append({
                    "cost": cost, "window": win, "max_disp": args.max_disp,
                    "n_images_used": len(maes),
                    "mean_MAE_px": mean_mae,
                    "mean_BadRate_pct": mean_bad,
                    "mean_validGT_pixels": mean_valid,
                    "outputs_dir": str(setting_dir)
                })

                w.writerow([cost, win, args.max_disp, len(maes), mean_mae, mean_bad, mean_valid])

    (out_base / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Part A_5 finished. Saved:")
    print(f"  - {csv_path}")
    print(f"  - {out_base/'eval_summary.json'}")

if __name__ == "__main__":
    main()
