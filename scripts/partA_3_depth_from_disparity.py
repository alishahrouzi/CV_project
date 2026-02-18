import argparse
from pathlib import Path
import numpy as np
import cv2

def parse_kitti_calib(calib_path: Path):
    """
    Reads KITTI odometry calib.txt.
    Expects lines like:
      P0: <12 numbers>
      P1: <12 numbers>
    Returns P0, P1 as 3x4 float arrays.
    """
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calib file: {calib_path}")

    P0 = None
    P1 = None
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("P0:"):
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) != 12:
                    raise ValueError("P0 must have 12 values")
                P0 = np.array(vals, dtype=np.float64).reshape(3,4)
            if line.startswith("P1:"):
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) != 12:
                    raise ValueError("P1 must have 12 values")
                P1 = np.array(vals, dtype=np.float64).reshape(3,4)

    if P0 is None or P1 is None:
        raise ValueError("Could not find P0 and P1 in calib.txt")

    return P0, P1

def compute_f_and_baseline(P0: np.ndarray, P1: np.ndarray):
    """
    KITTI convention:
      f = P0[0,0]
      tx = P[0,3]  (equals f * (-baseline) if rectified)
    baseline B = |tx1 - tx0| / f
    """
    f = float(P0[0,0])
    tx0 = float(P0[0,3])
    tx1 = float(P1[0,3])
    B = abs(tx1 - tx0) / f
    return f, B

def disparity_to_depth(disp: np.ndarray, f: float, B: float, min_disp: float = 0.1):
    """
    Z = f*B / d
    Avoid division by zero by clamping very small disparities.
    """
    d = disp.astype(np.float32)
    d_safe = np.where(d > min_disp, d, np.nan)  # invalid/too-small -> NaN
    Z = (f * B) / d_safe
    return Z.astype(np.float32)

def save_depth(depth: np.ndarray, out_prefix: Path, max_depth_m: float = 80.0):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_depth.npy", depth)

    # Visualization (NaN -> 0)
    d = depth.copy()
    d[np.isnan(d)] = 0.0
    d = np.clip(d, 0, max_depth_m)
    vis = (d / max_depth_m * 255.0).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_prefix) + "_depth_vis.png", vis_color)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seq", default="00")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--start", type=int, default=0)

    # Must match your Part A_2 fix output settings:
    p.add_argument("--disp_dir", default="outputs/partA_2_fix")
    p.add_argument("--disp_tag", default="SAD_w9_d96_lr2.0_med5", help="folder tag inside disp_dir/seq_xx/")
    p.add_argument("--outdir", default="outputs/partA_3_depth")

    p.add_argument("--max_depth", type=float, default=80.0)
    args = p.parse_args()

    root = Path(args.root)
    seq_dir = root / "data" / "kitti_odometry" / args.seq

    calib_path = seq_dir / "calib.txt"
    P0, P1 = parse_kitti_calib(calib_path)
    f, B = compute_f_and_baseline(P0, P1)

    print(f"[INFO] Using calib: {calib_path}")
    print(f"[INFO] f={f:.3f}, baseline B={B:.6f} meters")

    disp_base = root / args.disp_dir / f"seq_{args.seq}" / args.disp_tag
    if not disp_base.exists():
        raise FileNotFoundError(f"Disparity folder not found: {disp_base}")

    # Determine frames from available disp files
    disp_files = sorted(disp_base.glob("*_disp_pp.npy"))
    if len(disp_files) == 0:
        raise RuntimeError("No *_disp_pp.npy found. Did you run Part A_2 fix?")

    end = min(args.start + args.n, len(disp_files))
    out_base = root / args.outdir / f"seq_{args.seq}" / args.disp_tag
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing depth for frames {args.start}..{end-1} (total {end-args.start})")
    print(f"[INFO] Output: {out_base}")

    for i in range(args.start, end):
        disp_path = disp_files[i]
        frame_id = disp_path.name.replace("_disp_pp.npy", "")
        disp = np.load(disp_path)

        depth = disparity_to_depth(disp, f=f, B=B, min_disp=0.1)
        save_depth(depth, out_base / frame_id, max_depth_m=args.max_depth)

        valid_pct = float(np.isfinite(depth).mean()) * 100.0
        print(f"[OK] Frame {frame_id}: depth saved (valid={valid_pct:.2f}%)")

    print("[DONE] Part A_3 finished (depth maps saved).")

if __name__ == "__main__":
    main()
