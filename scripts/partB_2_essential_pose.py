import argparse
from pathlib import Path
import numpy as np
import cv2
import json

def parse_kitti_calib(calib_path: Path):
    P0 = None
    for line in calib_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("P0:"):
            vals = list(map(float, line.split()[1:]))
            P0 = np.array(vals).reshape(3, 4)
            break
    if P0 is None:
        raise ValueError("P0 not found in calib.txt")
    K = P0[:, :3].astype(np.float64)
    return K

def read_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def draw_matches_simple(img1_gray, img2_gray, pts1, pts2, inlier_mask=None, max_draw=1500):
    """
    Draw lines from pts1 (on img1) to pts2 (on img2) on a side-by-side canvas.
    inlier_mask: (N,) bool/0-1 -> inliers green, outliers red (if provided)
    """
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    H = max(h1, h2)
    W = w1 + w2

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)

    n = min(len(pts1), len(pts2))
    if n == 0:
        return canvas

    # downsample if too many
    idx = np.arange(n)
    if n > max_draw:
        step = max(1, n // max_draw)
        idx = idx[::step]

    for k in idx:
        x1, y1 = pts1[k]
        x2, y2 = pts2[k]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2 + w1)), int(round(y2)))

        if inlier_mask is None:
            color = (0, 255, 0)
        else:
            color = (0, 255, 0) if int(inlier_mask[k]) == 1 else (0, 0, 255)

        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, color, -1, cv2.LINE_AA)

    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--matchdir", default="outputs/partB_1_matching_sift", help="where pts1/pts2 are saved")
    ap.add_argument("--outdir", default="outputs/partB_2_essential_pose")
    ap.add_argument("--ransac_thresh", type=float, default=1.0, help="RANSAC reprojection threshold (pixels)")
    ap.add_argument("--ransac_prob", type=float, default=0.999)

    # NEW (optional) - doesn't break old pipeline
    ap.add_argument("--save_vis", action="store_true", help="Save match visualization images")
    ap.add_argument("--vis_max", type=int, default=1500, help="Max lines to draw per pair")
    ap.add_argument("--vis_first_k", type=int, default=3, help="Save vis only for first K pairs (to keep it light)")
    args = ap.parse_args()

    root = Path(args.root)
    seq_dir = root / "data" / "kitti_odometry" / args.seq
    calib_path = seq_dir / "calib.txt"
    K = parse_kitti_calib(calib_path)

    # for visualization (left images)
    img_dir = seq_dir / "image_0"

    match_base = root / args.matchdir / f"seq_{args.seq}"
    out_base = root / args.outdir / f"seq_{args.seq}"
    out_base.mkdir(parents=True, exist_ok=True)

    pts1_files = sorted(match_base.glob("*_pts1.npy"))
    if len(pts1_files) == 0:
        raise RuntimeError(f"No pts1 files found in {match_base}")

    pairs = []
    for p in pts1_files:
        name = p.name.replace("_pts1.npy", "")
        parts = name.split("_")
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    pairs = sorted(pairs)
    pairs = pairs[args.start: min(args.start + (args.n - 1), len(pairs))]

    print(f"[INFO] Part B_2: Essential matrix + pose with RANSAC, seq={args.seq}")
    print(f"[INFO] K=\n{K}")
    print(f"[INFO] Using matches from: {match_base}")
    print(f"[INFO] Output: {out_base}")
    print(f"[INFO] Pairs: {len(pairs)}")

    vis_dir = out_base / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    report = {"seq": args.seq, "pairs": []}

    for j, (t, t1) in enumerate(pairs):
        pts1 = np.load(match_base / f"{t}_{t1}_pts1.npy").astype(np.float32)
        pts2 = np.load(match_base / f"{t}_{t1}_pts2.npy").astype(np.float32)

        if pts1.shape[0] < 8:
            print(f"[WARN] {t}->{t1}: not enough matches")
            continue

        E, inlier_mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=args.ransac_prob,
            threshold=args.ransac_thresh
        )

        if E is None or inlier_mask is None:
            print(f"[WARN] {t}->{t1}: EssentialMat failed")
            continue

        inliers = int(inlier_mask.ravel().sum())

        # recover pose (uses the inliers mask)
        _, R, tvec, inlier_mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=inlier_mask)
        if inlier_mask_pose is None:
            inlier_mask_pose = inlier_mask
        inliers_pose = int(inlier_mask_pose.ravel().sum())

        # Save (KEEP EXACT NAMES)
        np.save(out_base / f"{t}_{t1}_R.npy", R)
        np.save(out_base / f"{t}_{t1}_t_unit.npy", tvec)  # unit translation (unknown scale)

        # Optional visualization (first K pairs only)
        if args.save_vis and j < args.vis_first_k:
            try:
                I1 = read_gray(img_dir / f"{t}.png")
                I2 = read_gray(img_dir / f"{t1}.png")
                vis = draw_matches_simple(I1, I2, pts1, pts2, inlier_mask=inlier_mask_pose.ravel(), max_draw=args.vis_max)
                cv2.imwrite(str(vis_dir / f"{t}_{t1}_matches_inliers.png"), vis)
            except Exception as e:
                print(f"[WARN] vis failed for {t}->{t1}: {e}")

        print(f"[OK] {t}->{t1}: matches={pts1.shape[0]}, inliers(RANSAC)={inliers}, inliersPose={inliers_pose}")

        report["pairs"].append({
            "t": t, "t1": t1,
            "matches": int(pts1.shape[0]),
            "inliers_ransac": inliers,
            "inliers_pose": inliers_pose,
            "R_path": str((out_base / f"{t}_{t1}_R.npy").as_posix()),
            "t_unit_path": str((out_base / f"{t}_{t1}_t_unit.npy").as_posix()),
            "vis_path": str((vis_dir / f"{t}_{t1}_matches_inliers.png").as_posix()) if (args.save_vis and j < args.vis_first_k) else None
        })

    (out_base / "pose_report.json").write_text(json.dumps(report, indent=2))
    print(f"[DONE] Part B_2 finished. Saved: {out_base/'pose_report.json'}")

if __name__ == "__main__":
    main()
