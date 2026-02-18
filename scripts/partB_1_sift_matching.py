import argparse
from pathlib import Path
import numpy as np
import cv2
import json

def read_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def create_sift():
    # OpenCV >= 4.4 usually supports SIFT in main build; otherwise need opencv-contrib-python
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    raise RuntimeError("cv2.SIFT_create not found. Install opencv-contrib-python.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--n", type=int, default=10, help="frames, uses pairs t->t+1")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--outdir", default="outputs/partB_1_matching_sift")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test")
    ap.add_argument("--keep_best", type=int, default=1200, help="cap matches after ratio test")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "data" / "kitti_odometry" / args.seq / "image_0"
    files = sorted(img_dir.glob("*.png"))
    end = min(args.start + args.n, len(files))
    if end - args.start < 2:
        raise RuntimeError("Need at least 2 frames")

    out_base = root / args.outdir / f"seq_{args.seq}"
    out_base.mkdir(parents=True, exist_ok=True)

    sift = create_sift()
    # SIFT -> L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2)

    report = {"seq": args.seq, "pairs": []}

    print(f"[INFO] Part B_1: SIFT detect+match on left images, frames {args.start}..{end-1}")
    for i in range(args.start, end-1):
        f1 = files[i]
        f2 = files[i+1]
        I1 = read_gray(f1)
        I2 = read_gray(f2)

        k1, d1 = sift.detectAndCompute(I1, None)
        k2, d2 = sift.detectAndCompute(I2, None)

        if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
            print(f"[WARN] {f1.stem}->{f2.stem}: not enough features")
            report["pairs"].append({"t": f1.stem, "t1": f2.stem, "kp1": len(k1), "kp2": len(k2), "matches": 0})
            continue

        # KNN match + Lowe ratio test
        knn = bf.knnMatch(d1, d2, k=2)
        good = []
        for m, n in knn:
            if m.distance < args.ratio * n.distance:
                good.append(m)

        good = sorted(good, key=lambda m: m.distance)
        good = good[:min(args.keep_best, len(good))]

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        np.save(out_base / f"{f1.stem}_{f2.stem}_pts1.npy", pts1)
        np.save(out_base / f"{f1.stem}_{f2.stem}_pts2.npy", pts2)

        print(f"[OK] {f1.stem}->{f2.stem}: kp=({len(k1)},{len(k2)}), goodMatches={len(good)}")
        report["pairs"].append({"t": f1.stem, "t1": f2.stem, "kp1": len(k1), "kp2": len(k2), "matches": len(good)})

    (out_base / "matching_report.json").write_text(json.dumps(report, indent=2))
    print(f"[DONE] Part B_1 (SIFT) finished. Saved: {out_base/'matching_report.json'}")

if __name__ == "__main__":
    main()
