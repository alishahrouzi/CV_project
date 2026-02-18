import argparse
from pathlib import Path
import numpy as np
import cv2
import json

def parse_kitti_K(calib_path: Path) -> np.ndarray:
    P0=None
    for line in calib_path.read_text().splitlines():
        line=line.strip()
        if line.startswith("P0:"):
            vals=list(map(float,line.split()[1:]))
            P0=np.array(vals).reshape(3,4)
            break
    if P0 is None:
        raise ValueError("P0 not found")
    K = P0[:, :3].astype(np.float64)
    return K

def unproject(u, v, z, K):
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)

    ap.add_argument("--matchdir", default="outputs/partB_1_matching_sift")
    ap.add_argument("--depthdir", default="outputs/partA_3_depth")
    ap.add_argument("--depthtag", default="SAD_w9_d96_lr2.0_med5")

    ap.add_argument("--outdir", default="outputs/partB_3_pnp_scale")
    ap.add_argument("--pnp_reproj", type=float, default=3.0)
    ap.add_argument("--pnp_iters", type=int, default=2000)
    ap.add_argument("--pnp_conf", type=float, default=0.999)
    ap.add_argument("--min_points", type=int, default=100)
    args=ap.parse_args()

    root=Path(args.root)
    seq_dir = root/"data"/"kitti_odometry"/args.seq
    K = parse_kitti_K(seq_dir/"calib.txt")

    match_base = root/args.matchdir/f"seq_{args.seq}"
    depth_base = root/args.depthdir/f"seq_{args.seq}"/args.depthtag
    out_base = root/args.outdir/f"seq_{args.seq}"/args.depthtag
    out_base.mkdir(parents=True, exist_ok=True)

    pts1_files = sorted(match_base.glob("*_pts1.npy"))
    pairs=[]
    for p in pts1_files:
        name=p.name.replace("_pts1.npy","")
        a,b=name.split("_")
        pairs.append((a,b))
    pairs=sorted(pairs)
    pairs = pairs[args.start: min(args.start+(args.n-1), len(pairs))]

    print(f"[INFO] Part B_3: PnP+RANSAC for metric scale using stereo depth")
    print(f"[INFO] K=\n{K}")
    print(f"[INFO] Depth from: {depth_base}")
    print(f"[INFO] Pairs: {len(pairs)}")
    report={"seq":args.seq,"pairs":[]}

    for (t, t1) in pairs:
        pts1 = np.load(match_base/f"{t}_{t1}_pts1.npy")  # (N,2) in frame t
        pts2 = np.load(match_base/f"{t}_{t1}_pts2.npy")  # (N,2) in frame t+1

        depth = np.load(depth_base/f"{t}_depth.npy")  # depth of frame t
        H,W = depth.shape

        # build 3D-2D correspondences
        obj=[]
        img=[]
        for (p1, p2) in zip(pts1, pts2):
            u,v = float(p1[0]), float(p1[1])
            ui,vi = int(round(u)), int(round(v))
            if ui<0 or ui>=W or vi<0 or vi>=H:
                continue
            z = float(depth[vi, ui])
            if not np.isfinite(z) or z<=0.1 or z>200.0:
                continue
            X,Y,Z = unproject(u,v,z,K)
            obj.append([X,Y,Z])
            img.append([float(p2[0]), float(p2[1])])

        obj=np.array(obj, dtype=np.float32)
        img=np.array(img, dtype=np.float32)

        if obj.shape[0] < args.min_points:
            print(f"[WARN] {t}->{t1}: not enough valid 3D points ({obj.shape[0]})")
            report["pairs"].append({"t":t,"t1":t1,"valid3d":int(obj.shape[0]),"pnp_ok":False})
            continue

        dist = np.zeros((4,1), dtype=np.float64)  # assume no distortion (rectified)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=K,
            distCoeffs=dist,
            reprojectionError=args.pnp_reproj,
            iterationsCount=args.pnp_iters,
            confidence=args.pnp_conf,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok or inliers is None:
            print(f"[WARN] {t}->{t1}: PnP failed")
            report["pairs"].append({"t":t,"t1":t1,"valid3d":int(obj.shape[0]),"pnp_ok":False})
            continue

        inl = int(inliers.shape[0])
        R,_ = cv2.Rodrigues(rvec)
        trans_norm = float(np.linalg.norm(tvec))

        np.save(out_base/f"{t}_{t1}_R_metric.npy", R)
        np.save(out_base/f"{t}_{t1}_t_metric.npy", tvec)

        print(f"[OK] {t}->{t1}: valid3D={obj.shape[0]}, inliersPnP={inl}, |t|={trans_norm:.4f} m")
        report["pairs"].append({
            "t":t,"t1":t1,"valid3d":int(obj.shape[0]),
            "inliers_pnp":inl,"t_norm_m":trans_norm,
            "R_metric":str((out_base/f"{t}_{t1}_R_metric.npy").as_posix()),
            "t_metric":str((out_base/f"{t}_{t1}_t_metric.npy").as_posix()),
        })

    (out_base/"pnp_report.json").write_text(json.dumps(report, indent=2))
    print(f"[DONE] Part B_3 finished. Saved: {out_base/'pnp_report.json'}")

if __name__=="__main__":
    main()
