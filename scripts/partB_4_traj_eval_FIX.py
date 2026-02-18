import argparse
from pathlib import Path
import numpy as np
import math
import json

def load_kitti_poses(pose_path: Path):
    Ts=[]
    for ln in pose_path.read_text().strip().splitlines():
        vals=list(map(float, ln.split()))
        M=np.array(vals,dtype=np.float64).reshape(3,4)
        T=np.eye(4,dtype=np.float64)
        T[:3,:4]=M
        Ts.append(T)
    return Ts

def make_T(R,t):
    T=np.eye(4,dtype=np.float64)
    T[:3,:3]=R
    T[:3,3]=t.reshape(3)
    return T

def invert_T(T):
    R=T[:3,:3]; t=T[:3,3]
    Ti=np.eye(4,dtype=np.float64)
    Ti[:3,:3]=R.T
    Ti[:3,3]=-(R.T@t)
    return Ti

def rotation_angle(R):
    tr=np.clip((np.trace(R)-1)/2, -1.0, 1.0)
    return float(np.arccos(tr))

def align_translation_scale_pos(est_xyz, gt_xyz):
    # translation align to first point + positive scale (abs)
    est0=est_xyz[0]; gt0=gt_xyz[0]
    est_c=est_xyz - est0
    gt_c=gt_xyz - gt0
    num=float(np.sum(gt_c*est_c))
    den=float(np.sum(est_c*est_c)+1e-12)
    s=num/den
    if s < 0:
        s = -s
        est_c = -est_c  # flip direction to keep "positive scale" equivalent
    est_aligned = s*est_c + gt0
    return est_aligned, float(s)

def ate_rmse(est_xyz, gt_xyz):
    e=est_xyz-gt_xyz
    return float(np.sqrt(np.mean(np.sum(e*e,axis=1))))

def compute_rpe(T_est, T_gt, step):
    trans_err=[]; rot_err=[]
    N=min(len(T_est), len(T_gt))
    for i in range(0, N-step):
        d_gt = invert_T(T_gt[i]) @ T_gt[i+step]
        d_est = invert_T(T_est[i]) @ T_est[i+step]
        E = invert_T(d_gt) @ d_est
        trans_err.append(np.linalg.norm(E[:3,3]))
        rot_err.append(rotation_angle(E[:3,:3]) * 180.0 / math.pi)
    if not trans_err:
        return None, None
    return float(np.sqrt(np.mean(np.array(trans_err)**2))), float(np.sqrt(np.mean(np.array(rot_err)**2)))

def build_traj(est_base: Path, pairs, mode: str):
    # mode A: chain with T_rel ; mode B: chain with inv(T_rel)
    T=[np.eye(4,dtype=np.float64)]
    for (t,t1) in pairs:
        R=np.load(est_base/f"{t}_{t1}_R_metric.npy")
        tv=np.load(est_base/f"{t}_{t1}_t_metric.npy")
        Trel=make_T(R,tv)
        if mode=="A":
            T.append(T[-1] @ Trel)
        else:
            T.append(T[-1] @ invert_T(Trel))
    return T

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--depthtag", default="SAD_w9_d96_lr2.0_med5")
    ap.add_argument("--pnpdir", default="outputs/partB_3_pnp_scale")
    ap.add_argument("--outdir", default="outputs/partB_4_traj_eval_fix")
    ap.add_argument("--steps", nargs="+", type=int, default=[1,5,10])
    args=ap.parse_args()

    root=Path(args.root)
    gt_path = root/"data"/"kitti_odometry"/"poses"/f"{args.seq}.txt"
    T_gt = load_kitti_poses(gt_path)

    est_base = root/args.pnpdir/f"seq_{args.seq}"/args.depthtag
    R_files = sorted(est_base.glob("*_R_metric.npy"))
    pairs=[]
    for rf in R_files:
        name=rf.name.replace("_R_metric.npy","")
        a,b=name.split("_")
        pairs.append((a,b))
    pairs=sorted(pairs)

    # Try both chaining conventions
    results=[]
    for mode in ["A","B"]:
        T_est = build_traj(est_base, pairs, mode)
        N=min(len(T_est), len(T_gt))
        est_xyz=np.array([T_est[i][:3,3] for i in range(N)], dtype=np.float64)
        gt_xyz =np.array([T_gt[i][:3,3]  for i in range(N)], dtype=np.float64)

        est_aligned, s = align_translation_scale_pos(est_xyz, gt_xyz)
        ate = ate_rmse(est_aligned, gt_xyz)
        rpe={}
        for st in args.steps:
            t_rmse, r_rmse = compute_rpe(T_est[:N], T_gt[:N], st)
            rpe[str(st)]={"trans_rmse_m":t_rmse,"rot_rmse_deg":r_rmse}
        results.append((mode, ate, s, T_est, est_xyz, est_aligned, gt_xyz, rpe))

    # Choose best (lower ATE)
    results.sort(key=lambda x: x[1])
    mode, ate, s, T_est, est_xyz, est_aligned, gt_xyz, rpe = results[0]

    out_base = root/args.outdir/f"seq_{args.seq}"/args.depthtag
    out_base.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_base/"traj_est_xyz.txt", est_xyz, fmt="%.6f")
    np.savetxt(out_base/"traj_est_aligned_xyz.txt", est_aligned, fmt="%.6f")
    np.savetxt(out_base/"traj_gt_xyz.txt", gt_xyz, fmt="%.6f")

    summary={
        "seq": args.seq,
        "chosen_mode": mode,
        "ATE_RMSE_m": ate,
        "scale_align_s": s,
        "RPE": rpe
    }
    (out_base/"vo_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[INFO] Chosen chaining mode = {mode} (A: Trel, B: inv(Trel))")
    print(f"[RESULT] ATE RMSE (aligned, meters) = {ate:.4f}")
    print(f"[RESULT] scale alignment s (positive) = {s:.4f}")
    for st,vals in rpe.items():
        print(f"[RESULT] RPE step={st}: transRMSE={vals['trans_rmse_m']}, rotRMSE(deg)={vals['rot_rmse_deg']}")
    print(f"[DONE] Saved: {out_base/'vo_eval_summary.json'}")

if __name__=="__main__":
    main()
