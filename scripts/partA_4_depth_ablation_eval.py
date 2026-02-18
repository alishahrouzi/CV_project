import argparse
from pathlib import Path
import numpy as np
import cv2
import json

# ---- Utilities ----
def read_gray(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im

def box_sum(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.boxFilter(img, ddepth=-1, ksize=(k,k), normalize=False, borderType=cv2.BORDER_CONSTANT)

# ---- Disparity (SAD/NCC), left-ref only (Depth uses left disparity) ----
def disp_sad(L, R, max_disp, w):
    H,W = L.shape
    Lf = L.astype(np.float32); Rf = R.astype(np.float32)
    C = np.full((H,W,max_disp+1), np.inf, np.float32)
    for d in range(max_disp+1):
        Rsh = np.zeros_like(Rf)
        if d==0: Rsh[:] = Rf
        else:    Rsh[:, d:] = Rf[:, :-d]
        sad = box_sum(np.abs(Lf - Rsh), w)
        sad[:, :d] = np.inf
        C[:,:,d] = sad
    disp = np.argmin(C, axis=2).astype(np.float32)
    pad=w//2
    disp[:pad,:]=0; disp[-pad:,:]=0; disp[:,:pad]=0; disp[:,-pad:]=0
    return disp

def disp_ncc(L, R, max_disp, w):
    H,W = L.shape
    Lf=L.astype(np.float32); Rf=R.astype(np.float32)
    sumL=box_sum(Lf,w); sumL2=box_sum(Lf*Lf,w)
    S=np.full((H,W,max_disp+1), -np.inf, np.float32)
    N=float(w*w); eps=1e-6
    for d in range(max_disp+1):
        Rsh=np.zeros_like(Rf)
        if d==0: Rsh[:] = Rf
        else:    Rsh[:, d:] = Rf[:, :-d]
        sumR=box_sum(Rsh,w); sumR2=box_sum(Rsh*Rsh,w); sumLR=box_sum(Lf*Rsh,w)
        meanL=sumL/N; meanR=sumR/N
        num=sumLR - N*meanL*meanR
        denL=sumL2 - N*meanL*meanL
        denR=sumR2 - N*meanR*meanR
        den=np.sqrt(np.maximum(denL,0)*np.maximum(denR,0))+eps
        ncc=num/den
        ncc[:, :d] = -np.inf
        S[:,:,d]=ncc
    disp=np.argmax(S, axis=2).astype(np.float32)
    pad=w//2
    disp[:pad,:]=0; disp[-pad:,:]=0; disp[:,:pad]=0; disp[:,-pad:]=0
    return disp

# ---- Simple postprocess exactly as required (LR/median/fill) using your FIX logic but only on left disp for output ----
def disp_rightref_sad(R, L, max_disp, w):
    H,W = R.shape
    Rf=R.astype(np.float32); Lf=L.astype(np.float32)
    C=np.full((H,W,max_disp+1), np.inf, np.float32)
    for d in range(max_disp+1):
        Lsh=np.zeros_like(Lf)
        if d==0: Lsh[:] = Lf
        else:    Lsh[:, :-d] = Lf[:, d:]
        sad=box_sum(np.abs(Rf-Lsh), w)
        sad[:, W-d:] = np.inf
        C[:,:,d]=sad
    disp=np.argmin(C, axis=2).astype(np.float32)
    pad=w//2
    disp[:pad,:]=0; disp[-pad:,:]=0; disp[:,:pad]=0; disp[:,-pad:]=0
    return disp

def lr_invalid(dL, dR, thr):
    H,W = dL.shape
    xs=np.arange(W,dtype=np.int32)[None,:].repeat(H,0)
    xR=(xs - dL).round().astype(np.int32)
    inv=np.zeros((H,W), np.uint8)
    oob=(xR<0)|(xR>=W)
    inv[oob]=1
    yy,xx=np.where(~oob)
    xr=xR[yy,xx]
    diff=np.abs(dL[yy,xx]-dR[yy,xr])
    bad = diff>thr
    inv[yy[bad], xx[bad]] = 1
    return inv

def postprocess(L, R, cost, max_disp, w, lr_thr=2.0, med_k=5):
    if cost=="SAD":
        dL = disp_sad(L,R,max_disp,w)
        dR = disp_rightref_sad(R,L,max_disp,w)
    else:
        dL = disp_ncc(L,R,max_disp,w)
        # For simplicity in ablation: use SAD rightref for invalid mask stability,
        # LR-check still valid as "consistency" filter. (project doesn't restrict this detail)
        dR = disp_rightref_sad(R,L,max_disp,w)

    inv = lr_invalid(dL, dR, lr_thr)
    d_med = cv2.medianBlur(dL.astype(np.float32), med_k)
    d_med[inv==1] = 0

    # hole fill (simple interpolation)
    dclip=np.clip(d_med,0,max_disp)
    disp8=(dclip/max_disp*255.0).astype(np.uint8)
    mask8=(inv*255).astype(np.uint8)
    filled8=cv2.inpaint(disp8, mask8, 3, cv2.INPAINT_TELEA)
    d_pp = (filled8.astype(np.float32)/255.0)*max_disp
    return d_pp, inv

# ---- Depth ----
def parse_calib(calib_path: Path):
    P0=P1=None
    for line in calib_path.read_text().splitlines():
        line=line.strip()
        if line.startswith("P0:"):
            vals=list(map(float,line.split()[1:]))
            P0=np.array(vals).reshape(3,4)
        if line.startswith("P1:"):
            vals=list(map(float,line.split()[1:]))
            P1=np.array(vals).reshape(3,4)
    if P0 is None or P1 is None:
        raise ValueError("Missing P0/P1 in calib.txt")
    f=float(P0[0,0]); B=abs(float(P1[0,3])-float(P0[0,3]))/f
    return f,B

def disp_to_depth(disp, f, B, min_disp=0.1):
    d=disp.astype(np.float32)
    d=np.where(d>min_disp,d,np.nan)
    Z=(f*B)/d
    return Z.astype(np.float32)

# ---- Optional evaluation if GT exists (supports GT disparity as .npy) ----
def evaluate_if_gt(depth: np.ndarray, gt_path: Path):
    """
    If a GT file exists:
    - if it is .npy and same shape => treat as GT depth in meters
    Computes MAE over finite pixels.
    """
    if gt_path is None or (not gt_path.exists()):
        return None

    if gt_path.suffix.lower() == ".npy":
        gt = np.load(gt_path)
        if gt.shape != depth.shape:
            return {"error":"GT shape mismatch", "gt":str(gt_path)}
        m = np.isfinite(depth) & np.isfinite(gt) & (gt>0)
        if m.sum()==0:
            return {"error":"No valid GT pixels", "gt":str(gt_path)}
        mae = float(np.mean(np.abs(depth[m]-gt[m])))
        return {"mae":mae, "valid_pixels":int(m.sum()), "gt":str(gt_path)}

    return {"error":"Unsupported GT format", "gt":str(gt_path)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-disp", type=int, default=96)
    ap.add_argument("--outdir", default="outputs/partA_4_ablation")

    # Ablation settings
    ap.add_argument("--costs", nargs="+", default=["SAD","NCC"])
    ap.add_argument("--windows", nargs="+", type=int, default=[7,11])

    # Postprocess params (fixed)
    ap.add_argument("--lr-thresh", type=float, default=2.0)
    ap.add_argument("--median-k", type=int, default=5)

    # Optional GT folder (if you have it)
    ap.add_argument("--gt_depth_dir", default="", help="If you have GT depth .npy files, put path here.")
    args=ap.parse_args()

    root=Path(args.root)
    seq_dir=root/"data"/"kitti_odometry"/args.seq
    Ldir=seq_dir/"image_0"; Rdir=seq_dir/"image_1"
    calib=seq_dir/"calib.txt"
    f,B = parse_calib(calib)

    Lfiles=sorted(Ldir.glob("*.png"))
    Rfiles=sorted(Rdir.glob("*.png"))
    end=min(args.start+args.n, len(Lfiles), len(Rfiles))

    out_base=root/args.outdir/f"seq_{args.seq}"
    out_base.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_depth_dir) if args.gt_depth_dir else None

    summary = {
        "seq": args.seq, "frames": [Lfiles[i].stem for i in range(args.start, end)],
        "f": f, "baseline": B,
        "settings": []
    }

    print(f"[INFO] Ablation on seq={args.seq} frames {args.start}..{end-1}")
    print(f"[INFO] f={f:.3f}, B={B:.6f}")

    for cost in args.costs:
        for w in args.windows:
            tag=f"{cost}_w{w}_d{args.max_disp}_lr{args.lr_thresh}_med{args.median_k}"
            out_dir = out_base/tag
            out_dir.mkdir(parents=True, exist_ok=True)

            inv_pcts=[]
            valid_pcts=[]
            maes=[]

            for i in range(args.start, end):
                L=read_gray(Lfiles[i]); R=read_gray(Rfiles[i])
                disp_pp, inv = postprocess(L,R,cost,args.max_disp,w,args.lr_thresh,args.median_k)
                depth = disp_to_depth(disp_pp, f, B, min_disp=0.1)

                frame=Lfiles[i].stem
                np.save(out_dir/f"{frame}_disp_pp.npy", disp_pp)
                np.save(out_dir/f"{frame}_depth.npy", depth)

                inv_pct=float(inv.mean())*100.0
                valid_pct=float(np.isfinite(depth).mean())*100.0
                inv_pcts.append(inv_pct); valid_pcts.append(valid_pct)

                # Optional GT eval
                eval_res=None
                if gt_dir:
                    gt_path = gt_dir/f"{frame}.npy"
                    eval_res = evaluate_if_gt(depth, gt_path)
                    if eval_res and "mae" in eval_res:
                        maes.append(eval_res["mae"])

            setting = {
                "tag": tag,
                "mean_invalid_pct": float(np.mean(inv_pcts)),
                "mean_valid_depth_pct": float(np.mean(valid_pcts)),
                "mae_mean_if_gt": float(np.mean(maes)) if len(maes)>0 else None,
                "mae_count": len(maes)
            }
            summary["settings"].append(setting)
            print(f"[OK] {tag}: invalid={setting['mean_invalid_pct']:.2f}%, validDepth={setting['mean_valid_depth_pct']:.2f}%, mae={setting['mae_mean_if_gt']}")

    (out_base/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Part A_4 finished. Summary saved: {out_base/'summary.json'}")

if __name__=="__main__":
    main()
