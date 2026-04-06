"""
compute_mccann_yodzis_gpu.py — Batch 3D Safe Set for McCann-Yodzis (Windows GPU)
=================================================================================
GPU (CuPy) for RK4 map, multi-threaded EDT for sculpting.

McCann-Yodzis with cooperative hunting (Duarte et al. 2009, Capeáns et al. 2014):
  dR/dt = R(1 - R/K) - xc·yc·C·R/(R + R0)
  dC/dt = xc·C·(yc·R/(R+R0) - 1) - c(P)·yp·C/(C + C0)
  dP/dt = c(P)·yp·C/(C + C0) - xp·P
  c(P)  = xp·(1-s)·P + xi·s·P²

Parameters: K=0.99, s=0.07, xc=0.4, yc=2.009, xp=xi=0.08, yp=2.876,
            R0=0.16129, C0=0.5
After boundary crisis (sc=0.04166): transient chaos → predator extinction.

Q0 = full box.  3 Δt × 4 (ξ₀,u₀) = 12 safe sets.

Install:  pip install numpy scipy scikit-image cupy-cuda12x edt
Usage:    python compute_mccann_yodzis_gpu.py
"""

import numpy as np
import os, time, sys

# ======================== GPU SETUP ======================================
try:
    import cupy as cp
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    gpu_mem_total = cp.cuda.Device(0).mem_info[1] / 1e9
    GPU = True
except Exception:
    cp = np; GPU = False
    gpu_name = "None"; gpu_mem_total = 0

try:
    import edt as edt_lib
    FAST_EDT = True
except ImportError:
    FAST_EDT = False
    print("WARNING: install 'edt' for fast sculpting: pip install edt\n")

from scipy.ndimage import distance_transform_edt as scipy_edt

ncores = os.cpu_count() or 4

def fast_edt(input_arr):
    if FAST_EDT:
        return edt_lib.edt(input_arr, parallel=ncores)
    return scipy_edt(input_arr)

# ======================== MODEL PARAMETERS ================================
K_PAR  = 0.99
S_PAR  = 0.07
XC_PAR = 0.4
YC_PAR = 2.009
XP_PAR = 0.08
XI_PAR = 0.08
YP_PAR = 2.876
R0_PAR = 0.16129
C0_PAR = 0.5

# Bounding box
X_MIN, X_MAX = -0.1, 1.1      # R (resource)
Y_MIN, Y_MAX = -0.1, 1.1      # C (consumer)
Z_MIN, Z_MAX =  0.4, 1.1      # P (predator)

DT_LIST = [50, 100, 150]
XU_LIST = [(0.10, 0.07), (0.06, 0.04), (0.02, 0.015), (0.01, 0.008)]

DT_RK4 = 0.01

# Grid: anchor on P axis (shortest range = 0.7), NP = 601
NZ = 601

# ======================== BANNER =========================================
print()
print("=" * 65)
print("  BATCH 3D SAFE SET — McCann-Yodzis ecological model (Win GPU)")
print("  Transient chaos → predator extinction — Q0 = full box")
print("=" * 65)
if GPU: print(f"  GPU: {gpu_name} ({gpu_mem_total:.1f} GB)")
else:   print("  GPU: NOT AVAILABLE — using CPU (slow)")
print(f"  CPU: {ncores} cores")
print(f"  Model: K={K_PAR}, s={S_PAR}, xc={XC_PAR}, yc={YC_PAR}")
print(f"         xp=xi={XP_PAR}, yp={YP_PAR}, R0={R0_PAR}, C0={C0_PAR}")
print(f"  Box: R∈[{X_MIN},{X_MAX}], C∈[{Y_MIN},{Y_MAX}], P∈[{Z_MIN},{Z_MAX}]")
print()
print(f"  Δt values:     {DT_LIST}")
print(f"  (ξ₀,u₀) pairs: {XU_LIST}")
print(f"  Total: {len(DT_LIST)*len(XU_LIST)} combinations")
print()

# ======================== DERIVED GRID ====================================
range_x = X_MAX - X_MIN
range_y = Y_MAX - Y_MIN
range_z = Z_MAX - Z_MIN

h = range_z / (NZ - 1)
NX = int(np.ceil(range_x / h)) + 1
NY = int(np.ceil(range_y / h)) + 1

xv = np.array([X_MIN + i*h for i in range(NX)], dtype=np.float32)
yv = np.array([Y_MIN + i*h for i in range(NY)], dtype=np.float32)
zv = np.array([Z_MIN + i*h for i in range(NZ)], dtype=np.float32)

total_vox = NX * NY * NZ
bool_gb = total_vox / 1e9
edt_gb = total_vox * 8 / 1e9
use_full_edt = (edt_gb < 10.0)

print(f"  Grid: {NX} × {NY} × {NZ} = {total_vox:,} voxels")
print(f"  h = {h:.6f}")
print(f"  R covers [{xv[0]:.4f}, {xv[-1]:.4f}]")
print(f"  C covers [{yv[0]:.4f}, {yv[-1]:.4f}]")
print(f"  P covers [{zv[0]:.4f}, {zv[-1]:.4f}]")
print(f"  Bool: {bool_gb:.2f} GB, EDT: {'full' if use_full_edt else 'slabs'} ({edt_gb:.1f} GB)")
for xi0, u0 in XU_LIST:
    print(f"  (ξ₀={xi0}, u₀={u0}): r_ξ₀={xi0/h:.1f}vox, r_u₀={u0/h:.1f}vox")
print()

batch_t0 = time.time()

# ======================== Q0 = FULL BOX ===================================
print("=" * 65)
print("  Q0 = full box (predator extinction = escape from box)")
print("=" * 65)
Q0 = np.ones((NX, NY, NZ), dtype=np.bool_)
print(f"  Q0: {int(Q0.sum()):,} pts\n")

# ======================== EDT HELPER =====================================
def edt_threshold_inplace(input_bool, threshold, out_bool):
    if use_full_edt:
        dist = fast_edt(input_bool)
        np.less_equal(dist, threshold, out=out_bool)
        del dist
    else:
        margin = int(np.ceil(threshold)) + 2
        chunk = max(30, min(120, NZ // max(ncores // 2, 1)))
        out_bool[:] = False
        for z0 in range(0, NZ, chunk):
            z1 = min(z0 + chunk, NZ)
            zlo = max(0, z0 - margin)
            zhi = min(NZ, z1 + margin)
            slab = np.array(input_bool[:, :, zlo:zhi])
            dist = fast_edt(slab)
            out_bool[:, :, z0:z1] = dist[:, :, (z0-zlo):(z1-zlo)] <= threshold
            del dist, slab

# ======================== GPU RK4 MAP ====================================
def compute_rk4_map(dt_map):
    nsteps = round(dt_map / DT_RK4)
    print(f"\n  RK4 map: Δt={dt_map}, {nsteps} steps, dt_rk4={DT_RK4}")
    t0 = time.time()

    Ix = np.empty((NX, NY, NZ), dtype=np.int16)
    Iy = np.empty((NX, NY, NZ), dtype=np.int16)
    Iz = np.empty((NX, NY, NZ), dtype=np.int16)
    img_ok = np.empty((NX, NY, NZ), dtype=np.bool_)

    if GPU:
        free_mem = cp.cuda.Device(0).mem_info[0]
        slab_ni = max(1, int(free_mem * 0.55 / (18 * NY * NZ * 4)))
    else:
        slab_ni = max(1, min(30, NX))
    slab_ni = min(slab_ni, NX)
    n_slabs = (NX + slab_ni - 1) // slab_ni
    print(f"    {nsteps} steps, {n_slabs} slab(s) of {slab_ni}")

    # Model parameters as float32
    kk  = np.float32(K_PAR)
    ss  = np.float32(S_PAR)
    xc  = np.float32(XC_PAR)
    ycc = np.float32(YC_PAR)
    xpp = np.float32(XP_PAR)
    xii = np.float32(XI_PAR)
    ypp = np.float32(YP_PAR)
    r0  = np.float32(R0_PAR)
    c0  = np.float32(C0_PAR)
    dt  = np.float32(DT_RK4)
    dt2 = dt / 2
    dt6 = dt / 6

    for slab in range(n_slabs):
        i0 = slab * slab_ni
        i1 = min(i0 + slab_ni, NX)
        st0 = time.time()

        R0s, C0s, P0s = np.meshgrid(xv[i0:i1], yv, zv, indexing='ij')
        xp_lib = cp if GPU else np
        Rf = xp_lib.asarray(R0s); Cf = xp_lib.asarray(C0s); Pf = xp_lib.asarray(P0s)
        del R0s, C0s, P0s

        for step in range(nsteps):
            # k1
            cP = xpp*(1-ss)*Pf + xii*ss*Pf*Pf
            kR1 = Rf*(1 - Rf/kk) - xc*ycc*Cf*Rf/(Rf + r0)
            kC1 = xc*Cf*(ycc*Rf/(Rf + r0) - 1) - cP*ypp*Cf/(Cf + c0)
            kP1 = cP*ypp*Cf/(Cf + c0) - xpp*Pf

            Rt = Rf + dt2*kR1; Ct = Cf + dt2*kC1; Pt = Pf + dt2*kP1

            # k2
            cP = xpp*(1-ss)*Pt + xii*ss*Pt*Pt
            kR2 = Rt*(1 - Rt/kk) - xc*ycc*Ct*Rt/(Rt + r0)
            kC2 = xc*Ct*(ycc*Rt/(Rt + r0) - 1) - cP*ypp*Ct/(Ct + c0)
            kP2 = cP*ypp*Ct/(Ct + c0) - xpp*Pt

            Rt = Rf + dt2*kR2; Ct = Cf + dt2*kC2; Pt = Pf + dt2*kP2

            # k3
            cP = xpp*(1-ss)*Pt + xii*ss*Pt*Pt
            kR3 = Rt*(1 - Rt/kk) - xc*ycc*Ct*Rt/(Rt + r0)
            kC3 = xc*Ct*(ycc*Rt/(Rt + r0) - 1) - cP*ypp*Ct/(Ct + c0)
            kP3 = cP*ypp*Ct/(Ct + c0) - xpp*Pt

            Rt = Rf + dt*kR3; Ct = Cf + dt*kC3; Pt = Pf + dt*kP3

            # k4
            cP = xpp*(1-ss)*Pt + xii*ss*Pt*Pt
            kR4 = Rt*(1 - Rt/kk) - xc*ycc*Ct*Rt/(Rt + r0)
            kC4 = xc*Ct*(ycc*Rt/(Rt + r0) - 1) - cP*ypp*Ct/(Ct + c0)
            kP4 = cP*ypp*Ct/(Ct + c0) - xpp*Pt

            Rf += dt6*(kR1 + 2*kR2 + 2*kR3 + kR4)
            Cf += dt6*(kC1 + 2*kC2 + 2*kC3 + kC4)
            Pf += dt6*(kP1 + 2*kP2 + 2*kP3 + kP4)

        if GPU:
            Rc = cp.asnumpy(Rf); Cc = cp.asnumpy(Cf); Pc = cp.asnumpy(Pf)
            del Rf, Cf, Pf, Rt, Ct, Pt, cP
            del kR1, kC1, kP1, kR2, kC2, kP2, kR3, kC3, kP3, kR4, kC4, kP4
            cp.get_default_memory_pool().free_all_blocks()
        else:
            Rc = Rf; Cc = Cf; Pc = Pf

        Ix[i0:i1] = np.clip(np.round((Rc - X_MIN) / h).astype(np.int16), 0, NX-1)
        Iy[i0:i1] = np.clip(np.round((Cc - Y_MIN) / h).astype(np.int16), 0, NY-1)
        Iz[i0:i1] = np.clip(np.round((Pc - Z_MIN) / h).astype(np.int16), 0, NZ-1)
        img_ok[i0:i1] = (
            (Rc >= xv[0]) & (Rc <= xv[-1]) &
            (Cc >= yv[0]) & (Cc <= yv[-1]) &
            (Pc >= zv[0]) & (Pc <= zv[-1])
        )
        del Rc, Cc, Pc
        print(f"    Slab {slab+1}/{n_slabs}: {time.time()-st0:.1f}s", flush=True)

    print(f"    RK4 done: {time.time()-t0:.1f}s")
    inside_pct = 100.0 * img_ok.sum() / total_vox
    print(f"    Image inside box: {inside_pct:.1f}%")
    return Ix, Iy, Iz, img_ok

# ======================== SCULPT + ASYMPTOTIC ============================
def compute_safeset(Q0, Ix, Iy, Iz, img_ok, xi0, u0):
    r_u0_vox  = u0 / h
    r_xi0_vox = xi0 / h
    print(f"\n    Sculpting (ξ₀={xi0}, u₀={u0}, r_u₀={r_u0_vox:.1f}vox, r_ξ₀={r_xi0_vox:.1f}vox)...")
    t0 = time.time()

    Q = Q0.copy()
    buf1 = np.empty((NX, NY, NZ), dtype=np.bool_)
    buf2 = np.empty((NX, NY, NZ), dtype=np.bool_)

    for it in range(60):
        it_t0 = time.time()
        np_ = int(Q.sum())

        # Dilate complement of Q by u0, then dilate result by xi0
        np.logical_not(Q, out=buf1)
        edt_threshold_inplace(buf1, r_u0_vox, buf2)
        edt_threshold_inplace(buf2, r_xi0_vox, buf1)
        np.logical_not(buf1, out=buf1)

        # Keep only points whose image lands in buf1 (reachable safe region)
        buf2[:] = False
        blk = max(10, min(50, NX // 10))
        for i0 in range(0, NX, blk):
            i1 = min(i0 + blk, NX)
            ix_sl = Ix[i0:i1].astype(np.intp)
            iy_sl = Iy[i0:i1].astype(np.intp)
            iz_sl = Iz[i0:i1].astype(np.intp)
            buf2[i0:i1] = Q[i0:i1] & img_ok[i0:i1] & buf1[ix_sl, iy_sl, iz_sl]

        nr = np_ - int(buf2.sum())
        print(f"      {it+1:2d}: {np_:>12,} -{nr:>10,}  ({time.time()-it_t0:.1f}s)", flush=True)
        if nr == 0:
            print(f"      Converged iter {it+1}")
            break
        Q, buf2 = buf2, Q

    N_safe = int(Q.sum())
    print(f"      Safe set: {N_safe:,} pts ({time.time()-t0:.1f}s)")

    # Asymptotic safe set
    print(f"    Asymptotic...")
    t0 = time.time()
    r_reach_vox = (xi0 + u0) / h
    A = Q.copy()
    for it in range(40):
        nb = int(A.sum())
        buf1[:] = False
        blk = max(10, min(50, NX // 10))
        for i0 in range(0, NX, blk):
            i1 = min(i0 + blk, NX)
            active = A[i0:i1] & img_ok[i0:i1]
            if not active.any():
                continue
            ix_sl = Ix[i0:i1][active].astype(np.intp)
            iy_sl = Iy[i0:i1][active].astype(np.intp)
            iz_sl = Iz[i0:i1][active].astype(np.intp)
            buf1[ix_sl, iy_sl, iz_sl] = True

        np.logical_not(buf1, out=buf2)
        edt_threshold_inplace(buf2, r_reach_vox, buf1)
        np.logical_and(A, buf1, out=buf2)
        na = int(buf2.sum())
        print(f"      {it+1:2d}: {nb:>12,} -> {na:,}", flush=True)
        if na == nb:
            print(f"      Converged iter {it+1}")
            break
        A, buf2 = buf2, A

    Q_asymp = A
    N_asymp = int(Q_asymp.sum())
    print(f"      Asymptotic: {N_asymp:,} pts ({time.time()-t0:.1f}s)")
    del buf1, buf2
    return Q, Q_asymp, N_safe, N_asymp

# ======================== MAIN BATCH LOOP ================================
results = []

for dt_idx, dt_map in enumerate(DT_LIST):
    print()
    print("#" * 65)
    print(f"#  MAP {dt_idx+1}/{len(DT_LIST)}: Δt = {dt_map}")
    print(f"#  RK4: {round(dt_map/DT_RK4)} steps")
    print("#" * 65)

    Ix, Iy, Iz, img_ok = compute_rk4_map(dt_map)

    for xu_idx, (xi0, u0) in enumerate(XU_LIST):
        combo_num = dt_idx * len(XU_LIST) + xu_idx + 1
        combo_total = len(DT_LIST) * len(XU_LIST)
        tag = f"dt{dt_map}_xi{xi0}_u{u0}"
        npz_file = f"mccann_yodzis_{tag}.npz"

        print()
        print("  " + "=" * 55)
        print(f"  >>>  COMBO {combo_num} / {combo_total}  <<<")
        print(f"  >>>  Δt={dt_map}  ξ₀={xi0}  u₀={u0}")
        print(f"  >>>  {npz_file}")
        print("  " + "=" * 55)

        r_u0_check = u0 / h
        if r_u0_check < 3:
            print(f"    SKIPPED: r_u₀={r_u0_check:.1f} vox < 3 (too coarse)")
            results.append((tag, "SKIP", 0, 0, 0))
            continue

        comb_t0 = time.time()
        Q_safe, Q_asymp, N_safe, N_asymp = compute_safeset(
            Q0, Ix, Iy, Iz, img_ok, xi0, u0)

        print(f"    Saving {npz_file}...", flush=True)
        npz_path = os.path.join(os.getcwd(), npz_file)
        np.savez_compressed(npz_path,
            Q       = np.packbits(Q_safe.ravel()),
            Q_asymp = np.packbits(Q_asymp.ravel()),
            NX=NX, NY=NY, NZ=NZ, h=h,
            x_min=X_MIN, x_max=float(xv[-1]),
            y_min=Y_MIN, y_max=float(yv[-1]),
            z_min=Z_MIN, z_max=float(zv[-1]),
            K_par=K_PAR, s_par=S_PAR,
            xc_par=XC_PAR, yc_par=YC_PAR,
            xp_par=XP_PAR, xi_par=XI_PAR,
            yp_par=YP_PAR, R0_par=R0_PAR, C0_par=C0_PAR,
            dt_map=dt_map, xi0=xi0, u0_ctrl=u0,
            N_safe=N_safe, N_asymp=N_asymp)
        npz_sz = os.path.getsize(npz_path) / 1e6

        del Q_safe, Q_asymp
        comb_time = time.time() - comb_t0
        results.append((tag, "OK", N_safe, N_asymp, comb_time))

        elapsed_total = time.time() - batch_t0
        print(f"    {npz_file} ({npz_sz:.1f} MB)")
        print(f"    Time: {comb_time:.0f}s ({comb_time/60:.1f} min)")
        print(f"    Elapsed total: {elapsed_total/3600:.1f} h")

    del Ix, Iy, Iz, img_ok

# ======================== FINAL SUMMARY ==================================
total_time = time.time() - batch_t0
print()
print("#" * 65)
print("#  BATCH COMPLETE")
print("#" * 65)
print(f"  Model: K={K_PAR}, s={S_PAR}")
print(f"  Grid: {NX}×{NY}×{NZ}, h={h:.6f}")
print(f"  Box: R∈[{X_MIN},{X_MAX}], C∈[{Y_MIN},{Y_MAX}], P∈[{Z_MIN},{Z_MAX}]")
print(f"  Q0 = full box")
print()
print(f"  {'Tag':<44} {'St':>4} {'Safe':>14} {'Asymp':>14} {'Time':>8}")
print(f"  {'-'*44} {'-'*4} {'-'*14} {'-'*14} {'-'*8}")
for tag, status, ns, na, t in results:
    if status == "OK":
        print(f"  {tag:<44} {status:>4} {ns:>14,} {na:>14,} {t/60:>6.1f}m")
    else:
        print(f"  {tag:<44} {status:>4} {'—':>14} {'—':>14} {'—':>8}")
print()
print(f"  Total: {total_time:.0f}s = {total_time/3600:.1f} h")
print("#" * 65)
print(f"\nOutput .npz files in: {os.getcwd()}")
