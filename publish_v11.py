"""
publish_v11.py — Convert .npz → .bin (pyfqmr) → .gz
=====================================================
Same ds/sigma/threshold for safe and asymptotic.
Adaptive ds based on u0. gc.collect between operations.
"""

import numpy as np
import os, struct, sys, time, glob, gzip, shutil, gc

from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

try:
    import pyfqmr
except ImportError:
    print("ERROR: pip install pyfqmr"); sys.exit(1)

def recompute_normals(verts, faces):
    normals = np.zeros_like(verts, dtype=np.float64)
    for i in range(len(faces)):
        v0, v1, v2 = verts[faces[i,0]], verts[faces[i,1]], verts[faces[i,2]]
        fn = np.cross(v1-v0, v2-v0)
        normals[faces[i,0]] += fn
        normals[faces[i,1]] += fn
        normals[faces[i,2]] += fn
    norms = np.sqrt(np.sum(normals**2, axis=1, keepdims=True))
    norms[norms==0] = 1
    return (normals / norms).astype(np.float32)

def decimate_mesh(verts, faces, target):
    if len(faces) <= target:
        n = recompute_normals(verts, faces)
        return verts.astype(np.float32), faces.astype(np.uint32), n
    s = pyfqmr.Simplify()
    s.setMesh(verts.astype(np.float64), faces.astype(np.int32))
    s.simplify_mesh(target_count=target, aggressiveness=7, verbose=0)
    v, f, _ = s.getMesh()
    n = recompute_normals(v, f)
    return v.astype(np.float32), f.astype(np.uint32), n

# ======================== PARSE ARGS =====================================
files = []
max_faces = 500000
ds_override = None
do_push = True
repo_path = None

i = 1
while i < len(sys.argv):
    a = sys.argv[i]
    if a == '--faces':       max_faces = int(sys.argv[i+1]); i += 2
    elif a == '--downsample': ds_override = int(sys.argv[i+1]); i += 2
    elif a == '--no-push':    do_push = False; i += 1
    elif a == '--repo':       repo_path = sys.argv[i+1]; i += 2
    else:                     files.extend(glob.glob(a)); i += 1

if not files:
    files = sorted(glob.glob('*.npz'))
if not files:
    print("No .npz files found."); sys.exit(1)

files = [f for f in files if f.endswith('.npz')]
print(f"\nProcessing {len(files)} file(s), max_faces={max_faces:,}\n")

# ======================== GIT REPO =======================================
if do_push and not repo_path:
    candidates = [
        os.path.expanduser('~/partial_control'),
        os.path.expanduser('~/Desktop/partial_control'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'partial_control'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, '.git')):
            repo_path = c; break
    if not repo_path:
        repo_input = input("Path to repo (or Enter to skip push): ").strip()
        if repo_input: repo_path = repo_input
        else: do_push = False

# ======================== PROCESS ========================================
bin_files = []

for npz_path in files:
    if not os.path.exists(npz_path): continue

    print(f"{'='*50}")
    print(f"  {npz_path}")
    print(f"{'='*50}")
    t0 = time.time()

    data = np.load(npz_path)
    N = int(data['N']); L = int(data['L']); h = float(data['h'])
    Cp = data['Cp']; Cm = data['Cm']
    x_min = -L; z_min = 0

    Q = np.unpackbits(data['Q'])[:N**3].reshape((N,N,N)).astype(np.bool_)
    Q_asymp = np.unpackbits(data['Q_asymp'])[:N**3].reshape((N,N,N)).astype(np.bool_)
    print(f"  Grid: {N}³, Safe: {int(data['N_safe']):,}, Asym: {int(data['N_asymp']):,}")

    # Adaptive ds based on u0 — same for both meshes
    u0_val = float(data.get('u0_ctrl', 1.0))
    r_u0_vox = u0_val / h
    if ds_override:
        ds = ds_override
    else:
        ds = max(2, min(int(r_u0_vox / 3), N // 200))

    sig = 0.8  # smooths voxel jaggies, preserves layers ≥4 voxels apart
    thr = 0.3
    sp = (h*ds, h*ds, h*ds)

    print(f"  u₀={u0_val}, r_u0={r_u0_vox:.1f}vox, ds={ds} ({N//ds}³), σ={sig}, thr={thr}")

    # === SAFE SET: smooth → marching cubes → decimate → free ===
    Q_ds = Q[::ds,::ds,::ds].astype(np.float32)
    del Q; gc.collect()
    Q_sm = gaussian_filter(Q_ds, sigma=sig)
    del Q_ds; gc.collect()
    vs,fs,ns,_ = marching_cubes(Q_sm, level=thr, spacing=sp)
    vs[:,0]+=x_min; vs[:,1]+=x_min; vs[:,2]+=z_min
    del Q_sm; gc.collect()
    print(f"  Safe raw: {len(fs):,}f")
    vs,fs,ns = decimate_mesh(vs, fs, max_faces)
    print(f"  Safe dec: {len(fs):,}f")
    gc.collect()

    # === ASYMPTOTIC: identical params ===
    Q_ds = Q_asymp[::ds,::ds,::ds].astype(np.float32)
    del Q_asymp; gc.collect()
    Q_asm = gaussian_filter(Q_ds, sigma=sig)
    del Q_ds; gc.collect()
    va,fa,na_,_ = marching_cubes(Q_asm, level=thr, spacing=sp)
    va[:,0]+=x_min; va[:,1]+=x_min; va[:,2]+=z_min
    del Q_asm; gc.collect()
    print(f"  Asym raw: {len(fa):,}f")
    va,fa,na_ = decimate_mesh(va, fa, max_faces)
    print(f"  Asym dec: {len(fa):,}f")
    gc.collect()

    # Write .bin
    bin_name = os.path.splitext(os.path.basename(npz_path))[0] + '.bin'
    bin_path = os.path.join(os.path.dirname(npz_path) or '.', bin_name)
    nVS,nFS = len(vs),len(fs)
    nVA,nFA = len(va),len(fa)
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<8I', nVS,nFS,0,0,nVA,nFA,0,0))
        f.write(vs.astype(np.float32).tobytes())
        f.write(ns.astype(np.float32).tobytes())
        f.write(fs.astype(np.uint32).tobytes())
        f.write(va.astype(np.float32).tobytes())
        f.write(na_.astype(np.float32).tobytes())
        f.write(fa.astype(np.uint32).tobytes())
        f.write(np.array([Cp[0],Cp[1],Cp[2],Cm[0],Cm[1],Cm[2]], dtype=np.float32).tobytes())

    bin_size = os.path.getsize(bin_path) / 1e6
    del vs,fs,ns,va,fa,na_; gc.collect()

    # Gzip
    gz_path = bin_path + '.gz'
    with open(bin_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_size = os.path.getsize(gz_path) / 1e6

    print(f"  .bin: {bin_size:.1f} MB → .gz: {gz_size:.1f} MB")
    print(f"  Done ({time.time()-t0:.1f}s)")
    bin_files.append((bin_name, gz_path))

# ======================== GIT PUSH =======================================
if do_push and bin_files and repo_path:
    print(f"\nPushing to GitHub...")
    for bin_name, gz_path in bin_files:
        dest = os.path.join(repo_path, bin_name + '.gz')
        shutil.copy2(gz_path, dest)
    os.chdir(repo_path)
    os.system('git add *.bin.gz')
    os.system('git commit -m "Update bin.gz files"')
    os.system('git push')
else:
    print(f"\nFiles ready in: {os.getcwd()}")

print(f"\nSummary:")
for bin_name, gz_path in bin_files:
    print(f"  {bin_name}.gz  ({os.path.getsize(gz_path)/1e6:.1f} MB)")
