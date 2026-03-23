"""
publish.py — Convert .npz → .bin (pyfqmr) → .gz → push to GitHub
==================================================================
One-stop script: generates quality meshes, compresses, and uploads.

Usage:
    python publish.py                        (all .npz in folder)
    python publish.py batch_lorenz_*.npz     (specific files)
    python publish.py --no-push *.npz        (skip git push)

Options:
    --faces N       Max faces (default: 150000)
    --downsample D  Grid downsample (default: auto)
    --sigma S       Gaussian smooth (default: 2.0)
    --no-push       Skip git push (just generate files)
    --repo PATH     Path to git repo (default: auto-detect)

Install:  pip install pyfqmr numpy scipy scikit-image
Git:      Install from https://git-scm.com/download/win
"""

import numpy as np
import os, struct, sys, time, glob, gzip, shutil

from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

try:
    import pyfqmr
except ImportError:
    print("ERROR: pip install pyfqmr"); sys.exit(1)

# ======================== MESH HELPERS ===================================
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
smooth_sigma = 2.0
do_push = True
repo_path = None

i = 1
while i < len(sys.argv):
    a = sys.argv[i]
    if a == '--faces':    max_faces = int(sys.argv[i+1]); i += 2
    elif a == '--downsample': ds_override = int(sys.argv[i+1]); i += 2
    elif a == '--sigma':  smooth_sigma = float(sys.argv[i+1]); i += 2
    elif a == '--no-push': do_push = False; i += 1
    elif a == '--repo':   repo_path = sys.argv[i+1]; i += 2
    else:
        files.extend(glob.glob(a)); i += 1

# If no files given, find all .npz in current dir
if not files:
    files = sorted(glob.glob('*.npz'))
    if not files:
        print("No .npz files found. Usage: python publish.py [files.npz]")
        sys.exit(1)

files = [f for f in files if f.endswith('.npz')]
print(f"\nProcessing {len(files)} file(s)...\n")

# ======================== FIND GIT REPO ==================================
if do_push and not repo_path:
    # Try to find the repo
    candidates = [
        os.path.expanduser('~/partial_control'),
        os.path.expanduser('~/Desktop/partial_control'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'partial_control'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, '.git')):
            repo_path = c
            break

if do_push and not repo_path:
    print("Git repo not found. Options:")
    print("  1. Clone:  git clone https://github.com/acuarelico/partial_control.git")
    print("  2. Specify: python publish.py --repo C:\\path\\to\\partial_control *.npz")
    print("  3. Skip:   python publish.py --no-push *.npz")
    print()
    repo_input = input("Path to repo (or Enter to skip push): ").strip()
    if repo_input:
        repo_path = repo_input
    else:
        do_push = False

if do_push:
    print(f"Git repo: {repo_path}")

# ======================== PROCESS EACH ===================================
bin_files = []

for npz_path in files:
    if not os.path.exists(npz_path):
        print(f"Not found: {npz_path}"); continue

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

    # Adaptive parameters based on structure thickness
    # u0 determines minimum feature size: smaller u0 → thinner sheets
    u0_val = float(data.get('u0_ctrl', 1.0))
    r_u0_vox = u0_val / h  # feature size in voxels

    # Downsample: must preserve sheets. ds < feature_size / 3
    if ds_override:
        ds_safe = ds_override
        ds_asym = ds_override
    else:
        ds_safe = max(1, min(int(r_u0_vox / 3), N // 200))
        ds_asym = max(1, min(int(r_u0_vox / 4), N // 400))

    # Smoothing: high enough to remove voxel jaggies (~1.0-1.2),
    # but less than feature_in_ds_coords / 3 to avoid merging sheets
    feat_safe = r_u0_vox / ds_safe
    feat_asym = r_u0_vox / ds_asym
    sigma_safe = min(1.5, feat_safe / 3)
    sigma_asym = min(1.2, feat_asym / 3)

    print(f"  u₀={u0_val}, r_u0={r_u0_vox:.1f} vox")
    print(f"  Safe:  ds={ds_safe} ({N//ds_safe}³), σ={sigma_safe:.2f}, feat={feat_safe:.1f}vox")
    print(f"  Asym:  ds={ds_asym} ({N//ds_asym}³), σ={sigma_asym:.2f}, feat={feat_asym:.1f}vox")

    # === SAFE SET isosurface ===
    Q_sm = gaussian_filter(Q[::ds_safe,::ds_safe,::ds_safe].astype(np.float32), sigma=sigma_safe)
    sp_safe = (h*ds_safe, h*ds_safe, h*ds_safe)
    vs,fs,ns,_ = marching_cubes(Q_sm, level=0.3, spacing=sp_safe)
    vs[:,0]+=x_min; vs[:,1]+=x_min; vs[:,2]+=z_min
    del Q_sm
    print(f"  Safe raw: {len(fs):,}f")

    # === ASYMPTOTIC isosurface ===
    Q_asm = gaussian_filter(Q_asymp[::ds_asym,::ds_asym,::ds_asym].astype(np.float32), sigma=sigma_asym)
    sp_asym = (h*ds_asym, h*ds_asym, h*ds_asym)
    va,fa,na_,_ = marching_cubes(Q_asm, level=0.35, spacing=sp_asym)
    va[:,0]+=x_min; va[:,1]+=x_min; va[:,2]+=z_min
    del Q_asm, Q, Q_asymp
    print(f"  Asym raw: {len(fa):,}f")

    vs,fs,ns = decimate_mesh(vs, fs, max_faces)
    va,fa,na_ = decimate_mesh(va, fa, max_faces)  # same cap for both
    print(f"  Decimated: safe={len(fs):,}f, asym={len(fa):,}f")

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

    # Gzip
    gz_path = bin_path + '.gz'
    with open(bin_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_size = os.path.getsize(gz_path) / 1e6

    print(f"  .bin: {bin_size:.1f} MB → .gz: {gz_size:.1f} MB ({gz_size/bin_size*100:.0f}%)")
    print(f"  Done ({time.time()-t0:.1f}s)")
    bin_files.append((bin_name, gz_path))

# ======================== GIT PUSH =======================================
if do_push and bin_files:
    print(f"\n{'='*50}")
    print(f"  Pushing {len(bin_files)} file(s) to GitHub...")
    print(f"{'='*50}")

    for bin_name, gz_path in bin_files:
        dest = os.path.join(repo_path, bin_name + '.gz')
        shutil.copy2(gz_path, dest)
        print(f"  Copied: {bin_name}.gz")

    os.chdir(repo_path)
    os.system('git add *.gz')
    os.system('git commit -m "Update compressed .bin.gz files"')
    os.system('git push')
    print("\nPushed to GitHub!")
else:
    print(f"\nFiles ready in: {os.getcwd()}")
    if not do_push:
        print("Upload .bin.gz files to GitHub manually.")

print(f"\nSummary:")
for bin_name, gz_path in bin_files:
    print(f"  {bin_name}.gz  ({os.path.getsize(gz_path)/1e6:.1f} MB)")
