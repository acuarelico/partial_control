"""
quick_view_mccann.py — Quick 3D visualization of McCann-Yodzis safe sets
=========================================================================
Reads one or more .npz files, does downsampled marching cubes, exports a
self-contained HTML file with Three.js. Open in browser.

Usage:
  python quick_view_mccann.py mccann_yodzis_dt50_xi0.1_u0.07.npz
  python quick_view_mccann.py *.npz          (all in one viewer with selector)
  python quick_view_mccann.py                (auto-detect all mccann_yodzis*.npz)
"""

import numpy as np
import sys, os, json, time, glob
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

try:
    import pyfqmr
    HAS_FQMR = True
except ImportError:
    HAS_FQMR = False
    print("(pyfqmr not found — will skip decimation)")

def decimate(verts, faces, target):
    if not HAS_FQMR or len(faces) <= target:
        return verts, faces
    s = pyfqmr.Simplify()
    s.setMesh(verts.astype(np.float64), faces.astype(np.int32))
    s.simplify_mesh(target_count=target, aggressiveness=7, verbose=0)
    v, f, _ = s.getMesh()
    return v.astype(np.float32), f.astype(np.uint32)

def extract_mesh(bool_packed, NX, NY, NZ, h, x_min, y_min, z_min, ds, sigma, thr, max_faces, label):
    print(f"  {label}: unpacking...", end=' ', flush=True)
    vol = np.unpackbits(bool_packed)[:NX*NY*NZ].reshape((NX, NY, NZ)).astype(np.bool_)
    n_on = vol.sum()
    print(f"{n_on:,} voxels on")

    if n_on == 0:
        print(f"  {label}: EMPTY - skipping")
        return None, None

    print(f"  {label}: downsample ds={ds}...", end=' ', flush=True)
    vol_ds = vol[::ds, ::ds, ::ds].astype(np.float32)
    del vol
    print(f"shape {vol_ds.shape}")

    print(f"  {label}: smooth sigma={sigma}...", end=' ', flush=True)
    gaussian_filter(vol_ds, sigma=sigma, output=vol_ds)
    print("done")

    print(f"  {label}: marching cubes (thr={thr})...", end=' ', flush=True)
    sp = (h*ds, h*ds, h*ds)
    try:
        verts, faces, _, _ = marching_cubes(vol_ds, level=thr, spacing=sp)
    except ValueError:
        print("FAILED (no surface found)")
        return None, None
    del vol_ds
    verts[:, 0] += x_min
    verts[:, 1] += y_min
    verts[:, 2] += z_min
    print(f"{len(verts):,}v, {len(faces):,}f")

    if len(faces) > max_faces:
        print(f"  {label}: decimate -> {max_faces:,}...", end=' ', flush=True)
        verts, faces = decimate(verts, faces, max_faces)
        print(f"{len(verts):,}v, {len(faces):,}f")

    return verts.astype(np.float32), faces.astype(np.uint32)

def mesh_to_json(verts, faces):
    return {
        'v': [round(float(x), 4) for x in verts.flatten()],
        'f': faces.flatten().tolist(),
    }

def process_file(npz_path):
    """Process one .npz and return dict with meshes + metadata."""
    print(f"\n{'='*60}")
    print(f"  Processing: {os.path.basename(npz_path)}")
    print(f"{'='*60}")
    data = np.load(npz_path)

    NX = int(data['NX']); NY = int(data['NY']); NZ = int(data['NZ'])
    h = float(data['h'])
    x_min = float(data['x_min']); y_min = float(data['y_min']); z_min = float(data['z_min'])
    x_max = float(data['x_max']); y_max = float(data['y_max']); z_max = float(data['z_max'])
    xi0 = float(data['xi0']); u0 = float(data['u0_ctrl']); dt_map = float(data['dt_map'])
    N_safe = int(data['N_safe']); N_asymp = int(data['N_asymp'])

    print(f"  Grid: {NX}x{NY}x{NZ}, h={h:.6f}")
    print(f"  dt={dt_map}, xi0={xi0}, u0={u0}")
    print(f"  N_safe={N_safe:,}, N_asymp={N_asymp:,}")

    DS_SAFE = 2
    DS_ASYM = 1
    MAX_FACES = 500000

    vs, fs = extract_mesh(data['Q'], NX, NY, NZ, h, x_min, y_min, z_min,
                          DS_SAFE, 0.6, 0.35, MAX_FACES, "Safe")
    va, fa = extract_mesh(data['Q_asymp'], NX, NY, NZ, h, x_min, y_min, z_min,
                          DS_ASYM, 0.4, 0.35, MAX_FACES, "Asym")

    meshes = {}
    if vs is not None:
        meshes['safe'] = mesh_to_json(vs, fs)
    if va is not None:
        meshes['asym'] = mesh_to_json(va, fa)

    return {
        'meshes': meshes,
        'dt_map': dt_map, 'xi0': xi0, 'u0': u0,
        'NX': NX, 'NY': NY, 'NZ': NZ, 'h': h,
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_min': z_min, 'z_max': z_max,
        'N_safe': N_safe, 'N_asymp': N_asymp,
    }

def main():
    # Gather files
    if len(sys.argv) < 2:
        files = sorted(glob.glob('mccann_yodzis*.npz'))
        if not files:
            print("Usage: python quick_view_mccann.py <file.npz> [file2.npz ...]")
            sys.exit(1)
        print(f"Auto-detected {len(files)} file(s)")
    else:
        files = []
        for a in sys.argv[1:]:
            files.extend(sorted(glob.glob(a)))
        if not files:
            print("No matching .npz files found.")
            sys.exit(1)

    print(f"Files to process: {len(files)}")
    for f in files:
        print(f"  {f}")

    t0 = time.time()
    all_data = []
    for f in files:
        result = process_file(f)
        all_data.append(result)

    # Use first file for box bounds
    d0 = all_data[0]
    cx = (d0['x_min'] + d0['x_max']) / 2
    cy = (d0['y_min'] + d0['y_max']) / 2
    cz = (d0['z_min'] + d0['z_max']) / 2

    # Build datasets JSON: list of {label, meshes, info}
    datasets = []
    for i, d in enumerate(all_data):
        label = f"Δt={int(d['dt_map'])} ξ₀={d['xi0']} u₀={d['u0']}"
        datasets.append({
            'label': label,
            'meshes': d['meshes'],
            'info': {
                'dt': d['dt_map'], 'xi0': d['xi0'], 'u0': d['u0'],
                'N_safe': d['N_safe'], 'N_asymp': d['N_asymp'],
                'NX': d['NX'], 'NY': d['NY'], 'NZ': d['NZ'], 'h': d['h'],
            }
        })

    json_str = json.dumps(datasets)

    print(f"\nBuilding HTML... ({time.time()-t0:.1f}s so far)")

    html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>McCann-Yodzis Safe Sets</title>
<style>
  body { margin:0; overflow:hidden; background:#111; color:#ccc; font-family:monospace; }
  #info { position:absolute; top:8px; left:12px; font-size:12px; z-index:10;
          pointer-events:none; line-height:1.6; }
  #info span { color:#ff6b3d; }
  #controls { position:absolute; top:8px; right:12px; z-index:10; font-size:11px; }
  #controls label { margin-right:12px; cursor:pointer; }
  #selector { position:absolute; bottom:12px; left:12px; z-index:10; font-size:12px; }
  #selector select { font-size:13px; font-family:monospace; background:#222; color:#eee;
                     border:1px solid #555; padding:4px 8px; }
  #selector button { font-size:12px; font-family:monospace; background:#333; color:#eee;
                     border:1px solid #555; padding:4px 10px; margin-left:6px; cursor:pointer; }
  #selector button:hover { background:#555; }
  canvas { display:block; }
</style>
</head><body>
<div id="info"></div>
<div id="controls">
  <label><input type="checkbox" id="cb-safe" checked> Safe set</label>
  <label><input type="checkbox" id="cb-asym" checked> Asymptotic</label>
  <label><input type="checkbox" id="cb-box" checked> Box</label>
</div>
<div id="selector"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATASETS = """ + json_str + """;
const CX=""" + str(round(cx,4)) + """, CY=""" + str(round(cy,4)) + """, CZ=""" + str(round(cz,4)) + """;
const XMIN=""" + str(round(d0['x_min'],4)) + """, XMAX=""" + str(round(d0['x_max'],4)) + """;
const YMIN=""" + str(round(d0['y_min'],4)) + """, YMAX=""" + str(round(d0['y_max'],4)) + """;
const ZMIN=""" + str(round(d0['z_min'],4)) + """, ZMAX=""" + str(round(d0['z_max'],4)) + """;

// Scene — axes: Three.js x=R, y=P, z=-C (so P is vertical)
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
const rng = Math.max(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN);
camera.position.set(CX + rng*1.2, CZ + rng*0.8, -(CY - rng*0.8));
camera.up.set(0,1,0);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.setClearColor(0x111111);
document.body.appendChild(renderer.domElement);

// Target = center of box, mapped: (R, P, -C)
const target = new THREE.Vector3(CX, CZ, -CY);
let spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(target));
let dragging=false, px=0, py=0;
renderer.domElement.addEventListener('mousedown', e=>{dragging=true;px=e.clientX;py=e.clientY});
window.addEventListener('mousemove', e=>{
  if(!dragging)return;
  spherical.theta -= (e.clientX-px)*0.005;
  spherical.phi -= (e.clientY-py)*0.005;
  spherical.phi = Math.max(0.01, Math.min(Math.PI-0.01, spherical.phi));
  px=e.clientX;py=e.clientY;
});
window.addEventListener('mouseup', ()=>dragging=false);
renderer.domElement.addEventListener('wheel', e=>{
  spherical.radius *= e.deltaY>0?1.05:0.95;
  spherical.radius = Math.max(0.3, Math.min(20, spherical.radius));
},{passive:true});
// Touch
let lt=0;
renderer.domElement.addEventListener('touchstart',e=>{
  if(e.touches.length==1){dragging=true;px=e.touches[0].clientX;py=e.touches[0].clientY}
  else if(e.touches.length==2){var dx=e.touches[1].clientX-e.touches[0].clientX,dy=e.touches[1].clientY-e.touches[0].clientY;lt=Math.sqrt(dx*dx+dy*dy)}
  e.preventDefault();
},{passive:false});
renderer.domElement.addEventListener('touchmove',e=>{
  if(e.touches.length==1&&dragging){
    spherical.theta-=(e.touches[0].clientX-px)*0.005;
    spherical.phi-=(e.touches[0].clientY-py)*0.005;
    spherical.phi=Math.max(0.01,Math.min(Math.PI-0.01,spherical.phi));
    px=e.touches[0].clientX;py=e.touches[0].clientY;
  }else if(e.touches.length==2){
    var dx=e.touches[1].clientX-e.touches[0].clientX,dy=e.touches[1].clientY-e.touches[0].clientY;
    var d=Math.sqrt(dx*dx+dy*dy);if(lt>0)spherical.radius*=lt/d;lt=d;
    spherical.radius=Math.max(0.3,Math.min(20,spherical.radius));
  }
  e.preventDefault();
},{passive:false});
renderer.domElement.addEventListener('touchend',()=>{dragging=false;lt=0});

// Lights
scene.add(new THREE.AmbientLight(0x404040, 0.8));
const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
dl1.position.set(2, 3, 2); scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0x6688ff, 0.4);
dl2.position.set(-1, -1, -2); scene.add(dl2);

// Bounding box wireframe (mapped coords)
function makeBox() {
  const g = new THREE.BoxGeometry(XMAX-XMIN, ZMAX-ZMIN, YMAX-YMIN);
  const m = new THREE.EdgesGeometry(g);
  const box = new THREE.LineSegments(m, new THREE.LineBasicMaterial({color:0x444444}));
  box.position.set(CX, CZ, -CY);
  return box;
}
const boxWire = makeBox();
scene.add(boxWire);

// Axis labels using sprites
function makeLabel(text, pos, color) {
  const canvas = document.createElement('canvas');
  canvas.width=128; canvas.height=64;
  const ctx = canvas.getContext('2d');
  ctx.font='bold 36px monospace'; ctx.fillStyle=color;
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(text, 64, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({map:tex, depthTest:false});
  const sp = new THREE.Sprite(mat);
  sp.position.copy(pos);
  sp.scale.set(0.25, 0.125, 1);
  return sp;
}
scene.add(makeLabel('R', new THREE.Vector3(XMAX+0.08, CZ, -CY), '#ff4444'));
scene.add(makeLabel('C', new THREE.Vector3(CX, CZ, -(YMAX+0.08)), '#44ff44'));
scene.add(makeLabel('P', new THREE.Vector3(CX, ZMAX+0.05, -CY), '#4488ff'));

// Meshes
function makeMesh(d, color, opacity) {
  const g = new THREE.BufferGeometry();
  const v = new Float32Array(d.v.length);
  // Map (R,C,P) -> Three.js (x=R, y=P, z=-C)
  for(let i=0;i<d.v.length;i+=3){ v[i]=d.v[i]; v[i+1]=d.v[i+2]; v[i+2]=-d.v[i+1]; }
  g.setAttribute('position', new THREE.BufferAttribute(v, 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(d.f), 1));
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshPhongMaterial({
    color, transparent:opacity<1, opacity, side:THREE.DoubleSide,
    depthWrite:opacity>=0.9, shininess:30
  }));
}

let safeMesh=null, asymMesh=null;
let currentIdx = 0;

function loadDataset(idx) {
  // Remove old
  if(safeMesh){ scene.remove(safeMesh); safeMesh.geometry.dispose(); safeMesh=null; }
  if(asymMesh){ scene.remove(asymMesh); asymMesh.geometry.dispose(); asymMesh=null; }

  const ds = DATASETS[idx];
  if(ds.meshes.safe){
    safeMesh = makeMesh(ds.meshes.safe, 0x3d88ff, 0.15);
    safeMesh.visible = document.getElementById('cb-safe').checked;
    scene.add(safeMesh);
  }
  if(ds.meshes.asym){
    asymMesh = makeMesh(ds.meshes.asym, 0x3dff8a, 0.6);
    asymMesh.visible = document.getElementById('cb-asym').checked;
    scene.add(asymMesh);
  }

  // Update info
  const info = ds.info;
  document.getElementById('info').innerHTML =
    'McCann-Yodzis &mdash; K=0.99, &sigma;=0.07<br>' +
    '&Delta;t=<span>'+info.dt+'</span> &xi;&#8320;=<span>'+info.xi0+'</span> u&#8320;=<span>'+info.u0+'</span><br>' +
    'Grid: '+info.NX+'&times;'+info.NY+'&times;'+info.NZ+', h='+info.h.toFixed(6)+'<br>' +
    'Safe: <span>'+(info.N_safe).toLocaleString()+'</span> &nbsp; Asym: <span>'+(info.N_asymp).toLocaleString()+'</span>';

  currentIdx = idx;
}

// Build selector
const selDiv = document.getElementById('selector');
if(DATASETS.length > 1) {
  const sel = document.createElement('select');
  sel.id = 'ds-select';
  DATASETS.forEach((ds, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.text = ds.label;
    sel.appendChild(opt);
  });
  sel.onchange = () => loadDataset(parseInt(sel.value));
  selDiv.appendChild(sel);

  // Prev/Next buttons
  const btnP = document.createElement('button');
  btnP.textContent = '◀ Prev';
  btnP.onclick = () => { sel.value = Math.max(0, currentIdx-1); loadDataset(parseInt(sel.value)); };
  selDiv.appendChild(btnP);
  const btnN = document.createElement('button');
  btnN.textContent = 'Next ▶';
  btnN.onclick = () => { sel.value = Math.min(DATASETS.length-1, currentIdx+1); loadDataset(parseInt(sel.value)); };
  selDiv.appendChild(btnN);
}

// Checkboxes
document.getElementById('cb-safe').onchange = e => { if(safeMesh) safeMesh.visible=e.target.checked; };
document.getElementById('cb-asym').onchange = e => { if(asymMesh) asymMesh.visible=e.target.checked; };
document.getElementById('cb-box').onchange = e => { boxWire.visible=e.target.checked; };

window.addEventListener('resize', () => {
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Load first dataset
loadDataset(0);

function animate() {
  requestAnimationFrame(animate);
  camera.position.setFromSpherical(spherical).add(target);
  camera.lookAt(target);
  renderer.render(scene, camera);
}
animate();
</script>
</body></html>"""

    if len(files) == 1:
        out_name = os.path.splitext(os.path.basename(files[0]))[0] + '_view.html'
    else:
        out_name = 'mccann_yodzis_view.html'

    with open(out_name, 'w', encoding='utf-8') as f:
        f.write(html)
    out_sz = os.path.getsize(out_name) / 1e6
    print(f"\n  Output: {out_name} ({out_sz:.1f} MB)")
    print(f"  Open in browser to view.")
    print(f"  Total time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
