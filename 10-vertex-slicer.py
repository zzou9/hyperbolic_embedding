# Required packages:
# numpy==1.23.0
# matplotlib==3.6.0
# To install: pip install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ---------- geometry: triangle-plane intersection ----------
def _triangle_plane_intersection(verts, tri, n, d, eps=1e-9):
    p = verts[tri]
    s = p @ n - d
    if np.all(np.abs(s) < eps):
        return []
    edges = [(0,1),(1,2),(2,0)]
    pts = []
    for i,j in edges:
        si,sj = s[i],s[j]
        pi,pj = p[i],p[j]
        if abs(si) < eps and abs(sj) < eps:
            continue
        if abs(si) < eps:
            pts.append(pi); continue
        if abs(sj) < eps:
            pts.append(pj); continue
        if si*sj < 0.0:
            t = si / (si - sj)
            pts.append(pi + t*(pj - pi))
    return [(pts[0], pts[1])] if len(pts)==2 else []

def slice_mesh(vertices, faces, n, d, eps=1e-9):
    segs=[]
    for tri in faces:
        segs.extend(_triangle_plane_intersection(vertices, tri, n, d, eps))
    return segs

# ---------- stitching segments -> polylines ----------
def _quantize_point(p, tol): return tuple(np.round(p/tol).astype(int))

def assemble_polylines(segments, tol=1e-6):
    if not segments: return []
    point_to_segments, seg_ids = {}, []
    for k,(a,b) in enumerate(segments):
        qa, qb = _quantize_point(a,tol), _quantize_point(b,tol)
        point_to_segments.setdefault(qa,[]).append((k,0))
        point_to_segments.setdefault(qb,[]).append((k,1))
        seg_ids.append((a,b))
    used=set(); polys=[]
    endpoints=[q for q,lst in point_to_segments.items() if len(lst)==1]
    seeds=endpoints + [q for q,lst in point_to_segments.items() if len(lst)>1]

    for seed in seeds:
        for seg_index,end_flag in point_to_segments.get(seed,[]):
            if seg_index in used: continue
            a,b = seg_ids[seg_index]
            chain=[a,b] if end_flag==0 else [b,a]
            used.add(seg_index)
            def grow(cur):
                while True:
                    q=_quantize_point(cur,tol)
                    found=False
                    for si,ef in point_to_segments.get(q,[]):
                        if si in used: continue
                        pa,pb = seg_ids[si]
                        nxt = pb if ef==0 else pa
                        chain.append(nxt); used.add(si); cur=nxt; found=True; break
                    if not found: return cur
            grow(chain[-1])
            chain.reverse(); grow(chain[-1]); chain.reverse()
            # dedup
            cleaned=[chain[0]]
            for p in chain[1:]:
                if np.linalg.norm(np.asarray(p)-np.asarray(cleaned[-1]))>tol:
                    cleaned.append(p)
            if len(cleaned)>=2: polys.append(np.asarray(cleaned))
    # unique
    uniq, seen = [], set()
    for poly in polys:
        key=( _quantize_point(poly[0],tol), _quantize_point(poly[-1],tol), len(poly) )
        if key in seen: continue
        seen.add(key); uniq.append(poly)
    return uniq


def plot_slices(vertices, faces, a=None, b=None, c=None,
                show_mesh_outline=True, show_x=False, show_y=False, show_z=False,
                show_custom=True, slice_linewidth=1.2):
    V, F = vertices, faces
    xmin, ymin, zmin = V.min(axis=0); xmax, ymax, zmax = V.max(axis=0)
    if a is None: a = 0.5*(xmin+xmax)
    if b is None: b = 0.5*(ymin+ymax)
    if c is None: c = 0.5*(zmin+zmax)

    fig = plt.figure(figsize=(12.5, 8.0))

    # --- Layout: small left (normal), large right (slicer)
    gs = GridSpec(1, 2, width_ratios=[1, 5], wspace=0.15, left=0.07, right=0.96, top=0.92, bottom=0.3)
    axn = fig.add_subplot(gs[0, 0], projection='3d')  # small normal viewer (LEFT)
    ax  = fig.add_subplot(gs[0, 1], projection='3d')  # big slicer (RIGHT)

    # --- Big slicer (right)
    ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
    ax.set_xlim([xmin,xmax]); ax.set_ylim([ymin,ymax]); ax.set_zlim([zmin,zmax])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('10-Vertex 2-Torus Slicer — axis slices + free-orientation plane')

    mesh_coll = None
    if show_mesh_outline:
        tris = V[F]
        mesh_coll = Poly3DCollection(tris, facecolor=(0.85,0.85,0.85,0.08),
                                     edgecolor=(0.4,0.4,0.4,0.25), linewidth=0.3)
        ax.add_collection3d(mesh_coll)

    col_x = Line3DCollection([], linewidths=slice_linewidth, colors='tab:red')
    col_y = Line3DCollection([], linewidths=slice_linewidth, colors='tab:green')
    col_z = Line3DCollection([], linewidths=slice_linewidth, colors='tab:blue')
    col_c = Line3DCollection([], linewidths=slice_linewidth, colors='tab:purple')
    for coll in (col_x, col_y, col_z, col_c): ax.add_collection3d(coll)
    col_x.set_visible(show_x); col_y.set_visible(show_y); col_z.set_visible(show_z); col_c.set_visible(show_custom)

    # --- Small normal viewer (left)
    axn.set_title('Custom normal')
    u = np.linspace(0, np.pi, 22); v = np.linspace(0, 2*np.pi, 34)
    xs = np.outer(np.sin(u), np.cos(v)); ys = np.outer(np.sin(u), np.sin(v)); zs = np.outer(np.cos(u), np.ones_like(v))
    axn.plot_wireframe(xs, ys, zs, linewidth=0.4, alpha=0.6)
    normal_line, = axn.plot([0,0], [0,0], [0,0], lw=3, color='tab:purple')
    axn.set_xlim([-1,1]); axn.set_ylim([-1,1]); axn.set_zlim([-1,1])
    axn.set_box_aspect((1,1,1)); axn.set_xticks([]); axn.set_yticks([]); axn.set_zticks([]); axn.grid(False)

    # --- Controls (under plots)
    # Sliders row (a,b,c,lw,theta,phi,offset)
    ax_a  = plt.axes((0.15, 0.26, 0.72, 0.028))
    ax_b  = plt.axes((0.15, 0.22, 0.72, 0.028))
    ax_c  = plt.axes((0.15, 0.18, 0.72, 0.028))
    ax_lw = plt.axes((0.15, 0.14, 0.72, 0.028))
    ax_th = plt.axes((0.15, 0.10, 0.72, 0.028))
    ax_ph = plt.axes((0.15, 0.06, 0.72, 0.028))
    ax_of = plt.axes((0.15, 0.02, 0.72, 0.028))

    s_a  = Slider(ax_a,  'a (x=a)', xmin, xmax, valinit=a)
    s_b  = Slider(ax_b,  'b (y=b)', ymin, ymax, valinit=b)
    s_c  = Slider(ax_c,  'c (z=c)', zmin, zmax, valinit=c)
    s_lw = Slider(ax_lw, 'Line width', 0.2, 6.0, valinit=slice_linewidth)
    s_th = Slider(ax_th, 'Custom θ (0–π)',   0.0, np.pi,   valinit=np.pi/3)
    s_ph = Slider(ax_ph, 'Custom φ (0–2π)',  0.0, 2*np.pi, valinit=np.pi/4)
    s_of = Slider(ax_of, 'Custom offset (0–1)', 0.0, 1.0,  valinit=0.5)

    # Smaller checkbox panel (moved up slightly, narrower & shorter)
    ax_chk = plt.axes((0.05, 0.8, 0.18, 0.15))
    chk = CheckButtons(ax_chk,
        ['Show X-slice','Show Y-slice','Show Z-slice','Show Custom slice','Show mesh outline'],
        [show_x, show_y, show_z, show_custom, show_mesh_outline]
    )

    # Helpers
    def _normal():
        th, ph = float(s_th.val), float(s_ph.val)
        n = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], float)
        n /= max(np.linalg.norm(n), 1e-12)
        proj = V @ n
        dmin, dmax = float(np.min(proj)), float(np.max(proj))
        d = (1.0 - float(s_of.val))*dmin + float(s_of.val)*dmax
        return n, d

    # Update
    def update(_):
        axa, axb, axc = float(s_a.val), float(s_b.val), float(s_c.val)
        col_x.set_segments(assemble_polylines(slice_mesh(V,F,np.array([1,0,0],float), axa), 1e-6))
        col_y.set_segments(assemble_polylines(slice_mesh(V,F,np.array([0,1,0],float), axb), 1e-6))
        col_z.set_segments(assemble_polylines(slice_mesh(V,F,np.array([0,0,1],float), axc), 1e-6))

        n, d = _normal()
        col_c.set_segments(assemble_polylines(slice_mesh(V,F,n, d), 1e-6))

        lw = float(s_lw.val)
        for coll in (col_x, col_y, col_z, col_c): coll.set_linewidth(lw)

        # update normal viewer arrow
        normal_line.set_data_3d([0.0, n[0]], [0.0, n[1]], [0.0, n[2]])
        fig.canvas.draw_idle()

    def on_check(label):
        nonlocal mesh_coll
        if label == 'Show X-slice':        col_x.set_visible(not col_x.get_visible())
        elif label == 'Show Y-slice':      col_y.set_visible(not col_y.get_visible())
        elif label == 'Show Z-slice':      col_z.set_visible(not col_z.get_visible())
        elif label == 'Show Custom slice': col_c.set_visible(not col_c.get_visible())
        elif label == 'Show mesh outline':
            if mesh_coll is None:
                tris = V[F]
                mesh_coll = Poly3DCollection(
                    tris, facecolor=(0.85,0.85,0.85,0.08),
                    edgecolor=(0.4,0.4,0.4,0.25), linewidth=0.3
                )
                ax.add_collection3d(mesh_coll)
                mesh_coll.set_visible(True)
            else:
                mesh_coll.set_visible(not mesh_coll.get_visible())
        fig.canvas.draw_idle()

    for s in (s_a, s_b, s_c, s_lw, s_th, s_ph, s_of): s.on_changed(update)
    chk.on_clicked(on_check)
    update(None)
    plt.show()


# -------------------------
# Example: replace with your mesh
if __name__ == "__main__":
    # coordinates of the vertices
    V = np.array([
        [0.7315, 0.0202, 0.2868802278156344],
        [-0.316, 0.5792, -0.2252919753182151],
        [0.3426, -0.592, -0.22851917827575875],
        [-0.4323, -0.592, -0.2327286389494384],
        [-0.7303, 0.04, -0.22959077803009317],
        [0.1464, 0.6149, 0.13588682780065944],
        [-0.5154, 0.0395, 0.4610277738320659],
        [0.6649, -0.1156, -0.22651115997910957],
        [0.152, 0.2539, -0.23985732806740792],
        [-0.03, 0.0606, 0.6439645661413604]
    ])

    # indices of faces
    F = np.array([
        [0, 1, 7], [0, 2, 1], [0, 3, 6], [0, 4, 9], [0, 5, 3], [0, 6, 4],
        [0, 7, 8], [0, 8, 5], [0, 9, 2], [1, 2, 4], [1, 3, 7], [1, 4, 6],
        [1, 5, 8], [1, 6, 5], [1, 8, 3], [2, 3, 8], [2, 6, 3], [2, 7, 4],
        [2, 8, 7], [2, 9, 6], [3, 4, 7], [3, 5, 4], [4, 5, 9], [5, 6, 9]
    ])

    plot_slices(V, F, show_mesh_outline=True, show_custom=True, slice_linewidth=1.0)
