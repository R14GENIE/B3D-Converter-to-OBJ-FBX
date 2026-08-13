#!/usr/bin/env python3
"""
b3d_convert.py -- Blitz3D (.b3d) -> Wavefront .obj (+ .mtl) and/or binary .fbx

Pure standard library. No Blender, no FBX SDK, no assimp.

Usage:
    python b3d_convert.py chair.b3d                      # writes chair.obj + chair.mtl
    python b3d_convert.py chair.b3d --format fbx         # writes chair.fbx
    python b3d_convert.py chair.b3d --format both -o out/chair
    python b3d_convert.py chair.b3d -d exported          # writes exported/chair.obj + .mtl
    python b3d_convert.py Props -d exported -f fbx       # whole folder in one go
    python b3d_convert.py Props -r -p -d exported -f fbx # ...including subfolders, tree kept
    python b3d_convert.py chair.b3d --info               # dump scene tree, convert nothing

Notes on conventions:
  * Blitz3D is left-handed, Y-up.  OBJ/FBX are right-handed, Y-up.  By default the
    Z axis is negated and triangle winding reversed.  Disable with --keep-handedness.
  * Blitz3D stores UVs DirectX-style (V grows downward).  V is flipped by default.
    Disable with --keep-uv.
  * Node transforms are baked into world space.  The output is a flat list of meshes.
"""

import argparse
import glob
import math
import os
import shutil
import struct
import sys
import zlib

# ---------------------------------------------------------------------------
# .b3d chunk reader
# ---------------------------------------------------------------------------


class Reader:
    """Little-endian cursor over a bytes object."""

    def __init__(self, data, pos=0, end=None):
        self.d = data
        self.p = pos
        self.end = len(data) if end is None else end

    def eof(self):
        return self.p >= self.end

    def raw(self, n):
        if self.p + n > self.end:
            raise EOFError("read past end of chunk")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def i32(self):
        return struct.unpack_from("<i", self.d, self._adv(4))[0]

    def f32(self):
        return struct.unpack_from("<f", self.d, self._adv(4))[0]

    def f32s(self, n):
        return list(struct.unpack_from("<%df" % n, self.d, self._adv(4 * n)))

    def i32s(self, n):
        return list(struct.unpack_from("<%di" % n, self.d, self._adv(4 * n)))

    def _adv(self, n):
        if self.p + n > self.end:
            raise EOFError("read past end of chunk")
        p = self.p
        self.p += n
        return p

    def string(self):
        """NUL-terminated string."""
        i = self.d.index(b"\x00", self.p, self.end)
        s = self.d[self.p:i]
        self.p = i + 1
        return s.decode("latin-1")

    def chunk(self):
        """Read one child chunk header, return (tag, Reader over its body)."""
        tag = self.raw(4).decode("ascii", "replace")
        size = self.i32()
        start = self.p
        if start + size > self.end:
            raise EOFError("chunk %r claims %d bytes, only %d left" % (tag, size, self.end - start))
        self.p = start + size
        return tag, Reader(self.d, start, start + size)


# ---------------------------------------------------------------------------
# scene model
# ---------------------------------------------------------------------------


class Texture:
    def __init__(self, file, flags, blend, pos, scale, rotation):
        self.file = file
        self.flags = flags
        self.blend = blend
        self.pos = pos
        self.scale = scale
        self.rotation = rotation


class Brush:
    def __init__(self, name, color, shininess, blend, fx, texture_ids):
        self.name = name or "brush"
        self.color = color              # r, g, b, a  (0..1)
        self.shininess = shininess
        self.blend = blend
        self.fx = fx
        self.texture_ids = texture_ids  # indices into scene.textures, -1 = none


class Surface:
    """A run of triangles sharing one brush."""

    def __init__(self, brush_id):
        self.brush_id = brush_id
        self.tris = []                  # list of (a, b, c) vertex indices


class Mesh:
    def __init__(self, name):
        self.name = name
        self.pos = []                   # [(x, y, z)]
        self.normal = []                # [(x, y, z)] or empty
        self.color = []                 # [(r, g, b, a)] or empty
        self.uv = []                    # [(u, v)] or empty
        self.surfaces = []


class Node:
    def __init__(self, name, position, scale, rotation):
        self.name = name
        self.position = position        # (x, y, z)
        self.scale = scale              # (x, y, z)
        self.rotation = rotation        # quaternion (w, x, y, z)
        self.mesh = None
        self.children = []


class Scene:
    def __init__(self):
        self.version = 0
        self.textures = []
        self.brushes = []
        self.root_nodes = []


# ---------------------------------------------------------------------------
# .b3d parsing
# ---------------------------------------------------------------------------


def parse_texs(r):
    out = []
    while not r.eof():
        file = r.string()
        flags = r.i32()
        blend = r.i32()
        x_pos, y_pos, x_scale, y_scale, rotation = r.f32s(5)
        out.append(Texture(file, flags, blend, (x_pos, y_pos), (x_scale, y_scale), rotation))
    return out


def parse_brus(r):
    n_texs = r.i32()
    out = []
    while not r.eof():
        name = r.string()
        color = tuple(r.f32s(4))
        shininess = r.f32()
        blend = r.i32()
        fx = r.i32()
        tex_ids = r.i32s(n_texs) if n_texs else []
        out.append(Brush(name, color, shininess, blend, fx, tex_ids))
    return out


def parse_vrts(r, mesh):
    flags = r.i32()
    tex_coord_sets = r.i32()
    tex_coord_set_size = r.i32()
    has_normal = bool(flags & 1)
    has_color = bool(flags & 2)

    while not r.eof():
        mesh.pos.append(tuple(r.f32s(3)))
        if has_normal:
            mesh.normal.append(tuple(r.f32s(3)))
        if has_color:
            mesh.color.append(tuple(r.f32s(4)))
        uv = None
        for s in range(tex_coord_sets):
            coords = r.f32s(tex_coord_set_size)
            if s == 0:
                uv = (coords[0], coords[1]) if len(coords) >= 2 else (coords[0], 0.0)
        if uv is not None:
            mesh.uv.append(uv)


def parse_tris(r):
    surf = Surface(r.i32())
    n = (r.end - r.p) // 12
    if n:
        flat = r.i32s(3 * n)
        surf.tris = [tuple(flat[i:i + 3]) for i in range(0, 3 * n, 3)]
    return surf


def parse_mesh(r, name):
    mesh = Mesh(name)
    mesh.brush_id = r.i32()
    while not r.eof():
        tag, sub = r.chunk()
        if tag == "VRTS":
            parse_vrts(sub, mesh)
        elif tag == "TRIS":
            surf = parse_tris(sub)
            if surf.brush_id < 0:
                surf.brush_id = mesh.brush_id
            mesh.surfaces.append(surf)
    return mesh


def parse_node(r):
    name = r.string()
    position = tuple(r.f32s(3))
    scale = tuple(r.f32s(3))
    rot = r.f32s(4)                     # w, x, y, z
    node = Node(name, position, scale, tuple(rot))
    while not r.eof():
        tag, sub = r.chunk()
        if tag == "MESH":
            node.mesh = parse_mesh(sub, name or "mesh")
        elif tag == "NODE":
            node.children.append(parse_node(sub))
        # BONE / KEYS / ANIM are skipped: skinning and animation are not exported
    return node


def parse_b3d(data):
    r = Reader(data)
    tag = r.raw(4)
    if tag != b"BB3D":
        raise ValueError("not a Blitz3D file (expected 'BB3D', got %r)" % tag)
    size = r.i32()
    body = Reader(data, r.p, min(r.p + size, len(data)))

    scene = Scene()
    scene.version = body.i32()
    while not body.eof():
        tag, sub = body.chunk()
        if tag == "TEXS":
            scene.textures.extend(parse_texs(sub))
        elif tag == "BRUS":
            scene.brushes.extend(parse_brus(sub))
        elif tag == "NODE":
            scene.root_nodes.append(parse_node(sub))
    return scene


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def mat_identity():
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def mat_mul(a, b):
    out = [0.0] * 16
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = (a[i * 4 + 0] * b[0 * 4 + j] +
                              a[i * 4 + 1] * b[1 * 4 + j] +
                              a[i * 4 + 2] * b[2 * 4 + j] +
                              a[i * 4 + 3] * b[3 * 4 + j])
    return out


def node_matrix(node):
    """T * R * S from a b3d node's position / quaternion / scale."""
    w, x, y, z = node.rotation
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-12:
        w, x, y, z = w / n, x / n, y / n, z / n
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0

    r = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
         2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]

    sx, sy, sz = node.scale
    px, py, pz = node.position
    return [r[0] * sx, r[1] * sy, r[2] * sz, px,
            r[3] * sx, r[4] * sy, r[5] * sz, py,
            r[6] * sx, r[7] * sy, r[8] * sz, pz,
            0.0, 0.0, 0.0, 1.0]


def xform_point(m, p):
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11])


def xform_dir(m, v):
    x, y, z = v
    return (m[0] * x + m[1] * y + m[2] * z,
            m[4] * x + m[5] * y + m[6] * z,
            m[8] * x + m[9] * y + m[10] * z)


def flatten(scene, scale=1.0, flip_handedness=True, flip_uv=True):
    """Walk the node tree, bake world transforms, return a flat list of Mesh."""
    out = []
    seen_names = {}

    def unique(name):
        base = name or "mesh"
        k = seen_names.get(base, 0)
        seen_names[base] = k + 1
        return base if k == 0 else "%s_%d" % (base, k)

    def walk(node, parent_m):
        m = mat_mul(parent_m, node_matrix(node))
        if node.mesh is not None:
            src = node.mesh
            dst = Mesh(unique(node.name or src.name))
            for p in src.pos:
                x, y, z = xform_point(m, p)
                x *= scale
                y *= scale
                z *= scale
                dst.pos.append((x, y, -z) if flip_handedness else (x, y, z))
            for nv in src.normal:
                nx, ny, nz = xform_dir(m, nv)
                ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                if ln > 1e-12:
                    nx, ny, nz = nx / ln, ny / ln, nz / ln
                dst.normal.append((nx, ny, -nz) if flip_handedness else (nx, ny, nz))
            for u, v in src.uv:
                dst.uv.append((u, 1.0 - v) if flip_uv else (u, v))
            dst.color = list(src.color)
            for s in src.surfaces:
                ns = Surface(s.brush_id)
                ns.tris = [(a, c, b) for (a, b, c) in s.tris] if flip_handedness else list(s.tris)
                if ns.tris:
                    dst.surfaces.append(ns)
            if dst.surfaces:
                out.append(dst)
        for child in node.children:
            walk(child, m)

    for root in scene.root_nodes:
        walk(root, mat_identity())
    return out


# ---------------------------------------------------------------------------
# material naming shared by both exporters
# ---------------------------------------------------------------------------


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds", ".gif", ".tif", ".tiff")


class TextureResolver:
    """Maps the texture names stored in a .b3d onto files that actually exist.

    Blitz3D files in the wild reference textures in four different ways:
      * relative to the model                 -> "metal.jpg"
      * relative to the game root             -> "GFX\\Items\\gas_mask.png"
      * absolute, from the author's machine   -> "C:\\Users\\bob\\...\\radio.png"
      * bare name, file living somewhere else -> "dirtymetal.jpg"

    Every image under the search roots is indexed by each suffix of its relative
    path, so a reference is matched by the longest path tail that exists on disk.
    """

    def __init__(self, roots, enabled=True, copy_dir=None):
        self.enabled = enabled
        self.copy_dir = copy_dir
        self.by_suffix = {}
        self.by_stem = {}
        self.resolved = 0
        self.unresolved = []            # [(b3d path, reference)]
        self._copied = {}               # source abs path -> name inside copy_dir
        self._copy_names = {}           # taken name -> source abs path
        self._cache = {}                # (ref, model dir) -> resolved path or None
        if enabled:
            for root in roots:
                self._scan(root)

    def _scan(self, root):
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            return
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace("\\", "/").lower()
                parts = rel.split("/")
                for i in range(len(parts)):
                    self.by_suffix.setdefault("/".join(parts[i:]), []).append(full)
                self.by_stem.setdefault(os.path.splitext(fn)[0].lower(), []).append(full)

    @staticmethod
    def _closest(candidates, model_dir):
        """Prefer the candidate sharing the longest directory prefix with the model."""
        want = os.path.abspath(model_dir).replace("\\", "/").lower().split("/")

        def score(p):
            have = os.path.dirname(os.path.abspath(p)).replace("\\", "/").lower().split("/")
            n = 0
            for a, b in zip(want, have):
                if a != b:
                    break
                n += 1
            return (-n, len(have), p)

        return sorted(candidates, key=score)[0]

    def resolve(self, ref, source_path):
        """Return an absolute path to the texture, or None."""
        if not ref or not ref.strip():
            return None
        model_dir = os.path.dirname(os.path.abspath(source_path))
        key = (ref, os.path.normcase(model_dir))
        if key in self._cache:          # keeps --format both from counting twice
            return self._cache[key]
        return self._cache.setdefault(key, self._resolve_uncached(ref, model_dir, source_path))

    def _resolve_uncached(self, ref, model_dir, source_path):
        norm = ref.replace("\\", "/").strip()

        direct = norm if os.path.isabs(norm) else os.path.join(model_dir, norm)
        if os.path.isfile(direct):
            self.resolved += 1
            return os.path.abspath(direct)

        if self.enabled:
            parts = [p for p in norm.lower().split("/") if p not in ("", ".")]
            for i in range(len(parts)):
                hits = self.by_suffix.get("/".join(parts[i:]))
                if hits:
                    self.resolved += 1
                    return self._closest(hits, model_dir)
            # same name, different extension (png <-> jpg conversions are common)
            hits = self.by_stem.get(os.path.splitext(parts[-1])[0]) if parts else None
            if hits:
                self.resolved += 1
                return self._closest(hits, model_dir)

        self.unresolved.append((source_path, ref))
        return None

    def gather(self, abs_path, out_dir):
        """Copy a texture into out_dir/copy_dir, returning its name there."""
        key = os.path.normcase(abs_path)
        if key in self._copied:
            return self._copied[key]

        name = os.path.basename(abs_path)
        stem, ext = os.path.splitext(name)
        n = 1
        while name.lower() in self._copy_names and \
                self._copy_names[name.lower()] != key:
            name = "%s_%d%s" % (stem, n, ext)
            n += 1
        self._copy_names[name.lower()] = key
        self._copied[key] = name

        dest_dir = os.path.join(out_dir, self.copy_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if not os.path.isfile(dest) or os.path.getsize(dest) != os.path.getsize(abs_path):
            shutil.copyfile(abs_path, dest)
        return name

    def link(self, ref, source_path, out_dir):
        """Resolve `ref` and return the string to write into the .mtl / .fbx.

        Returns (link_text, absolute_path_or_None).
        """
        abs_path = self.resolve(ref, source_path)
        if abs_path is None:
            return (ref.replace("\\", "/") if ref else None), None

        if self.copy_dir:
            name = self.gather(abs_path, out_dir)
            return "%s/%s" % (self.copy_dir.replace("\\", "/"), name), abs_path

        try:
            rel = os.path.relpath(abs_path, os.path.abspath(out_dir))
        except ValueError:              # different drive on Windows
            return abs_path.replace("\\", "/"), abs_path
        return rel.replace("\\", "/"), abs_path


def material_name(scene, brush_id):
    if 0 <= brush_id < len(scene.brushes):
        b = scene.brushes[brush_id]
        return sanitize("%s_%d" % (b.name or "brush", brush_id))
    return "default_%d" % (brush_id if brush_id >= 0 else 0)


def brush_texture(scene, brush_id):
    if not (0 <= brush_id < len(scene.brushes)):
        return None
    for tid in scene.brushes[brush_id].texture_ids:
        if 0 <= tid < len(scene.textures):
            f = scene.textures[tid].file
            if f and f.strip():
                return f
    return None


def sanitize(name):
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name) or "unnamed"


def used_brush_ids(meshes):
    ids = []
    for m in meshes:
        for s in m.surfaces:
            if s.brush_id not in ids:
                ids.append(s.brush_id)
    return ids


# ---------------------------------------------------------------------------
# OBJ / MTL export
# ---------------------------------------------------------------------------


def write_obj(scene, meshes, obj_path, resolver=None, source_path=None):
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# converted from Blitz3D .b3d by b3d_convert.py\n")
        f.write("mtllib %s\n" % os.path.basename(mtl_path))

        v_base = vt_base = vn_base = 1
        for mesh in meshes:
            has_uv = len(mesh.uv) == len(mesh.pos)
            has_n = len(mesh.normal) == len(mesh.pos)

            f.write("\no %s\n" % sanitize(mesh.name))
            for x, y, z in mesh.pos:
                f.write("v %.6f %.6f %.6f\n" % (x, y, z))
            if has_uv:
                for u, v in mesh.uv:
                    f.write("vt %.6f %.6f\n" % (u, v))
            if has_n:
                for nx, ny, nz in mesh.normal:
                    f.write("vn %.6f %.6f %.6f\n" % (nx, ny, nz))

            for s in mesh.surfaces:
                f.write("usemtl %s\n" % material_name(scene, s.brush_id))
                for tri in s.tris:
                    parts = []
                    for i in tri:
                        vi = v_base + i
                        if has_uv and has_n:
                            parts.append("%d/%d/%d" % (vi, vt_base + i, vn_base + i))
                        elif has_uv:
                            parts.append("%d/%d" % (vi, vt_base + i))
                        elif has_n:
                            parts.append("%d//%d" % (vi, vn_base + i))
                        else:
                            parts.append("%d" % vi)
                    f.write("f %s\n" % " ".join(parts))

            v_base += len(mesh.pos)
            if has_uv:
                vt_base += len(mesh.uv)
            if has_n:
                vn_base += len(mesh.normal)

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("# converted from Blitz3D .b3d by b3d_convert.py\n")
        for bid in used_brush_ids(meshes):
            b = scene.brushes[bid] if 0 <= bid < len(scene.brushes) else None
            r, g, bl, a = b.color if b else (1.0, 1.0, 1.0, 1.0)
            shin = b.shininess if b else 0.0
            f.write("\nnewmtl %s\n" % material_name(scene, bid))
            f.write("Kd %.6f %.6f %.6f\n" % (r, g, bl))
            f.write("Ka 0.000000 0.000000 0.000000\n")
            f.write("Ks %.6f %.6f %.6f\n" % (shin, shin, shin))
            f.write("Ns %.6f\n" % (shin * 128.0))
            f.write("d %.6f\n" % a)
            f.write("illum 2\n")
            tex = brush_texture(scene, bid)
            if not tex:
                continue
            if resolver is not None:
                link, found = resolver.link(tex, source_path, os.path.dirname(os.path.abspath(obj_path)))
                if link is None:
                    continue
                if found is None:
                    f.write("# unresolved texture reference, path left as authored\n")
                f.write("map_Kd %s\n" % link)
            else:
                f.write("map_Kd %s\n" % tex.replace("\\", "/"))
    return mtl_path


# ---------------------------------------------------------------------------
# binary FBX 7400 export
# ---------------------------------------------------------------------------

_FBX_VERSION = 7400
_FOOTER_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOTER_TAIL = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_SENTINEL = b"\x00" * 13


class FbxNode:
    """One FBX node record: a name, a property list, and nested nodes."""

    def __init__(self, name, *props):
        self.name = name.encode("ascii")
        self.props = list(props)
        self.children = []

    def add(self, name, *props):
        n = FbxNode(name, *props)
        self.children.append(n)
        return n

    # -- property encoders ---------------------------------------------------
    @staticmethod
    def _prop_bytes(p):
        kind, val = p
        if kind == "I":
            return b"I" + struct.pack("<i", val)
        if kind == "L":
            return b"L" + struct.pack("<q", val)
        if kind == "D":
            return b"D" + struct.pack("<d", val)
        if kind == "F":
            return b"F" + struct.pack("<f", val)
        if kind == "C":
            return b"C" + struct.pack("<B", 1 if val else 0)
        if kind == "S":
            b = val.encode("utf-8") if isinstance(val, str) else val
            return b"S" + struct.pack("<I", len(b)) + b
        if kind == "R":
            return b"R" + struct.pack("<I", len(val)) + val
        if kind in ("d", "i", "l", "f", "b"):
            fmt = {"d": "<%dd", "i": "<%di", "l": "<%dq", "f": "<%df", "b": "<%dB"}[kind]
            raw = struct.pack(fmt % len(val), *val)
            comp = zlib.compress(raw)
            if len(comp) < len(raw):
                return (kind.encode() + struct.pack("<III", len(val), 1, len(comp)) + comp)
            return (kind.encode() + struct.pack("<III", len(val), 0, len(raw)) + raw)
        raise ValueError("unknown FBX property kind %r" % kind)

    def encode(self, offset):
        """Serialize this record; `offset` is its absolute start in the file."""
        props = b"".join(self._prop_bytes(p) for p in self.props)
        header_len = 4 + 4 + 4 + 1 + len(self.name)
        body_start = offset + header_len + len(props)

        children = b""
        pos = body_start
        for c in self.children:
            enc = c.encode(pos)
            children += enc
            pos += len(enc)
        if self.children:
            children += _SENTINEL
            pos += len(_SENTINEL)

        end_offset = pos
        return (struct.pack("<III", end_offset, len(self.props), len(props)) +
                struct.pack("<B", len(self.name)) + self.name + props + children)


def _p70(parent, name, ptype, subtype, flags, *values):
    props = [("S", name), ("S", ptype), ("S", subtype), ("S", flags)]
    props.extend(values)
    parent.children.append(FbxNode("P", *props))


def _fbx_obj_name(name, cls):
    """Binary FBX object names are 'name\\x00\\x01Class'."""
    return name.encode("utf-8") + b"\x00\x01" + cls.encode("ascii")


def write_fbx(scene, meshes, fbx_path, source_path, resolver=None):
    ids = [1000000]

    def new_id():
        ids[0] += 1
        return ids[0]

    out_dir = os.path.dirname(os.path.abspath(fbx_path))
    brush_ids = used_brush_ids(meshes)
    mat_ids = {bid: new_id() for bid in brush_ids}

    # resolve textures up front: only brushes with a usable image get Texture ids,
    # which keeps the Definitions counts honest
    tex_links = {}
    for bid in brush_ids:
        tex = brush_texture(scene, bid)
        if not tex:
            continue
        if resolver is not None:
            rel, found = resolver.link(tex, source_path, out_dir)
            if rel is None:
                continue
            absolute = found or os.path.join(out_dir, rel)
        else:
            rel = tex.replace("\\", "/")
            absolute = os.path.join(os.path.dirname(os.path.abspath(source_path)), rel)
        tex_links[bid] = (rel, absolute)

    tex_ids = {}
    vid_ids = {}
    for bid in tex_links:
        tex_ids[bid] = new_id()
        vid_ids[bid] = new_id()

    root = FbxNode("__root__")   # container, never written itself

    # --- header ------------------------------------------------------------
    hdr = FbxNode("FBXHeaderExtension")
    hdr.add("FBXHeaderVersion", ("I", 1003))
    hdr.add("FBXVersion", ("I", _FBX_VERSION))
    hdr.add("EncryptionType", ("I", 0))
    ct = hdr.add("CreationTimeStamp")
    ct.add("Version", ("I", 1000))
    for k, v in (("Year", 2000), ("Month", 1), ("Day", 1), ("Hour", 0),
                 ("Minute", 0), ("Second", 0), ("Millisecond", 0)):
        ct.add(k, ("I", v))
    hdr.add("Creator", ("S", "b3d_convert.py"))
    root.children.append(hdr)

    root.children.append(FbxNode("Creator", ("S", "b3d_convert.py")))

    # --- global settings ---------------------------------------------------
    gs = FbxNode("GlobalSettings")
    gs.add("Version", ("I", 1000))
    p = gs.add("Properties70")
    _p70(p, "UpAxis", "int", "Integer", "", ("I", 1))
    _p70(p, "UpAxisSign", "int", "Integer", "", ("I", 1))
    _p70(p, "FrontAxis", "int", "Integer", "", ("I", 2))
    _p70(p, "FrontAxisSign", "int", "Integer", "", ("I", 1))
    _p70(p, "CoordAxis", "int", "Integer", "", ("I", 0))
    _p70(p, "CoordAxisSign", "int", "Integer", "", ("I", 1))
    _p70(p, "UnitScaleFactor", "double", "Number", "", ("D", 1.0))
    _p70(p, "OriginalUnitScaleFactor", "double", "Number", "", ("D", 1.0))
    root.children.append(gs)

    # --- documents ---------------------------------------------------------
    doc_id = new_id()
    docs = FbxNode("Documents")
    docs.add("Count", ("I", 1))
    d = docs.add("Document", ("L", doc_id), ("S", "Scene"), ("S", "Scene"))
    dp = d.add("Properties70")
    _p70(dp, "SourceObject", "object", "", "")
    _p70(dp, "ActiveAnimStackName", "KString", "", "", ("S", ""))
    d.add("RootNode", ("L", 0))
    root.children.append(docs)

    root.children.append(FbxNode("References"))

    # --- definitions -------------------------------------------------------
    n_models = len(meshes)
    n_geoms = len(meshes)
    n_mats = len(mat_ids)
    n_texs = len(tex_ids)
    defs = FbxNode("Definitions")
    defs.add("Version", ("I", 100))
    defs.add("Count", ("I", 1 + n_models + n_geoms + n_mats + n_texs * 2))
    defs.add("ObjectType", ("S", "GlobalSettings")).add("Count", ("I", 1))
    if n_models:
        defs.add("ObjectType", ("S", "Model")).add("Count", ("I", n_models))
    if n_geoms:
        defs.add("ObjectType", ("S", "Geometry")).add("Count", ("I", n_geoms))
    if n_mats:
        defs.add("ObjectType", ("S", "Material")).add("Count", ("I", n_mats))
    if n_texs:
        defs.add("ObjectType", ("S", "Texture")).add("Count", ("I", n_texs))
        defs.add("ObjectType", ("S", "Video")).add("Count", ("I", n_texs))
    root.children.append(defs)

    # --- objects -----------------------------------------------------------
    objs = FbxNode("Objects")
    connections = []                                # (kind, child, parent, prop)

    for mesh in meshes:
        name = sanitize(mesh.name)
        geom_id = new_id()
        model_id = new_id()

        # flatten polygons; last index of each polygon is bit-inverted
        poly_idx = []
        poly_mats = []
        slots = []                                  # material slot order for this mesh
        for s in mesh.surfaces:
            if s.brush_id not in slots:
                slots.append(s.brush_id)
            slot = slots.index(s.brush_id)
            for a, b, c in s.tris:
                poly_idx.extend((a, b, ~c))
                poly_mats.append(slot)

        verts = []
        for x, y, z in mesh.pos:
            verts.extend((float(x), float(y), float(z)))

        g = objs.add("Geometry", ("L", geom_id), ("R", _fbx_obj_name(name, "Geometry")), ("S", "Mesh"))
        g.add("GeometryVersion", ("I", 124))
        g.add("Vertices", ("d", verts))
        g.add("PolygonVertexIndex", ("i", poly_idx))

        if len(mesh.normal) == len(mesh.pos):
            nrm = []
            for nx, ny, nz in mesh.normal:
                nrm.extend((float(nx), float(ny), float(nz)))
            ln = g.add("LayerElementNormal", ("I", 0))
            ln.add("Version", ("I", 102))
            ln.add("Name", ("S", ""))
            ln.add("MappingInformationType", ("S", "ByVertice"))
            ln.add("ReferenceInformationType", ("S", "Direct"))
            ln.add("Normals", ("d", nrm))

        if len(mesh.uv) == len(mesh.pos):
            uvs = []
            for u, v in mesh.uv:
                uvs.extend((float(u), float(v)))
            uv_index = [(~i if i < 0 else i) for i in poly_idx]
            lu = g.add("LayerElementUV", ("I", 0))
            lu.add("Version", ("I", 101))
            lu.add("Name", ("S", "UVMap"))
            lu.add("MappingInformationType", ("S", "ByPolygonVertex"))
            lu.add("ReferenceInformationType", ("S", "IndexToDirect"))
            lu.add("UV", ("d", uvs))
            lu.add("UVIndex", ("i", uv_index))

        lm = g.add("LayerElementMaterial", ("I", 0))
        lm.add("Version", ("I", 101))
        lm.add("Name", ("S", ""))
        if len(slots) <= 1:
            lm.add("MappingInformationType", ("S", "AllSame"))
            lm.add("ReferenceInformationType", ("S", "IndexToDirect"))
            lm.add("Materials", ("i", [0]))
        else:
            lm.add("MappingInformationType", ("S", "ByPolygon"))
            lm.add("ReferenceInformationType", ("S", "IndexToDirect"))
            lm.add("Materials", ("i", poly_mats))

        layer = g.add("Layer", ("I", 0))
        layer.add("Version", ("I", 100))
        for etype in ("LayerElementNormal", "LayerElementUV", "LayerElementMaterial"):
            if etype == "LayerElementNormal" and len(mesh.normal) != len(mesh.pos):
                continue
            if etype == "LayerElementUV" and len(mesh.uv) != len(mesh.pos):
                continue
            le = layer.add("LayerElement")
            le.add("Type", ("S", etype))
            le.add("TypedIndex", ("I", 0))

        m = objs.add("Model", ("L", model_id), ("R", _fbx_obj_name(name, "Model")), ("S", "Mesh"))
        m.add("Version", ("I", 232))
        mp = m.add("Properties70")
        _p70(mp, "Lcl Translation", "Lcl Translation", "", "A", ("D", 0.0), ("D", 0.0), ("D", 0.0))
        _p70(mp, "Lcl Rotation", "Lcl Rotation", "", "A", ("D", 0.0), ("D", 0.0), ("D", 0.0))
        _p70(mp, "Lcl Scaling", "Lcl Scaling", "", "A", ("D", 1.0), ("D", 1.0), ("D", 1.0))
        _p70(mp, "DefaultAttributeIndex", "int", "Integer", "", ("I", 0))
        m.add("Shading", ("C", True))
        m.add("Culling", ("S", "CullingOff"))

        connections.append(("OO", geom_id, model_id, None))
        connections.append(("OO", model_id, 0, None))
        for bid in slots:
            connections.append(("OO", mat_ids[bid], model_id, None))

    for bid in brush_ids:
        b = scene.brushes[bid] if 0 <= bid < len(scene.brushes) else None
        r, gg, bb, a = b.color if b else (1.0, 1.0, 1.0, 1.0)
        shin = b.shininess if b else 0.0
        mname = material_name(scene, bid)
        mat = objs.add("Material", ("L", mat_ids[bid]),
                       ("R", _fbx_obj_name(mname, "Material")), ("S", ""))
        mat.add("Version", ("I", 102))
        mat.add("ShadingModel", ("S", "phong"))
        mat.add("MultiLayer", ("I", 0))
        pp = mat.add("Properties70")
        _p70(pp, "ShadingModel", "KString", "", "", ("S", "phong"))
        _p70(pp, "DiffuseColor", "Color", "", "A", ("D", r), ("D", gg), ("D", bb))
        _p70(pp, "AmbientColor", "Color", "", "A", ("D", 0.0), ("D", 0.0), ("D", 0.0))
        _p70(pp, "SpecularColor", "Color", "", "A", ("D", shin), ("D", shin), ("D", shin))
        _p70(pp, "Shininess", "double", "Number", "", ("D", shin * 128.0))
        _p70(pp, "Opacity", "double", "Number", "", ("D", a))

        if bid not in tex_links:
            continue
        rel, absolute = tex_links[bid]
        tname = sanitize(os.path.basename(rel))

        vid = objs.add("Video", ("L", vid_ids[bid]),
                       ("R", _fbx_obj_name(tname, "Video")), ("S", "Clip"))
        vid.add("Type", ("S", "Clip"))
        vp = vid.add("Properties70")
        _p70(vp, "Path", "KString", "XRefUrl", "", ("S", absolute))
        vid.add("UseMipMap", ("I", 0))
        vid.add("Filename", ("S", absolute))
        vid.add("RelativeFilename", ("S", rel))

        t = objs.add("Texture", ("L", tex_ids[bid]),
                     ("R", _fbx_obj_name(tname, "Texture")), ("S", ""))
        t.add("Type", ("S", "TextureVideoClip"))
        t.add("Version", ("I", 202))
        t.add("TextureName", ("R", _fbx_obj_name(tname, "Texture")))
        tp = t.add("Properties70")
        _p70(tp, "UseMaterial", "bool", "", "", ("I", 1))
        t.add("Media", ("R", _fbx_obj_name(tname, "Video")))
        t.add("FileName", ("S", absolute))
        t.add("RelativeFilename", ("S", rel))
        t.add("ModelUVTranslation", ("D", 0.0), ("D", 0.0))
        t.add("ModelUVScaling", ("D", 1.0), ("D", 1.0))
        t.add("Texture_Alpha_Source", ("S", "None"))
        t.add("Cropping", ("I", 0), ("I", 0), ("I", 0), ("I", 0))

        connections.append(("OP", tex_ids[bid], mat_ids[bid], "DiffuseColor"))
        connections.append(("OO", vid_ids[bid], tex_ids[bid], None))

    root.children.append(objs)

    # --- connections -------------------------------------------------------
    conn = FbxNode("Connections")
    for kind, child, parent, prop in connections:
        if prop is None:
            conn.add("C", ("S", kind), ("L", child), ("L", parent))
        else:
            conn.add("C", ("S", kind), ("L", child), ("L", parent), ("S", prop))
    root.children.append(conn)

    # --- serialize ---------------------------------------------------------
    out = bytearray()
    out += b"Kaydara FBX Binary  \x00\x1a\x00"
    out += struct.pack("<I", _FBX_VERSION)
    for node in root.children:
        out += node.encode(len(out))
    out += _SENTINEL

    out += _FOOTER_ID
    pad = 16 - (len(out) % 16)
    out += b"\x00" * (pad if pad else 16)
    out += struct.pack("<I", 0)
    out += struct.pack("<I", _FBX_VERSION)
    out += b"\x00" * 120
    out += _FOOTER_TAIL

    with open(fbx_path, "wb") as f:
        f.write(bytes(out))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_info(scene, meshes):
    print("b3d version : %d" % scene.version)
    print("textures    : %d" % len(scene.textures))
    for i, t in enumerate(scene.textures):
        print("  [%d] %s (flags=%d blend=%d)" % (i, t.file, t.flags, t.blend))
    print("brushes     : %d" % len(scene.brushes))
    for i, b in enumerate(scene.brushes):
        print("  [%d] %-20s rgba=%.2f,%.2f,%.2f,%.2f tex=%s"
              % (i, b.name, b.color[0], b.color[1], b.color[2], b.color[3], b.texture_ids))

    def walk(node, depth):
        tag = ""
        if node.mesh:
            tag = " [mesh: %d verts, %d surfaces, %d tris]" % (
                len(node.mesh.pos), len(node.mesh.surfaces),
                sum(len(s.tris) for s in node.mesh.surfaces))
        print("  %s%s%s" % ("  " * depth, node.name or "<unnamed>", tag))
        for c in node.children:
            walk(c, depth + 1)

    print("node tree   :")
    for n in scene.root_nodes:
        walk(n, 1)

    tv = sum(len(m.pos) for m in meshes)
    tt = sum(len(s.tris) for m in meshes for s in m.surfaces)
    print("flattened   : %d mesh(es), %d verts, %d tris" % (len(meshes), tv, tt))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect_inputs(patterns, recursive):
    """Expand the positional arguments into a list of .b3d files.

    Each argument may be a file, a directory (scanned for *.b3d), or a glob.
    Globs are expanded here too, so the script works identically in PowerShell,
    cmd and bash -- none of which agree on wildcard handling.

    Returns a list of (path, root) pairs; `root` is the folder the match was
    found under, used by --preserve-tree to rebuild the subfolder layout.
    """
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            hits = glob.glob(os.path.join(pat, "**", "*.b3d"), recursive=True) if recursive \
                else glob.glob(os.path.join(pat, "*.b3d"))
            root = pat
        elif os.path.isfile(pat):
            hits = [pat]
            root = os.path.dirname(pat) or "."
        else:
            hits = glob.glob(pat, recursive=recursive)
            root = os.path.dirname(pat.split("*")[0].split("?")[0]) or "."
        if not hits:
            print("warning: no .b3d files matched %r" % pat, file=sys.stderr)
        files.extend((h, root) for h in hits)

    seen = set()
    out = []
    for f, root in files:
        key = os.path.normcase(os.path.abspath(f))
        if key not in seen:
            seen.add(key)
            out.append((f, root))
    return out


def convert_one(path, root, args, resolver=None):
    with open(path, "rb") as f:
        data = f.read()

    scene = parse_b3d(data)
    meshes = flatten(scene,
                     scale=args.scale,
                     flip_handedness=not args.keep_handedness,
                     flip_uv=not args.keep_uv)

    if args.info:
        print("=== %s ===" % path)
        print_info(scene, meshes)
        return

    if not meshes:
        print("warning: no geometry found in %s" % path, file=sys.stderr)

    base = args.output or os.path.splitext(path)[0]
    if args.outdir:
        sub = ""
        if args.preserve_tree:
            rel = os.path.relpath(os.path.dirname(os.path.abspath(path)),
                                  os.path.abspath(root))
            if rel != os.curdir and not rel.startswith(os.pardir):
                sub = rel
        base = os.path.join(args.outdir, sub, os.path.basename(base))
    out_dir = os.path.dirname(os.path.abspath(base))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    written = []
    if args.format in ("obj", "both"):
        obj_path = base + ".obj"
        mtl_path = write_obj(scene, meshes, obj_path, resolver, path)
        written += [obj_path, mtl_path]
    if args.format in ("fbx", "both"):
        fbx_path = base + ".fbx"
        write_fbx(scene, meshes, fbx_path, path, resolver)
        written.append(fbx_path)

    tv = sum(len(m.pos) for m in meshes)
    tt = sum(len(s.tris) for m in meshes for s in m.surfaces)
    print("%s -> %d mesh(es), %d verts, %d tris" % (path, len(meshes), tv, tt))
    for w in written:
        print("  wrote %s (%d bytes)" % (w, os.path.getsize(w)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert Blitz3D .b3d to .obj and/or .fbx")
    ap.add_argument("input", nargs="+",
                    help="source .b3d file(s), a folder to scan, or a wildcard pattern")
    ap.add_argument("-o", "--output",
                    help="output path without extension (default: alongside the input). "
                         "Only valid with a single input file.")
    ap.add_argument("-d", "--outdir",
                    help="output directory; the input's base name is kept. Created if missing. "
                         "Combined with -o, this supplies the folder and -o the file name.")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="recurse into subfolders when the input is a folder or pattern")
    ap.add_argument("-p", "--preserve-tree", action="store_true",
                    help="with -d, rebuild the input's subfolder layout inside the output "
                         "directory instead of flattening it (avoids name collisions)")
    ap.add_argument("-f", "--format", choices=("obj", "fbx", "both"), default="obj")
    ap.add_argument("-s", "--scale", type=float, default=1.0, help="uniform scale factor")
    ap.add_argument("--keep-handedness", action="store_true",
                    help="do not negate Z / reverse winding (leave in Blitz3D left-handed space)")
    ap.add_argument("--keep-uv", action="store_true", help="do not flip the V coordinate")
    ap.add_argument("--info", action="store_true", help="print the scene tree and convert nothing")

    tex = ap.add_argument_group("texture re-linking")
    tex.add_argument("-t", "--texroot", action="append", metavar="DIR",
                     help="folder to search for texture images; repeatable. "
                          "Defaults to the input folder. Point this at the game's GFX "
                          "root when models reference textures from other folders.")
    tex.add_argument("--copy-textures", nargs="?", const="textures", metavar="SUBDIR",
                     help="copy every resolved texture next to the output, into SUBDIR "
                          "(default 'textures'), and reference it from there")
    tex.add_argument("--no-relink", action="store_true",
                     help="write texture names exactly as stored in the .b3d")
    tex.add_argument("--report-missing", action="store_true",
                     help="list every texture reference that could not be resolved")
    args = ap.parse_args(argv)

    files = collect_inputs(args.input, args.recursive)
    if not files:
        print("error: no input files", file=sys.stderr)
        return 1
    if args.output and len(files) > 1:
        ap.error("-o/--output takes a single input file; use -d/--outdir for batches")

    tex_roots = args.texroot or sorted({r for _f, r in files})
    resolver = TextureResolver(tex_roots,
                               enabled=not args.no_relink,
                               copy_dir=args.copy_textures)

    failed = 0
    for path, root in files:
        try:
            convert_one(path, root, args, resolver)
        except Exception as exc:                      # keep going through a batch
            failed += 1
            print("error: %s: %s" % (path, exc), file=sys.stderr)

    if not args.info:
        n_missing = len(resolver.unresolved)
        print("textures: %d linked, %d unresolved" % (resolver.resolved, n_missing))
        if n_missing:
            if args.report_missing:
                for src, ref in resolver.unresolved:
                    print("  %s -> %s" % (src, ref or "<empty>"))
            else:
                print("  (run with --report-missing to list them, "
                      "or point -t/--texroot at the game's GFX folder)")

    if len(files) > 1:
        print("%d/%d file(s) converted" % (len(files) - failed, len(files)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
