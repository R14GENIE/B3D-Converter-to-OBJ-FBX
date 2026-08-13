# b3d_convert

Convert Blitz3D `.b3d` models to Wavefront `.obj` (+ `.mtl`) or binary `.fbx`.

Pure Python, standard library only. No Blender, no Autodesk FBX SDK, no assimp, no pip install. One file, drop it anywhere.

Written for extracting props from *SCP – Containment Breach* and similar Blitz3D games, but it reads any well-formed `.b3d`.

## Requirements

Python 3.7 or newer. That's it.

## Quick start

```bash
python b3d_convert.py chair.b3d
```

Writes `chair.obj` and `chair.mtl` next to the input.

```bash
python b3d_convert.py chair.b3d --format fbx
python b3d_convert.py chair.b3d --format both -d exported
```

Inspect a file without converting anything:

```bash
python b3d_convert.py chair.b3d --info
```

```
b3d version : 1
textures    : 1
  [0] metal.jpg (flags=1 blend=2)
brushes     : 1
  [0] metal                rgba=0.59,0.59,0.59,1.00 tex=[0]
node tree   :
  Scene_Root [mesh: 782 verts, 1 surfaces, 682 tris]
flattened   : 1 mesh(es), 782 verts, 682 tris
```

## Batch conversion

The input accepts files, folders, or wildcards. Globs are expanded by the script itself, so the exact same command works in PowerShell, cmd and bash — none of which agree on wildcard handling.

```bash
# every .b3d in a folder
python b3d_convert.py Props -d exported -f fbx

# whole tree, rebuilding the subfolder layout in the output
python b3d_convert.py GFX -r -p -d exported -f fbx
```

`-p` / `--preserve-tree` matters more than it looks: game asset trees routinely contain several files with the same name in different folders. Without it they overwrite each other silently.

One bad file doesn't abort the batch — it's reported and the run continues.

## Texture re-linking

The main reason this script exists. Texture paths inside `.b3d` files are usually broken by the time you get them, in four distinct ways:

| Stored in the file | What it actually is |
|---|---|
| `metal.jpg` | relative to the model — usually fine |
| `GFX\Items\gas_mask.png` | relative to the *game root*, not the model |
| `C:\Users\bob\Desktop\proj\GFX\radio.png` | absolute path from the original author's PC |
| `dirtymetal.jpg` | bare name, file living in a completely different folder |

`b3d_convert` indexes every image under the search roots by each suffix of its relative path, then matches the longest tail of the reference that exists on disk. It also falls back to a same-stem match with a different extension, since `.jpg` references pointing at converted `.png` files are common. When several candidates match, the one sharing the longest directory prefix with the model wins.

Point `-t` at the game's `GFX` root:

```bash
python b3d_convert.py GFX -r -p -f obj -d exported -t GFX
```

Or produce a self-contained folder with the textures copied in beside the models:

```bash
python b3d_convert.py GFX -r -p -f obj -d exported -t GFX --copy-textures
```

That writes `map_Kd textures/gas_mask.png` and copies each resolved image into `exported/textures/`, deduplicating by content and renaming on basename collisions. This is the mode to use when importing into Blender, Roblox Studio, or anything else that resolves paths relative to the model.

Every run prints a tally. References that genuinely don't exist on disk are kept as authored, marked with a comment in the `.mtl`, and listed with `--report-missing` rather than being silently dropped:

```
textures: 223 linked, 4 unresolved
```

## Options

| Option | Meaning |
|---|---|
| `input...` | files, folders, or wildcard patterns |
| `-o`, `--output` | output path without extension (single input only) |
| `-d`, `--outdir` | output folder, input base name kept; created if missing |
| `-r`, `--recursive` | recurse into subfolders |
| `-p`, `--preserve-tree` | rebuild the input's subfolder layout under `-d` |
| `-f`, `--format` | `obj` (default), `fbx`, or `both` |
| `-s`, `--scale` | uniform scale factor |
| `-t`, `--texroot` | folder to search for textures; repeatable |
| `--copy-textures [SUBDIR]` | copy resolved textures next to the output (default `textures`) |
| `--no-relink` | write texture names exactly as stored |
| `--report-missing` | list every unresolved texture reference |
| `--keep-handedness` | skip the left-handed to right-handed conversion |
| `--keep-uv` | don't flip the V coordinate |
| `--info` | print the scene tree, convert nothing |

## Conversions applied by default

- **Handedness.** Blitz3D is left-handed Y-up; OBJ and FBX are right-handed Y-up. Z is negated and triangle winding reversed so normals still face outward. Disable with `--keep-handedness`.
- **UVs.** Blitz3D stores DirectX-style UVs where V grows downward. V is flipped. Disable with `--keep-uv`.
- **Transforms.** Node transforms are baked into world space and the output is a flat list of meshes. This avoids Euler-order ambiguity entirely.

## What it reads and writes

Parses the full chunk tree: `TEXS`, `BRUS`, `NODE` (recursive), `MESH`, `VRTS` (including the normal, vertex-colour and multi-UV-set flags) and `TRIS`.

The FBX output is binary FBX 7400, written by hand — Geometry with `LayerElementNormal` / `LayerElementUV` / `LayerElementMaterial`, Model, Material, Texture and Video objects, a Connections graph, zlib-compressed property arrays and a correct footer. Multi-material meshes get per-polygon material assignment.

## Limitations

- **No skeletons or animation.** `BONE`, `KEYS` and `ANIM` chunks are skipped. Animated characters export as a static pose. Geometry only.
- Node hierarchy is flattened rather than preserved.
- Vertex colours are parsed but not written to either output format.
- Only the first UV set is exported.

## License

[0BSD](LICENSE) (BSD Zero Clause). Do whatever you want with it — no attribution required, no notice to keep. Copy the file straight into your own project if that's easier than depending on it.

## Credits

Written with [Claude](https://claude.ai) (Claude Opus 5, via Claude Code). The `.b3d` chunk layout follows the public Blitz3D file format specification.
