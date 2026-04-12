# cli-anything-geoai

CLI harness for [GeoAI](https://github.com/opengeos/geoai) -- AI-powered
geospatial analysis from the command line.

## Prerequisites

GeoAI must be installed:

```bash
pip install geoai-py
# or
conda install -c conda-forge geoai
```

For GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Installation

```bash
cd agent-harness
pip install -e .
```

Verify:

```bash
which cli-anything-geoai
cli-anything-geoai --version
```

## Usage

### Interactive REPL

```bash
cli-anything-geoai
```

### One-shot commands

```bash
# Raster info
cli-anything-geoai raster info image.tif

# Raster statistics
cli-anything-geoai raster stats image.tif --band 1

# SAM segmentation
cli-anything-geoai segment sam image.tif -o mask.tif

# Text-prompted segmentation
cli-anything-geoai segment grounded-sam image.tif -o mask.tif -p "buildings"

# Object detection
cli-anything-geoai detect run image.tif -m model.pth -n 5 -ov detections.geojson

# Change detection
cli-anything-geoai change detect before.tif after.tif -o changes.tif

# Download NAIP imagery
cli-anything-geoai data download naip --bbox "-84.0,35.9,-83.9,36.0" -o naip.tif

# Search satellite imagery
cli-anything-geoai data search --bbox "-84.0,35.9,-83.9,36.0" -c sentinel-2-l2a
```

### JSON output (for AI agents)

```bash
cli-anything-geoai --json raster info image.tif
cli-anything-geoai --json segment sam image.tif -o mask.tif
```

### Project management

```bash
# Create project
cli-anything-geoai project new -n "my_analysis" -o project.json

# Add files
cli-anything-geoai --project project.json project add-file image.tif

# Save
cli-anything-geoai --project project.json project save
```

## Command Groups

| Group | Commands | Description |
|-------|----------|-------------|
| project | new, open, save, info, add-file, list-files, list-results | Workspace management |
| raster | info, stats, vectorize, tile | Raster inspection/operations |
| vector | info, rasterize | Vector inspection/operations |
| data | search, download, sources | Data discovery/download |
| segment | sam, grounded-sam, semantic, train, list-models, list-architectures | Image segmentation |
| detect | run, train, list-models | Object detection |
| classify | train, predict | Image classification |
| change | detect, list-methods | Change detection |
| pipeline | run, show | Batch processing |
| session | status, undo, redo, history | State management |
| system-info | (standalone) | System diagnostics |

## Using with Claude Code

There are several ways to make Claude Code aware of this CLI so it can use it
for geospatial AI tasks.

### Finding the SKILL.md path

After installation, the SKILL.md path can be resolved reliably with:

```bash
python -c "from pathlib import Path; import cli_anything.geoai; print(Path(cli_anything.geoai.__file__).parent / 'skills' / 'SKILL.md')"
```

The REPL banner also displays the absolute path on startup. Use the resolved
absolute path in the instructions below.

### Option 1: Reference in CLAUDE.md

Add a section to your project or global `~/.claude/CLAUDE.md`:

```markdown
## GeoAI CLI Skill

When working with geospatial AI tasks (segmentation, detection, classification,
raster/vector operations), use the `cli-anything-geoai` CLI tool with `--json`
for machine-readable output.

Run this to find the full command reference, then read the file:
python -c "from pathlib import Path; import cli_anything.geoai; print(Path(cli_anything.geoai.__file__).parent / 'skills' / 'SKILL.md')"
```

Claude Code will resolve the path, read the SKILL.md, and learn the full
command set.

### Option 2: Mention in conversation

Tell Claude Code directly:

> Use `cli-anything-geoai` for geospatial operations. Find the skill
> reference by running:
> `python -c "from pathlib import Path; import cli_anything.geoai; print(Path(cli_anything.geoai.__file__).parent / 'skills' / 'SKILL.md')"`
> then read that file for the full command reference.

### Option 3: Custom slash command

Create a skill file at `~/.claude/commands/geoai.md`:

```markdown
---
name: geoai
description: "Run geospatial AI operations via cli-anything-geoai"
---

Find the SKILL.md path by running:
python -c "from pathlib import Path; import cli_anything.geoai; print(Path(cli_anything.geoai.__file__).parent / 'skills' / 'SKILL.md')"

Read the file at that path, then use cli-anything-geoai with the --json flag
to accomplish the user's geospatial AI task. Always inspect data first
(raster info, vector info) before running operations.
```

Then invoke it in Claude Code with `/geoai <your task>`.

### How Claude Code uses it

Once aware of the CLI, Claude Code will:

1. Use `--json` on every command so it can parse structured output
2. Chain commands in multi-step workflows (inspect, process, verify)
3. Handle errors by reading the JSON error response and adjusting

Example: if you ask "segment the buildings in satellite.tif", Claude Code runs:

```bash
# 1. Inspect the file
cli-anything-geoai --json raster info satellite.tif

# 2. Run text-prompted segmentation
cli-anything-geoai --json segment grounded-sam satellite.tif \
  -o buildings_mask.tif -p "buildings"

# 3. Vectorize the result
cli-anything-geoai --json raster vectorize buildings_mask.tif \
  -o buildings.geojson

# 4. Verify the output
cli-anything-geoai --json vector info buildings.geojson
```

Every step returns structured JSON, letting Claude Code make decisions based
on the results (check band count, verify output size, retry with different
parameters).

## Running Tests

```bash
cd agent-harness
python -m pytest cli_anything/geoai/tests/ -v -s
```

Force installed command testing:

```bash
CLI_ANYTHING_FORCE_INSTALLED=1 python -m pytest cli_anything/geoai/tests/ -v -s
```
