# MCP Server and Agent CLI

GeoAI includes two companion tools for connecting geospatial AI workflows to LLM applications and AI coding agents:

- **GeoAI MCP Server (`geoai-mcp-server`)**: A locally hosted Model Context Protocol (MCP) server for conversational AI clients such as Claude Desktop.
- **Agent CLI Harness (`cli-anything-geoai`)**: A structured command-line harness (with interactive REPL and `--json` output) designed for AI coding agents such as Claude Code and automated workflows.

## GeoAI MCP Server (`geoai-mcp-server`)

Located in the repository's [`geoai-mcp-server/`](https://github.com/opengeos/geoai/tree/main/geoai-mcp-server) directory, the MCP server exposes GeoAI's segmentation, detection, change detection, imagery download, foundation model, canopy height, and vision-language model capabilities through a sandboxed MCP interface.

### Installation

```bash
git clone https://github.com/opengeos/geoai.git
cd geoai/geoai-mcp-server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

Edit `.env` to set your preferred input and output directories. You can also launch the server directly with:

```bash
python -m geoai_mcp_server.server
```

### MCP Client Configuration (Claude Desktop)

Add the server to your Claude Desktop configuration file:

- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "geoai": {
      "command": "/path/to/geoai/geoai-mcp-server/.venv/bin/python",
      "args": ["-m", "geoai_mcp_server.server"],
      "cwd": "/path/to/geoai/geoai-mcp-server",
      "env": {
        "GEOAI_INPUT_DIR": "/path/to/your/input/data",
        "GEOAI_OUTPUT_DIR": "/path/to/your/output/results",
        "GEOAI_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Use the absolute path to the virtual environment's Python interpreter (on Windows, `C:\\path\\to\\geoai\\geoai-mcp-server\\.venv\\Scripts\\python.exe`), because Claude Desktop launches MCP servers outside your activated shell environment. Restart Claude Desktop after saving the configuration to load the GeoAI tools.

### Available MCP Tools

| Category | Tool | Description |
| -------- | ---- | ----------- |
| Segmentation | `segment_objects_with_prompts` | Segment objects in imagery using natural language text prompts. |
| Segmentation | `auto_segment_image` | Automatically segment objects in an image without prompts. |
| Detection & Classification | `detect_and_classify_features` | Detect and classify features such as buildings, vehicles, and ships. |
| Detection & Classification | `classify_land_cover` | Perform pixel-wise land cover classification. |
| Change Detection | `detect_temporal_changes` | Compare two co-registered images to identify temporal changes. |
| Data & Training | `download_satellite_imagery` | Download NAIP, Sentinel-2, or Landsat imagery for a bounding box (WGS84). |
| Data & Training | `prepare_training_data` | Export tiled training datasets from imagery and labels. |
| Foundation Models & VLM | `extract_features_with_foundation_model` | Extract embeddings or features with DINOv3 or Prithvi. |
| Foundation Models & VLM | `estimate_canopy_height` | Estimate vegetation canopy height from RGB imagery. |
| Foundation Models & VLM | `analyze_with_vision_language_model` | Caption, query, or detect objects with a vision-language model. |
| Utilities & Inspection | `clean_segmentation_results` | Post-process and clean segmentation masks or vectors. |
| Utilities & Inspection | `list_available_files` | List and inspect available files in the sandboxed input/output directories. |

### Configuration and Security

All file operations are sandboxed to `GEOAI_INPUT_DIR` and `GEOAI_OUTPUT_DIR` with path-traversal protection and glob validation. Supported environment variables:

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `GEOAI_INPUT_DIR` | Sandboxed directory for input files | `./input` |
| `GEOAI_OUTPUT_DIR` | Sandboxed directory for output results | `./output` |
| `GEOAI_TIMEOUT` | Operation timeout in seconds | `300` |
| `GEOAI_MAX_MEMORY_GB` | Maximum memory budget in GB | `8` |
| `GEOAI_DEVICE` | Compute device (`auto`, `cuda`, `mps`, `cpu`) | `auto` |
| `GEOAI_LOG_LEVEL` | Server log level (`INFO`, `DEBUG`, etc.) | `INFO` |
| `GEOAI_LOG_FILE` | Optional log file path | `None` (stderr only) |
| `GEOAI_MODEL_CACHE_SIZE` | Maximum number of cached models | `3` |

## Agent CLI Harness (`cli-anything-geoai`)

Located in [`agent-harness/`](https://github.com/opengeos/geoai/tree/main/agent-harness), `cli-anything-geoai` provides a comprehensive, agent-friendly CLI wrapper around GeoAI with both an interactive REPL and one-shot JSON commands.

### Installation

Install GeoAI first (`pip install geoai-py`), then install the harness:

```bash
cd agent-harness
pip install -e .
cli-anything-geoai --version
```

### Interactive REPL and One-Shot Commands

Start an interactive session with no arguments:

```bash
cli-anything-geoai
```

Or run one-shot commands directly:

```bash
# Inspect raster metadata and band statistics
cli-anything-geoai raster info image.tif
cli-anything-geoai raster stats image.tif --band 1

# Run SAM or text-prompted Grounded-SAM segmentation
cli-anything-geoai segment sam image.tif -o mask.tif
cli-anything-geoai segment grounded-sam image.tif -o mask.tif -p "buildings"

# Run object detection and change detection
cli-anything-geoai detect run image.tif -m model.pth -n 5 -ov detections.geojson
cli-anything-geoai change detect before.tif after.tif -o changes.tif

# Search and download imagery
cli-anything-geoai data search --bbox "-84.0,35.9,-83.9,36.0" -c sentinel-2-l2a
cli-anything-geoai data download naip --bbox "-84.0,35.9,-83.9,36.0" -o naip.tif
```

### Structured JSON Output and Project State

Pass `--json` before any subcommand for machine-readable output suited to AI agents:

```bash
# Chain structured JSON commands
cli-anything-geoai --json raster info satellite.tif
cli-anything-geoai --json segment grounded-sam satellite.tif -o buildings_mask.tif -p "buildings"
cli-anything-geoai --json raster vectorize buildings_mask.tif -o buildings.geojson
cli-anything-geoai --json vector info buildings.geojson
```

To track files, results, and undo/redo history in a workspace project, run project mutations and `project save` within the same interactive REPL session (project modifications and undo/redo history are session-local until saved):

```bash
# Create a project file, then open an interactive REPL session bound to it
cli-anything-geoai project new -n "my_analysis" -o project.json
cli-anything-geoai --project project.json
```

Inside the REPL session:

```text
project add-file image.tif
project list-files
project save
session history
```

### Command Groups

| Group | Commands | Description |
| ----- | -------- | ----------- |
| `project` | `new`, `open`, `save`, `info`, `add-file`, `list-files`, `list-results` | Workspace project management |
| `raster` | `info`, `stats`, `vectorize`, `tile` | Raster inspection and operations |
| `vector` | `info`, `rasterize` | Vector inspection and operations |
| `data` | `search`, `download`, `sources` | Data discovery and downloads |
| `segment` | `sam`, `grounded-sam`, `semantic`, `train`, `list-models`, `list-architectures` | Image segmentation |
| `detect` | `run`, `train`, `list-models` | Object detection |
| `classify` | `train`, `predict` | Image classification |
| `change` | `detect`, `list-methods` | Change detection |
| `pipeline` | `run`, `show` | Batch pipeline execution |
| `session` | `status`, `undo`, `redo`, `history` | Session state management |
| `system-info` | (standalone) | System and GPU diagnostics |

### Using with Coding Agents (Claude Code)

After installing `cli-anything-geoai`, locate the bundled `SKILL.md` command reference so coding agents can discover all flags automatically:

```bash
python -c "from pathlib import Path; import cli_anything.geoai; print(Path(cli_anything.geoai.__file__).parent / 'skills' / 'SKILL.md')"
```

You can reference that command in your project `CLAUDE.md` or agent instructions and instruct the agent to run `cli-anything-geoai --json` for geospatial tasks.
