# GeoAI MCP Server

A locally-hosted **Model Context Protocol (MCP) server** that exposes GeoAI's geospatial AI capabilities to AI agents and LLM applications like Claude Desktop.

## 🌍 Overview

The GeoAI MCP Server bridges the gap between powerful geospatial AI tools and conversational AI agents. It provides a secure, sandboxed interface for:

- **Object Segmentation** - Segment buildings, roads, vegetation using text prompts
- **Feature Detection** - Detect vehicles, ships, solar panels, and more
- **Change Detection** - Identify temporal changes between images
- **Satellite Imagery Download** - Fetch NAIP, Sentinel-2, Landsat data
- **Foundation Models** - Extract features using DINOv3 and Prithvi
- **Canopy Height Estimation** - Estimate vegetation heights from RGB imagery
- **Vision-Language Analysis** - Caption, query, and detect objects with natural language

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- GeoAI package (`geoai-py >= 0.29.0`)
- Claude Desktop (for AI agent integration)

### Installation

1. **Clone or navigate to the MCP server directory:**
   ```bash
   cd geoai-mcp-server
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

4. **Set up your environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred directories
   ```

### Claude Desktop Integration

Add the following to your Claude Desktop configuration file:

**Linux:** `~/.config/Claude/claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "geoai": {
      "command": "python",
      "args": ["-m", "geoai_mcp_server.server"],
      "cwd": "/path/to/geoai-mcp-server",
      "env": {
        "GEOAI_INPUT_DIR": "/path/to/your/input/data",
        "GEOAI_OUTPUT_DIR": "/path/to/your/output/results",
        "GEOAI_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Restart Claude Desktop to load the GeoAI tools.

## 📁 Directory Structure

```
geoai-mcp-server/
├── pyproject.toml              # Package configuration
├── .env.example                # Environment template
├── README.md                   # This file
└── src/
    └── geoai_mcp_server/
        ├── __init__.py
        ├── server.py           # Main MCP server
        ├── config.py           # Configuration management
        ├── schemas/            # Pydantic schemas
        │   ├── __init__.py
        │   └── tool_schemas.py
        └── utils/              # Utility modules
            ├── __init__.py
            ├── error_handling.py
            ├── file_management.py
            └── validation.py
```

## 🛠️ Available Tools

### Segmentation Tools

| Tool | Description |
|------|-------------|
| `segment_objects_with_prompts` | Segment objects using natural language prompts |
| `auto_segment_image` | Automatically segment all objects without prompts |

### Detection & Classification

| Tool | Description |
|------|-------------|
| `detect_and_classify_features` | Detect buildings, vehicles, ships, etc. |
| `classify_land_cover` | Pixel-wise land cover classification |

### Change Detection

| Tool | Description |
|------|-------------|
| `detect_temporal_changes` | Compare two images to find changes |

### Data Download

| Tool | Description |
|------|-------------|
| `download_satellite_imagery` | Download NAIP, Sentinel-2, Landsat imagery |
| `prepare_training_data` | Create tiled training datasets |

### Foundation Models

| Tool | Description |
|------|-------------|
| `extract_features_with_foundation_model` | Extract features using DINOv3/Prithvi |
| `estimate_canopy_height` | Estimate vegetation heights from RGB |
| `analyze_with_vision_language_model` | Caption, query, or detect with VLM |

### Utilities

| Tool | Description |
|------|-------------|
| `clean_segmentation_results` | Post-process and clean results |
| `list_available_files` | List files in input/output directories |

## 💬 Example Conversations

### Segmenting Buildings

> **User:** Find all buildings in the satellite image `urban_area.tif`

> **Claude:** I'll segment the buildings using text prompts...
>
> *[Uses `segment_objects_with_prompts` with prompts=["building"]]*
>
> Found 147 buildings in the image. The results have been saved to `urban_area_segmented.geojson`.

### Change Detection

> **User:** What changed between `area_2020.tif` and `area_2023.tif`?

> **Claude:** I'll analyze the temporal changes between these images...
>
> *[Uses `detect_temporal_changes`]*
>
> Detected 12.3% change in the area. The change map shows new construction in the northeast quadrant and vegetation loss in the south.

### Downloading Imagery

> **User:** Download recent NAIP imagery for San Francisco

> **Claude:** I'll download NAIP imagery for the San Francisco area...
>
> *[Uses `download_satellite_imagery` with data_source="naip"]*
>
> Downloaded 5 images totaling 234 MB. Files saved to the output directory.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEOAI_INPUT_DIR` | Directory for input files | `./input` |
| `GEOAI_OUTPUT_DIR` | Directory for output files | `./output` |
| `GEOAI_TEMP_DIR` | Temporary file directory | `./temp` |
| `GEOAI_LOG_LEVEL` | Logging level | `INFO` |
| `GEOAI_MAX_FILE_SIZE_MB` | Maximum file size limit | `5000` |
| `GEOAI_MAX_PROCESSING_TIME` | Max processing time (seconds) | `3600` |

### Security

The server implements several security measures:

- **Sandboxed file access** - Files can only be read from/written to configured directories
- **Path traversal protection** - Prevents `../` attacks
- **File size limits** - Configurable maximum file sizes
- **Processing timeouts** - Prevents runaway processes

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Running the Server Directly

```bash
python -m geoai_mcp_server.server
```

### Debugging

Enable debug logging:
```bash
export GEOAI_LOG_LEVEL=DEBUG
python -m geoai_mcp_server.server
```

## 📖 API Reference

### Input/Output Formats

**Supported Input Formats:**
- GeoTIFF (`.tif`, `.tiff`)
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- Cloud Optimized GeoTIFF (`.cog`)

**Supported Output Formats:**
- GeoJSON (`.geojson`) - Vector data
- GeoTIFF (`.tif`) - Raster data
- Shapefile (`.shp`) - Vector data
- GeoPackage (`.gpkg`) - Vector data

### Bounding Box Format

Geographic coordinates use WGS84 (EPSG:4326):
```
min_lon, min_lat, max_lon, max_lat
```

Example (San Francisco):
```
-122.5, 37.7, -122.4, 37.8
```

## 🔗 Related Projects

- [GeoAI](https://github.com/opengeos/geoai) - The underlying geospatial AI library
- [Model Context Protocol](https://modelcontextprotocol.io/) - The MCP specification
- [Claude Desktop](https://claude.ai/) - Anthropic's AI assistant

## 📄 License

This project is part of GeoAI and follows the same [MIT License](../LICENSE).

## 🙏 Acknowledgments

- The [GeoAI](https://github.com/opengeos/geoai) team for the excellent geospatial AI library
- Anthropic for the Model Context Protocol specification
- The open-source geospatial community
