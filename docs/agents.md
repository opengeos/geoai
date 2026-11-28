# agents module

The `geoai.agents` package provides LLM-powered geospatial AI agents built on [Strands Agents](https://strandsagents.com) for interactive mapping, STAC imagery discovery, data catalog search, and sandboxed raster/vector processing.

Install the optional `agents` dependency group to use this package:

```bash
pip install "geoai-py[agents]"
```

## Overview

### Agents and Model Providers

- **`GeoAgent`**: General-purpose geospatial AI agent combining interactive map visualization (`MapTools`) and geospatial workflows.
- **`STACAgent`**: Specialized agent for searching and exploring STAC catalogs such as Microsoft Planetary Computer (`STACTools`).
- **`CatalogAgent`**: Specialized agent for discovering geospatial datasets across data catalogs (`CatalogTools`).
- **Model provider factories**: `create_ollama_model`, `create_anthropic_model`, `create_openai_model`, `create_gemini_model`, `create_bedrock_model`, `create_minimax_model`, and `create_vllm_model`.

See the [geo_agents module](geo_agents.md) and [map_tools module](map_tools.md) pages for full details on agent classes, UI callbacks, and interactive map controls.

### Tool Suites

- **`DataTools`** (`geoai.agents.data_tools`): Agent-callable tools for `inspect_raster`, `inspect_vector`, `clip_raster`, `raster_to_vector`, and `list_files`. Supports directory sandboxing via `workspace_dir` and `allowed_roots`: relative paths resolve inside `workspace_dir`, symlinks are normalized with `os.path.realpath`, and any read, listing, or write made through `DataTools` outside approved roots is denied with a `ValueError`. Note that this restriction applies only to `DataTools` calls and does not sandbox other tool suites such as `MapTools`.
- **`STACTools`** (`geoai.agents.stac_tools`): Search STAC collections and items, resolve locations, and return structured `STACSearchResult` metadata (`geoai.agents.stac_models`).
- **`CatalogTools`** (`geoai.agents.catalog_tools`): Query geospatial dataset catalogs and return structured `CatalogSearchResult` metadata (`geoai.agents.catalog_models`).
- **`MapTools`** (`geoai.agents.map_tools`): Control Leafmap/MapLibre interactive maps (basemaps, layers, zoom, pan, and bounds).

## Quickstart

```python
from geoai.agents import DataTools, GeoAgent, STACAgent, create_ollama_model

# Restrict DataTools file operations (inspect, clip, vectorize, list) to a workspace directory.
# Note: this sandboxes calls through `data_tools` only; `GeoAgent` registers its own `MapTools`
# separately without DataTools path restrictions.
data_tools = DataTools(workspace_dir="./geospatial_workspace")
print(data_tools.list_files("."))

# Create an agent backed by a local Ollama model (or OpenAI, Anthropic, Gemini, Bedrock)
model = create_ollama_model(model_id="llama3.1")
agent = GeoAgent(model=model)

# Or search STAC catalogs with STACAgent
stac_agent = STACAgent(model=model)
```

For end-to-end interactive notebooks, see:

- [AI Agents](examples/AI_agents.md)
- [STAC Agents](examples/STAC_agents.md)
- [Catalog Search Agent](examples/catalog_search_agent.md)

## API Reference

::: geoai.agents

### `geoai.agents.catalog_models`

::: geoai.agents.catalog_models

### `geoai.agents.catalog_tools`

::: geoai.agents.catalog_tools

### `geoai.agents.data_tools`

::: geoai.agents.data_tools

### `geoai.agents.stac_models`

::: geoai.agents.stac_models

### `geoai.agents.stac_tools`

::: geoai.agents.stac_tools
