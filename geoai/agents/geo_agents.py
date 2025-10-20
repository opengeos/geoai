from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import boto3
import ipywidgets as widgets
import leafmap.maplibregl as leafmap
from botocore.config import Config as BotocoreConfig
from ipyevents import Event
from IPython.display import display
from strands import Agent
from strands.models import BedrockModel
from strands.models.anthropic import AnthropicModel
from strands.models.ollama import OllamaModel as _OllamaModel
from strands.models.openai import OpenAIModel

from .catalog_tools import CatalogTools
from .map_tools import MapSession, MapTools
from .stac_tools import STACTools


class UICallbackHandler:
    """Callback handler that updates UI status widget with agent progress.

    This handler intercepts tool calls and progress events to provide
    real-time feedback without overwhelming the user.
    """

    def __init__(self, status_widget=None):
        """Initialize the callback handler.

        Args:
            status_widget: Optional ipywidgets.HTML widget to update with status.
        """
        self.status_widget = status_widget
        self.current_tool = None

    def __call__(self, **kwargs):
        """Handle callback events from the agent.

        Args:
            **kwargs: Event data from the agent execution.
        """
        # Track when tools are being called
        if "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
            tool_name = kwargs["current_tool_use"]["name"]
            self.current_tool = tool_name

            # Update status widget if available
            if self.status_widget is not None:
                # Make tool names more user-friendly
                friendly_name = tool_name.replace("_", " ").title()
                self.status_widget.value = (
                    f"<span style='color:#0a7'>"
                    f"<i class='fas fa-spinner fa-spin' style='font-size:1.2em'></i> "
                    f"{friendly_name}...</span>"
                )


class OllamaModel(_OllamaModel):
    """Fixed OllamaModel that ensures proper model_id handling."""

    async def stream(self, *args, **kwargs):
        """Override stream to ensure model_id is passed as string."""
        # Patch the ollama client to handle model object correctly
        import ollama

        # Save original method if not already saved
        if not hasattr(ollama.AsyncClient, "_original_chat"):
            ollama.AsyncClient._original_chat = ollama.AsyncClient.chat

            async def fixed_chat(self, **chat_kwargs):
                # If model is an OllamaModel object, extract the model_id
                if "model" in chat_kwargs and hasattr(chat_kwargs["model"], "config"):
                    chat_kwargs["model"] = chat_kwargs["model"].config["model_id"]
                return await ollama.AsyncClient._original_chat(self, **chat_kwargs)

            ollama.AsyncClient.chat = fixed_chat

        # Call the original stream method
        async for chunk in super().stream(*args, **kwargs):
            yield chunk


def create_ollama_model(
    host: str = "http://localhost:11434",
    model_id: str = "llama3.1",
    client_args: dict = None,
    **kwargs: Any,
) -> OllamaModel:
    """Create an Ollama model.

    Args:
        host: Ollama host URL.
        model_id: Ollama model ID.
        client_args: Client arguments for the Ollama model.
        **kwargs: Additional keyword arguments for the Ollama model.

    Returns:
        OllamaModel: An Ollama model.
    """
    if client_args is None:
        client_args = {}
    return OllamaModel(host=host, model_id=model_id, client_args=client_args, **kwargs)


def create_openai_model(
    model_id: str = "gpt-4o-mini",
    api_key: str = None,
    client_args: dict = None,
    **kwargs: Any,
) -> OpenAIModel:
    """Create an OpenAI model.

    Args:
        model_id: OpenAI model ID.
        api_key: OpenAI API key.
        client_args: Client arguments for the OpenAI model.
        **kwargs: Additional keyword arguments for the OpenAI model.

    Returns:
        OpenAIModel: An OpenAI model.
    """

    if api_key is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY", None)
            if api_key is None:
                raise ValueError("OPENAI_API_KEY is not set")
        except Exception:
            raise ValueError("OPENAI_API_KEY is not set")

    if client_args is None:
        client_args = kwargs.get("client_args", {})
    if "api_key" not in client_args and api_key is not None:
        client_args["api_key"] = api_key

    return OpenAIModel(client_args=client_args, model_id=model_id, **kwargs)


def create_anthropic_model(
    model_id: str = "claude-sonnet-4-20250514",
    api_key: str = None,
    client_args: dict = None,
    **kwargs: Any,
) -> AnthropicModel:
    """Create an Anthropic model.

    Args:
        model_id: Anthropic model ID. Defaults to "claude-sonnet-4-20250514".
            For a complete list of supported models,
            see https://docs.claude.com/en/docs/about-claude/models/overview.
        api_key: Anthropic API key.
        client_args: Client arguments for the Anthropic model.
        **kwargs: Additional keyword arguments for the Anthropic model.
    """

    if api_key is None:
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", None)
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is not set")
        except Exception:
            raise ValueError("ANTHROPIC_API_KEY is not set")

    if client_args is None:
        client_args = kwargs.get("client_args", {})
    if "api_key" not in client_args and api_key is not None:
        client_args["api_key"] = api_key

    return AnthropicModel(client_args=client_args, model_id=model_id, **kwargs)


def create_bedrock_model(
    model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0",
    region_name: str = None,
    boto_session: Optional[boto3.Session] = None,
    boto_client_config: Optional[BotocoreConfig] = None,
    **kwargs: Any,
) -> BedrockModel:
    """Create a Bedrock model.

    Args:
        model_id: Bedrock model ID. Run the following command to get the model ID:
            aws bedrock list-foundation-models | jq -r '.modelSummaries[].modelId'
        region_name: Bedrock region name.
        boto_session: Bedrock boto session.
        boto_client_config: Bedrock boto client config.
        **kwargs: Additional keyword arguments for the Bedrock model.
    """

    return BedrockModel(
        model_id=model_id,
        region_name=region_name,
        boto_session=boto_session,
        boto_client_config=boto_client_config,
        **kwargs,
    )


class GeoAgent(Agent):
    """Geospatial AI agent with interactive mapping capabilities."""

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        map_instance: Optional[leafmap.Map] = None,
        system_prompt: str = "default",
        model_args: dict = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GeoAgent.

        Args:
            model: Model identifier (default: "llama3.1").
            map_instance: Optional existing map instance.
            model_args: Additional keyword arguments for the model.
            **kwargs: Additional keyword arguments for the model.
        """
        self.session: MapSession = MapSession(map_instance)
        self.tools: MapTools = MapTools(self.session)

        if model_args is None:
            model_args = {}

        # --- save a model factory we can call each turn ---
        if model == "llama3.1":
            self._model_factory: Callable[[], OllamaModel] = (
                lambda: create_ollama_model(
                    host="http://localhost:11434", model_id=model, **model_args
                )
            )
        elif isinstance(model, str):
            self._model_factory: Callable[[], BedrockModel] = (
                lambda: create_bedrock_model(model_id=model, **model_args)
            )
        elif isinstance(model, OllamaModel):
            # Extract configuration from existing OllamaModel and create new instances
            model_id = model.config["model_id"]
            host = model.host
            client_args = model.client_args
            self._model_factory: Callable[[], OllamaModel] = (
                lambda: create_ollama_model(
                    host=host, model_id=model_id, client_args=client_args, **model_args
                )
            )
        elif isinstance(model, OpenAIModel):
            # Extract configuration from existing OpenAIModel and create new instances
            model_id = model.config["model_id"]
            client_args = model.client_args.copy()
            self._model_factory: Callable[[], OpenAIModel] = (
                lambda mid=model_id, client_args=client_args: create_openai_model(
                    model_id=mid, client_args=client_args, **model_args
                )
            )
        elif isinstance(model, AnthropicModel):
            # Extract configuration from existing AnthropicModel and create new instances
            model_id = model.config["model_id"]
            client_args = model.client_args.copy()
            self._model_factory: Callable[[], AnthropicModel] = (
                lambda mid=model_id, client_args=client_args: create_anthropic_model(
                    model_id=mid, client_args=client_args, **model_args
                )
            )
        else:
            raise ValueError(f"Invalid model: {model}")

        # build initial model (first turn)
        model = self._model_factory()

        if system_prompt == "default":
            system_prompt = """
            You are a map control agent. Call tools with MINIMAL parameters only.

            CRITICAL: Treat all kwargs parameters as optional parameters.
            CRITICAL: NEVER include optional parameters unless user explicitly asks for them.

            TOOL CALL RULES:
            - zoom_to(zoom=N) - ONLY zoom parameter, OMIT options completely
            - add_cog_layer(url='X') - NEVER include bands, nodata, opacity, etc.
            - fly_to(longitude=N, latitude=N) - NEVER include zoom parameter
            - add_basemap(name='X') - NEVER include any other parameters
            - add_marker(lng_lat=[lon,lat]) - NEVER include popup or options

            - remove_layer(name='X') - call get_layer_names() to get the layer name closest to
            the name of the layer you want to remove before calling this tool

            - add_overture_3d_buildings(kwargs={}) - kwargs parameter required by tool validation
            FORBIDDEN: Optional parameters, string representations like '{}' or '[1,2,3]'
            REQUIRED: Minimal tool calls with only what's absolutely necessary
            """

        super().__init__(
            name="Leafmap Visualization Agent",
            model=model,
            tools=[
                # Core navigation tools
                self.tools.fly_to,
                self.tools.create_map,
                self.tools.zoom_to,
                self.tools.jump_to,
                # Essential layer tools
                self.tools.add_basemap,
                self.tools.add_vector,
                self.tools.add_raster,
                self.tools.add_cog_layer,
                self.tools.remove_layer,
                self.tools.get_layer_names,
                self.tools.set_terrain,
                self.tools.remove_terrain,
                self.tools.add_overture_3d_buildings,
                self.tools.set_paint_property,
                self.tools.set_layout_property,
                self.tools.set_color,
                self.tools.set_opacity,
                self.tools.set_visibility,
                # self.tools.save_map,
                # Basic interaction tools
                self.tools.add_marker,
                self.tools.set_pitch,
            ],
            system_prompt=system_prompt,
            callback_handler=None,
        )

    def ask(self, prompt: str) -> str:
        """Send a single-turn prompt to the agent.

        Args:
            prompt: The text prompt to send to the agent.

        Returns:
            The agent's response as a string.
        """
        # Use strands' built-in __call__ method which now supports multiple calls
        result = self(prompt)
        return getattr(result, "final_text", str(result))

    def show_ui(self, *, height: int = 700) -> None:
        """Display an interactive UI with map and chat interface.

        Args:
            height: Height of the UI in pixels (default: 700).
        """

        m = self.tools.session.m
        if not hasattr(m, "container") or m.container is None:
            m.create_container()

        map_panel = widgets.VBox(
            [
                widgets.HTML("<h3 style='margin:0 0 8px 0'>Map</h3>"),
                m.floating_sidebar_widget,
            ],
            layout=widgets.Layout(
                flex="2 1 0%",
                min_width="520px",
                border="1px solid #ddd",
                padding="8px",
                height=f"{height}px",
                overflow="hidden",
            ),
        )

        # ----- chat widgets -----
        session_id = str(uuid.uuid4())[:8]
        title = widgets.HTML(
            f"<h3 style='margin:0'>Chatbot</h3>"
            f"<p style='margin:4px 0 8px;color:#666'>Session: {session_id}</p>"
        )
        log = widgets.HTML(
            value="<div style='color:#777'>No messages yet.</div>",
            layout=widgets.Layout(
                border="1px solid #ddd",
                padding="8px",
                height="520px",
                overflow_y="auto",
            ),
        )
        inp = widgets.Textarea(
            placeholder="Ask to add/remove/list layers, set basemap, save the map, etc.",
            layout=widgets.Layout(width="99%", height="90px"),
        )
        btn_send = widgets.Button(
            description="Send",
            button_style="primary",
            icon="paper-plane",
            layout=widgets.Layout(width="120px"),
        )
        btn_stop = widgets.Button(
            description="Stop", icon="stop", layout=widgets.Layout(width="120px")
        )
        btn_clear = widgets.Button(
            description="Clear", icon="trash", layout=widgets.Layout(width="120px")
        )
        status = widgets.HTML(
            "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>"
            "<span style='color:#666'>Ready.</span>"
        )

        examples = widgets.Dropdown(
            options=[
                ("— Examples —", ""),
                ("Fly to", "Fly to Chicago"),
                ("Add basemap", "Add basemap OpenTopoMap"),
                (
                    "Add COG layer",
                    "Add COG layer https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_rgb_train.tif",
                ),
                (
                    "Add GeoJSON",
                    "Add GeoJSON layer: https://github.com/opengeos/datasets/releases/download/us/us_states.geojson",
                ),
                ("Remove layer", "Remove layer OpenTopoMap"),
                ("Save map", "Save the map as demo.html and return the path"),
            ],
            value="",
            layout=widgets.Layout(width="auto"),
        )

        # --- state kept on self so it persists ---
        self._ui = SimpleNamespace(
            messages=[],
            map_panel=map_panel,
            title=title,
            log=log,
            inp=inp,
            btn_send=btn_send,
            btn_stop=btn_stop,
            btn_clear=btn_clear,
            status=status,
            examples=examples,
        )
        self._pending = {"fut": None}

        def _esc(s: str) -> str:
            """Escape HTML characters in a string.

            Args:
                s: Input string to escape.

            Returns:
                HTML-escaped string.
            """
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )

        def _append(role: str, msg: str) -> None:
            """Append a message to the chat log.

            Args:
                role: Role of the message sender ("user" or "assistant").
                msg: Message content.
            """
            self._ui.messages.append((role, msg))
            parts = []
            for r, mm in self._ui.messages:
                if r == "user":
                    parts.append(
                        f"<div style='margin:6px 0;padding:6px 8px;border-radius:8px;background:#eef;'><b>You</b>: {_esc(mm)}</div>"
                    )
                else:
                    parts.append(
                        f"<div style='margin:6px 0;padding:6px 8px;border-radius:8px;background:#f7f7f7;'><b>Agent</b>: {_esc(mm)}</div>"
                    )
            self._ui.log.value = (
                "<div>"
                + (
                    "".join(parts)
                    if parts
                    else "<div style='color:#777'>No messages yet.</div>"
                )
                + "</div>"
            )

        def _lock(lock: bool) -> None:
            """Lock or unlock UI controls.

            Args:
                lock: True to lock controls, False to unlock.
            """
            self._ui.btn_send.disabled = lock
            self._ui.btn_stop.disabled = not lock
            self._ui.btn_clear.disabled = lock
            self._ui.inp.disabled = lock
            self._ui.examples.disabled = lock

        def _on_send(_: Any = None) -> None:
            """Handle send button click or Enter key press."""
            text = self._ui.inp.value.strip()
            if not text:
                return
            _append("user", text)
            _lock(True)
            self._ui.status.value = "<span style='color:#0a7'>Running…</span>"
            try:
                # Create a callback handler that updates the status widget
                callback_handler = UICallbackHandler(status_widget=self._ui.status)

                # Temporarily set callback_handler for this call
                old_callback = self.callback_handler
                self.callback_handler = callback_handler

                out = self.ask(text)  # fresh Agent/model per call, with callback
                _append("assistant", out)
                self._ui.status.value = "<span style='color:#0a7'>Done.</span>"
            except Exception as e:
                _append("assistant", f"[error] {type(e).__name__}: {e}")
                self._ui.status.value = (
                    "<span style='color:#c00'>Finished with an issue.</span>"
                )
            finally:
                # Restore old callback handler
                self.callback_handler = old_callback
                self._ui.inp.value = ""
                _lock(False)

        def _on_stop(_: Any = None) -> None:
            """Handle stop button click."""
            fut = self._pending.get("fut")
            if fut and not fut.done():
                self._pending["fut"] = None
                self._ui.status.value = "<span style='color:#c00'>Stop requested. If it finishes, result will be ignored.</span>"
                _lock(False)

        def _on_clear(_: Any = None) -> None:
            """Handle clear button click."""
            self._ui.messages.clear()
            self._ui.log.value = "<div style='color:#777'>No messages yet.</div>"
            self._ui.status.value = "<span style='color:#666'>Cleared.</span>"

        def _on_example_change(change: dict[str, Any]) -> None:
            """Handle example dropdown selection change.

            Args:
                change: Change event dictionary from the dropdown widget.
            """
            if change["name"] == "value" and change["new"]:
                self._ui.inp.value = change["new"]
                self._ui.examples.value = ""
                self._ui.inp.send({"method": "focus"})

        # keep handler refs
        self._handlers = SimpleNamespace(
            on_send=_on_send,
            on_stop=_on_stop,
            on_clear=_on_clear,
            on_example_change=_on_example_change,
        )

        # wire events
        self._ui.btn_send.on_click(self._handlers.on_send)
        self._ui.btn_stop.on_click(self._handlers.on_stop)
        self._ui.btn_clear.on_click(self._handlers.on_clear)
        self._ui.examples.observe(self._handlers.on_example_change, names="value")

        # Ctrl+Enter on textarea (keyup only; do not block defaults)
        self._keyev = Event(
            source=self._ui.inp, watched_events=["keyup"], prevent_default_action=False
        )

        def _on_key(e: dict[str, Any]) -> None:
            """Handle keyboard events on the input textarea.

            Args:
                e: Keyboard event dictionary.
            """
            if (
                e.get("type") == "keyup"
                and e.get("key") == "Enter"
                and e.get("ctrlKey")
            ):
                if self._ui.inp.value.endswith("\n"):
                    self._ui.inp.value = self._ui.inp.value[:-1]
                self._handlers.on_send()

        # store callback too
        self._on_key_cb: Callable[[dict[str, Any]], None] = _on_key
        self._keyev.on_dom_event(self._on_key_cb)

        buttons = widgets.HBox(
            [
                self._ui.btn_send,
                self._ui.btn_stop,
                self._ui.btn_clear,
                widgets.Box(
                    [self._ui.examples], layout=widgets.Layout(margin="0 0 0 auto")
                ),
            ]
        )
        right = widgets.VBox(
            [
                self._ui.title if hasattr(self._ui, "title") else title,
                self._ui.log,
                self._ui.inp,
                buttons,
                self._ui.status,
            ],
            layout=widgets.Layout(flex="1 1 0%", min_width="360px"),
        )
        root = widgets.HBox(
            [map_panel, right], layout=widgets.Layout(width="100%", gap="8px")
        )
        display(root)


class STACAgent(Agent):
    """AI agent for searching and interacting with STAC catalogs."""

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        system_prompt: str = "default",
        endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        model_args: dict = None,
        map_instance: Optional[leafmap.Map] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the STAC Agent.

        Args:
            model: Model identifier (default: "llama3.1").
            system_prompt: System prompt for the agent (default: "default").
            endpoint: STAC API endpoint URL. Defaults to Microsoft Planetary Computer.
            model_args: Additional keyword arguments for the model.
            map_instance: Optional leafmap.Map instance for visualization. If None, creates a new one.
            **kwargs: Additional keyword arguments for the Agent.
        """
        self.stac_tools: STACTools = STACTools(endpoint=endpoint)
        self.map_instance = map_instance if map_instance is not None else leafmap.Map()

        if model_args is None:
            model_args = {}

        # --- save a model factory we can call each turn ---
        if model == "llama3.1":
            self._model_factory: Callable[[], OllamaModel] = (
                lambda: create_ollama_model(
                    host="http://localhost:11434", model_id=model, **model_args
                )
            )
        elif isinstance(model, str):
            self._model_factory: Callable[[], BedrockModel] = (
                lambda: create_bedrock_model(model_id=model, **model_args)
            )
        elif isinstance(model, OllamaModel):
            # Extract configuration from existing OllamaModel and create new instances
            model_id = model.config["model_id"]
            host = model.host
            client_args = model.client_args
            self._model_factory: Callable[[], OllamaModel] = (
                lambda: create_ollama_model(
                    host=host, model_id=model_id, client_args=client_args, **model_args
                )
            )
        elif isinstance(model, OpenAIModel):
            # Extract configuration from existing OpenAIModel and create new instances
            model_id = model.config["model_id"]
            client_args = model.client_args.copy()
            self._model_factory: Callable[[], OpenAIModel] = (
                lambda mid=model_id, client_args=client_args: create_openai_model(
                    model_id=mid, client_args=client_args, **model_args
                )
            )
        elif isinstance(model, AnthropicModel):
            # Extract configuration from existing AnthropicModel and create new instances
            model_id = model.config["model_id"]
            client_args = model.client_args.copy()
            self._model_factory: Callable[[], AnthropicModel] = (
                lambda mid=model_id, client_args=client_args: create_anthropic_model(
                    model_id=mid, client_args=client_args, **model_args
                )
            )
        else:
            raise ValueError(f"Invalid model: {model}")

        # build initial model (first turn)
        model = self._model_factory()

        if system_prompt == "default":
            system_prompt = """You are a STAC search agent. Follow these steps EXACTLY:

1. Determine collection ID based on data type:
   - "sentinel-2-l2a" for Sentinel-2 or optical satellite imagery
   - "landsat-c2-l2" for Landsat
   - "naip" for NAIP or aerial imagery (USA only)
   - "sentinel-1-grd" for Sentinel-1 or SAR/radar
   - "cop-dem-glo-30" for DEM, elevation, or terrain data
   - "aster-l1t" for ASTER
   For other data (e.g., MODIS, land cover): call list_collections(filter_keyword="<keyword>")

2. If location mentioned:
   - Call geocode_location("<name>") FIRST
   - WAIT for the response
   - Extract the "bbox" array from the JSON response
   - This bbox is [west, south, east, north] format

3. Call search_items():
   - collection: REQUIRED
   - bbox: Use the EXACT bbox array from geocode_location (REQUIRED if location mentioned)
   - time_range: "YYYY-MM-DD/YYYY-MM-DD" format if dates mentioned
   - query: Use for cloud cover filtering (see examples)
   - max_items: 1

Cloud cover filtering:
   - "<10% cloud": query={"eo:cloud_cover": {"lt": 10}}
   - "<20% cloud": query={"eo:cloud_cover": {"lt": 20}}
   - "<5% cloud": query={"eo:cloud_cover": {"lt": 5}}

Examples:
1. "Find Landsat over Paris from June to July 2023"
   geocode_location("Paris") → {"bbox": [2.224, 48.815, 2.469, 48.902], ...}
   search_items(collection="landsat-c2-l2", bbox=[2.224, 48.815, 2.469, 48.902], time_range="2023-06-01/2023-07-31")

2. "Find Landsat with <10% cloud cover over Paris"
   geocode_location("Paris") → {"bbox": [2.224, 48.815, 2.469, 48.902], ...}
   search_items(collection="landsat-c2-l2", bbox=[2.224, 48.815, 2.469, 48.902], query={"eo:cloud_cover": {"lt": 10}})

4. Return first item as JSON:
{"id": "...", "collection": "...", "datetime": "...", "bbox": [...], "assets": [...]}

ERROR HANDLING:
- If no items found: {"error": "No items found"}
- If tool result too large: {"error": "Result too large, try narrower search"}
- If tool error: {"error": "Search failed: <error message>"}

CRITICAL: Return ONLY JSON. NO explanatory text, NO made-up data."""

        super().__init__(
            name="STAC Search Agent",
            model=model,
            tools=[
                self.stac_tools.list_collections,
                self.stac_tools.search_items,
                self.stac_tools.get_item_info,
                self.stac_tools.geocode_location,
                self.stac_tools.get_common_collections,
            ],
            system_prompt=system_prompt,
            callback_handler=None,
        )

    def _extract_search_items_payload(self, result: Any) -> Optional[Dict[str, Any]]:
        """Return the parsed payload from the search_items tool, if available."""
        # Try to get tool_results from the result object
        tool_results = getattr(result, "tool_results", None)
        if tool_results:
            for tool_result in tool_results:
                if getattr(tool_result, "tool_name", "") != "search_items":
                    continue

                payload = getattr(tool_result, "result", None)
                if payload is None:
                    continue

                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                if isinstance(payload, dict):
                    return payload

        # Alternative: check messages for tool results
        messages = getattr(self, "messages", [])
        for msg in reversed(messages):  # Check recent messages first
            # Handle dict-style messages
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", [])

                # Look for tool results in user messages (strands pattern)
                if role == "user" and isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "toolResult" in item:
                            tool_result = item["toolResult"]
                            # Check if this is a search_items result
                            # We need to look at the preceding assistant message to identify the tool
                            if tool_result.get("status") == "success":
                                result_content = tool_result.get("content", [])
                                if isinstance(result_content, list) and result_content:
                                    text_content = result_content[0].get("text", "")
                                    try:
                                        payload = json.loads(text_content)
                                        # Return ANY search_items payload, even if items is empty
                                        # This is identified by having "query" and "collection" fields
                                        if (
                                            "query" in payload
                                            and "collection" in payload
                                            and "items" in payload
                                        ):
                                            return payload
                                    except json.JSONDecodeError:
                                        continue

        return None

    def ask(self, prompt: str) -> str:
        """Send a single-turn prompt to the agent.

        Args:
            prompt: The text prompt to send to the agent.

        Returns:
            The agent's response as a string (JSON format for search queries).
        """
        # Use strands' built-in __call__ method which now supports multiple calls
        result = self(prompt)

        search_payload = self._extract_search_items_payload(result)
        if search_payload is not None:
            if "error" in search_payload:
                return json.dumps({"error": search_payload["error"]}, indent=2)

            items = search_payload.get("items") or []
            if items:
                return json.dumps(items[0], indent=2)

            return json.dumps({"error": "No items found"}, indent=2)

        return getattr(result, "final_text", str(result))

    def search_and_get_first_item(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Search for imagery and return the first item as a structured dict.

        This method sends a search query to the agent, extracts the search results
        directly from the tool calls, and returns the first item as a STACItemInfo-compatible
        dictionary.

        Note: This method uses LLM inference which adds ~5-10 seconds overhead.
        For faster searches, use STACTools directly:
            >>> from geoai.agents import STACTools
            >>> tools = STACTools()
            >>> result = tools.search_items(
            ...     collection="sentinel-2-l2a",
            ...     bbox=[-122.5, 37.7, -122.4, 37.8],
            ...     time_range="2024-08-01/2024-08-31"
            ... )

        Args:
            prompt: Natural language search query (e.g., "Find Sentinel-2 imagery
                    over San Francisco in September 2024").

        Returns:
            Dictionary containing STACItemInfo fields (id, collection, datetime,
            bbox, assets, properties), or None if no results found.

        Example:
            >>> agent = STACAgent()
            >>> item = agent.search_and_get_first_item(
            ...     "Find Sentinel-2 imagery over Paris in summer 2023"
            ... )
            >>> print(item['id'])
            >>> print(item['assets'][0]['key'])    # or 'title'
        """
        # Use strands' built-in __call__ method which now supports multiple calls
        result = self(prompt)

        search_payload = self._extract_search_items_payload(result)
        if search_payload is not None:
            if "error" in search_payload:
                print(f"Search error: {search_payload['error']}")
                return None

            items = search_payload.get("items") or []
            if items:
                return items[0]

            print("No items found in search results")
            return None

        # Fallback: try to parse the final text response
        response = getattr(result, "final_text", str(result))

        try:
            item_data = json.loads(response)

            if "error" in item_data:
                print(f"Search error: {item_data['error']}")
                return None

            if not all(k in item_data for k in ["id", "collection"]):
                print("Response missing required fields (id, collection)")
                return None

            return item_data

        except json.JSONDecodeError:
            print("Could not extract item data from agent response")
            return None

    def _visualize_stac_item(self, item: Dict[str, Any]) -> None:
        """Visualize a STAC item on the map.

        Args:
            item: STAC item dictionary with id, collection, assets, etc.
        """
        if not item or "id" not in item or "collection" not in item:
            return

        # Get the collection and item ID
        collection = item.get("collection")
        item_id = item.get("id")

        kwargs = {}

        # Determine which assets to display based on collection
        assets = None
        if collection == "sentinel-2-l2a":
            assets = ["B04", "B03", "B02"]  # True color RGB
        elif collection == "landsat-c2-l2":
            assets = ["red", "green", "blue"]  # Landsat RGB
        elif collection == "naip":
            assets = ["image"]  # NAIP 4-band imagery
        elif "sentinel-1" in collection:
            assets = ["vv"]  # Sentinel-1 VV polarization
        elif collection == "cop-dem-glo-30":
            assets = ["data"]
            kwargs["colormap_name"] = "terrain"
        elif collection == "aster-l1t":
            assets = ["VNIR"]  # ASTER L1T imagery
        elif collection == "3dep-lidar-hag":
            assets = ["data"]
            kwargs["colormap_name"] = "terrain"
        else:
            # Try to find common asset names
            if "assets" in item:
                asset_keys = [
                    a.get("key") for a in item["assets"] if isinstance(a, dict)
                ]
                # Look for visual or RGB assets
                for possible in ["visual", "image", "data"]:
                    if possible in asset_keys:
                        assets = [possible]
                        break
                # If still no assets, use first few assets
                if not assets and asset_keys:
                    assets = asset_keys[:1]

        if not assets:
            return None
        try:
            # Add the STAC layer to the map
            layer_name = f"{collection[:20]}_{item_id[:15]}"
            self.map_instance.add_stac_layer(
                collection=collection,
                item=item_id,
                assets=assets,
                name=layer_name,
                before_id=self.map_instance.first_symbol_layer_id,
                **kwargs,
            )
            return assets  # Return the assets that were visualized
        except Exception as e:
            print(f"Could not visualize item on map: {e}")
            return None

    def show_ui(self, *, height: int = 700) -> None:
        """Display an interactive UI with map and chat interface for STAC searches.

        Args:
            height: Height of the UI in pixels (default: 700).
        """
        m = self.map_instance
        if not hasattr(m, "container") or m.container is None:
            m.create_container()

        map_panel = widgets.VBox(
            [
                widgets.HTML("<h3 style='margin:0 0 8px 0'>Map</h3>"),
                m.floating_sidebar_widget,
            ],
            layout=widgets.Layout(
                flex="2 1 0%",
                min_width="520px",
                border="1px solid #ddd",
                padding="8px",
                height=f"{height}px",
                overflow="hidden",
            ),
        )

        # ----- chat widgets -----
        session_id = str(uuid.uuid4())[:8]
        title = widgets.HTML(
            f"<h3 style='margin:0'>STAC Search Agent</h3>"
            f"<p style='margin:4px 0 8px;color:#666'>Session: {session_id}</p>"
        )
        log = widgets.HTML(
            value="<div style='color:#777'>No messages yet. Try searching for satellite imagery!</div>",
            layout=widgets.Layout(
                border="1px solid #ddd",
                padding="8px",
                height="520px",
                overflow_y="auto",
            ),
        )
        inp = widgets.Textarea(
            placeholder="Search for satellite/aerial imagery (e.g., 'Find Sentinel-2 imagery over Paris in summer 2024')",
            layout=widgets.Layout(width="99%", height="90px"),
        )
        btn_send = widgets.Button(
            description="Search",
            button_style="primary",
            icon="search",
            layout=widgets.Layout(width="120px"),
        )
        btn_stop = widgets.Button(
            description="Stop", icon="stop", layout=widgets.Layout(width="120px")
        )
        btn_clear = widgets.Button(
            description="Clear", icon="trash", layout=widgets.Layout(width="120px")
        )
        status = widgets.HTML(
            "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>"
            "<span style='color:#666'>Ready to search.</span>"
        )

        examples = widgets.Dropdown(
            options=[
                ("— Example Searches —", ""),
                (
                    "Sentinel-2 over Las Vegas",
                    "Find Sentinel-2 imagery over Las Vegas in August 2025",
                ),
                (
                    "Landsat over Paris",
                    "Find Landsat imagery over Paris from June to July 2025",
                ),
                (
                    "Landsat with <10% cloud cover",
                    "Find Landsat imagery over Paris with <10% cloud cover in June 2025",
                ),
                ("NAIP for NYC", "Show me NAIP aerial imagery for New York City"),
                ("DEM for Seattle", "Show me DEM data for Seattle"),
                (
                    "3DEP Lidar HAG",
                    "Show me data over Austin from 3dep-lidar-hag collection",
                ),
                ("ASTER for Tokyo", "Show me ASTER imagery for Tokyo"),
            ],
            value="",
            layout=widgets.Layout(width="auto"),
        )

        # --- state kept on self so it persists ---
        self._ui = SimpleNamespace(
            messages=[],
            map_panel=map_panel,
            title=title,
            log=log,
            inp=inp,
            btn_send=btn_send,
            btn_stop=btn_stop,
            btn_clear=btn_clear,
            status=status,
            examples=examples,
        )
        self._pending = {"fut": None}

        def _esc(s: str) -> str:
            """Escape HTML characters in a string."""
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )

        def _append(role: str, msg: str) -> None:
            """Append a message to the chat log."""
            self._ui.messages.append((role, msg))
            parts = []
            for r, mm in self._ui.messages:
                if r == "user":
                    parts.append(
                        f"<div style='margin:6px 0;padding:6px 8px;border-radius:8px;background:#eef;'><b>You</b>: {_esc(mm)}</div>"
                    )
                else:
                    parts.append(
                        f"<div style='margin:6px 0;padding:6px 8px;border-radius:8px;background:#f7f7f7;'><b>Agent</b>: {_esc(mm)}</div>"
                    )
            self._ui.log.value = (
                "<div>"
                + (
                    "".join(parts)
                    if parts
                    else "<div style='color:#777'>No messages yet.</div>"
                )
                + "</div>"
            )

        def _lock(lock: bool) -> None:
            """Lock or unlock UI controls."""
            self._ui.btn_send.disabled = lock
            self._ui.btn_stop.disabled = not lock
            self._ui.btn_clear.disabled = lock
            self._ui.inp.disabled = lock
            self._ui.examples.disabled = lock

        def _on_send(_: Any = None) -> None:
            """Handle send button click or Enter key press."""
            text = self._ui.inp.value.strip()
            if not text:
                return
            _append("user", text)
            _lock(True)
            self._ui.status.value = "<span style='color:#0a7'>Searching…</span>"
            try:
                # Create a callback handler that updates the status widget
                callback_handler = UICallbackHandler(status_widget=self._ui.status)

                # Temporarily set callback_handler for this call
                old_callback = self.callback_handler
                self.callback_handler = callback_handler

                # Get the structured search result directly (will show progress via callback)
                item_data = self.search_and_get_first_item(text)

                if item_data is not None:
                    # Update status for visualization step
                    self._ui.status.value = (
                        "<span style='color:#0a7'>Adding to map...</span>"
                    )

                    # Visualize on map
                    visualized_assets = self._visualize_stac_item(item_data)

                    # Format response for display
                    formatted_response = (
                        f"Found item: {item_data['id']}\n"
                        f"Collection: {item_data['collection']}\n"
                        f"Date: {item_data.get('datetime', 'N/A')}\n"
                    )

                    if visualized_assets:
                        assets_str = ", ".join(visualized_assets)
                        formatted_response += f"✓ Added to map (assets: {assets_str})"
                    else:
                        formatted_response += "✓ Added to map"

                    _append("assistant", formatted_response)
                else:
                    _append(
                        "assistant",
                        "No items found. Try adjusting your search query or date range.",
                    )

                self._ui.status.value = "<span style='color:#0a7'>Done.</span>"
            except Exception as e:
                _append("assistant", f"[error] {type(e).__name__}: {e}")
                self._ui.status.value = (
                    "<span style='color:#c00'>Finished with an issue.</span>"
                )
            finally:
                # Restore old callback handler
                self.callback_handler = old_callback
                self._ui.inp.value = ""
                _lock(False)

        def _on_stop(_: Any = None) -> None:
            """Handle stop button click."""
            fut = self._pending.get("fut")
            if fut and not fut.done():
                self._pending["fut"] = None
                self._ui.status.value = (
                    "<span style='color:#c00'>Stop requested.</span>"
                )
                _lock(False)

        def _on_clear(_: Any = None) -> None:
            """Handle clear button click."""
            self._ui.messages.clear()
            self._ui.log.value = "<div style='color:#777'>No messages yet.</div>"
            self._ui.status.value = "<span style='color:#666'>Cleared.</span>"

        def _on_example_change(change: dict[str, Any]) -> None:
            """Handle example dropdown selection change."""
            if change["name"] == "value" and change["new"]:
                self._ui.inp.value = change["new"]
                self._ui.examples.value = ""
                self._ui.inp.send({"method": "focus"})

        # keep handler refs
        self._handlers = SimpleNamespace(
            on_send=_on_send,
            on_stop=_on_stop,
            on_clear=_on_clear,
            on_example_change=_on_example_change,
        )

        # wire events
        self._ui.btn_send.on_click(self._handlers.on_send)
        self._ui.btn_stop.on_click(self._handlers.on_stop)
        self._ui.btn_clear.on_click(self._handlers.on_clear)
        self._ui.examples.observe(self._handlers.on_example_change, names="value")

        # Ctrl+Enter on textarea
        self._keyev = Event(
            source=self._ui.inp, watched_events=["keyup"], prevent_default_action=False
        )

        def _on_key(e: dict[str, Any]) -> None:
            """Handle keyboard events on the input textarea."""
            if (
                e.get("type") == "keyup"
                and e.get("key") == "Enter"
                and e.get("ctrlKey")
            ):
                if self._ui.inp.value.endswith("\n"):
                    self._ui.inp.value = self._ui.inp.value[:-1]
                self._handlers.on_send()

        # store callback too
        self._on_key_cb: Callable[[dict[str, Any]], None] = _on_key
        self._keyev.on_dom_event(self._on_key_cb)

        buttons = widgets.HBox(
            [
                self._ui.btn_send,
                self._ui.btn_stop,
                self._ui.btn_clear,
                widgets.Box(
                    [self._ui.examples], layout=widgets.Layout(margin="0 0 0 auto")
                ),
            ]
        )
        right = widgets.VBox(
            [
                self._ui.title,
                self._ui.log,
                self._ui.inp,
                buttons,
                self._ui.status,
            ],
            layout=widgets.Layout(flex="1 1 0%", min_width="360px"),
        )
        root = widgets.HBox(
            [map_panel, right], layout=widgets.Layout(width="100%", gap="8px")
        )
        display(root)


class CatalogAgent(Agent):
    """AI agent for searching data catalogs with natural language queries."""

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        system_prompt: str = "default",
        catalog_url: Optional[str] = None,
        catalog_df: Optional[Any] = None,
        model_args: dict = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Catalog Agent.

        Args:
            model: Model identifier (default: "llama3.1").
            system_prompt: System prompt for the agent (default: "default").
            catalog_url: URL to a catalog file (TSV, CSV, or JSON). Use JSON format for spatial search support.
                Example: "https://raw.githubusercontent.com/opengeos/Earth-Engine-Catalog/refs/heads/master/gee_catalog.json"
            catalog_df: Pre-loaded catalog as a pandas DataFrame.
            model_args: Additional keyword arguments for the model.
            **kwargs: Additional keyword arguments for the Agent.
        """
        self.catalog_tools: CatalogTools = CatalogTools(
            catalog_url=catalog_url, catalog_df=catalog_df
        )

        if model_args is None:
            model_args = {}

        # --- save a model factory we can call each turn ---
        if model == "llama3.1":
            self._model_factory: Callable[[], OllamaModel] = (
                lambda: create_ollama_model(
                    host="http://localhost:11434", model_id=model, **model_args
                )
            )
        elif isinstance(model, str):
            self._model_factory: Callable[[], BedrockModel] = (
                lambda: create_bedrock_model(model_id=model, **model_args)
            )
        elif isinstance(model, OllamaModel):
            # Extract configuration from existing OllamaModel and create new instances
            model_id = model.config["model_id"]
            host = model.host
            client_args = model.client_args
            self._model_factory: Callable[[], OllamaModel] = (
                lambda: create_ollama_model(
                    host=host, model_id=model_id, client_args=client_args, **model_args
                )
            )
        elif isinstance(model, OpenAIModel):
            # Extract configuration from existing OpenAIModel and create new instances
            model_id = model.config["model_id"]
            client_args = model.client_args.copy()
            self._model_factory: Callable[[], OpenAIModel] = (
                lambda mid=model_id, client_args=client_args: create_openai_model(
                    model_id=mid, client_args=client_args, **model_args
                )
            )
        elif isinstance(model, AnthropicModel):
            # Extract configuration from existing AnthropicModel and create new instances
            model_id = model.config["model_id"]
            client_args = model.client_args.copy()
            self._model_factory: Callable[[], AnthropicModel] = (
                lambda mid=model_id, client_args=client_args: create_anthropic_model(
                    model_id=mid, client_args=client_args, **model_args
                )
            )
        else:
            raise ValueError(f"Invalid model: {model}")

        # build initial model (first turn)
        model = self._model_factory()

        if system_prompt == "default":
            system_prompt = """You are a data catalog search agent. Your job is to help users find datasets from a data catalog.

IMPORTANT: Follow these steps EXACTLY:

1. Understand the user's query:
   - What type of data are they looking for? (e.g., landcover, elevation, imagery)
   - Are they searching for a specific geographic region? (e.g., California, San Francisco, bounding box)
   - Are they filtering by time period? (e.g., "from 2020", "between 2015-2020", "recent data")
   - Are they filtering by provider? (e.g., NASA, USGS)
   - Are they filtering by dataset type? (e.g., image, image_collection, table)

2. Use the appropriate tool:
   - search_by_region: PREFERRED for spatial queries - search datasets covering a geographic region
     * Use location parameter for place names (e.g., "California", "San Francisco")
     * Use bbox parameter for coordinates [west, south, east, north]
     * Can combine with keywords, dataset_type, provider, start_date, end_date filters
   - search_datasets: For keyword-only searches without spatial filter
     * Can filter by start_date and end_date for temporal queries
   - geocode_location: Convert location names to coordinates (called automatically by search_by_region)
   - get_dataset_info: Get details about a specific dataset by ID
   - list_dataset_types: Show available dataset types
   - list_providers: Show available data providers
   - get_catalog_stats: Get overall catalog statistics

3. Search strategy:
   - SPATIAL QUERIES: If user mentions ANY location or region, IMMEDIATELY use search_by_region
     * Pass location names directly to the location parameter - DO NOT ask user for bbox coordinates
     * Examples of locations: California, San Francisco, New York, Paris, any city/state/country name
     * search_by_region will automatically geocode location names - you don't need to call geocode_location separately
   - TEMPORAL QUERIES: If user mentions ANY time period, ALWAYS add start_date/end_date parameters
     * "from 2022" or "since 2022" or "2022 onwards" → start_date="2022-01-01"
     * "until 2023" or "before 2023" → end_date="2023-12-31"
     * "between 2020 and 2023" → start_date="2020-01-01", end_date="2023-12-31"
     * "recent" or "latest" → start_date="2020-01-01"
     * Time indicators: from, since, after, before, until, between, onwards, recent, latest
   - KEYWORD QUERIES: If no location mentioned, use search_datasets
   - Extract key search terms from the user's query
   - Use keywords parameter for the main search terms
   - Use dataset_type parameter if user specifies type (image, table, etc.)
   - Use provider parameter if user specifies provider (NASA, USGS, etc.)
   - Default max_results is 10, but can be adjusted

CRITICAL RULES:
1. NEVER ask the user to provide bbox coordinates. If they mention a location name, pass it directly to search_by_region(location="name")
2. ALWAYS add start_date or end_date when user mentions ANY time period (from, since, onwards, recent, etc.)
3. Convert years to YYYY-MM-DD format: 2022 → "2022-01-01"

4. Examples:
   - "Find landcover datasets covering California" → search_by_region(location="California", keywords="landcover")
   - "Show elevation data for San Francisco" → search_by_region(location="San Francisco", keywords="elevation")
   - "Find datasets in bbox [-122, 37, -121, 38]" → search_by_region(bbox=[-122, 37, -121, 38])
   - "Find landcover datasets from NASA" → search_datasets(keywords="landcover", provider="NASA")
   - "Show me elevation data" → search_datasets(keywords="elevation")
   - "What types of datasets are available?" → list_dataset_types()
   - "Find image collections about forests" → search_datasets(keywords="forest", dataset_type="image_collection")
   - "Find landcover data from 2020 onwards" → search_datasets(keywords="landcover", start_date="2020-01-01")
   - "Show California datasets between 2015 and 2020" → search_by_region(location="California", start_date="2015-01-01", end_date="2020-12-31")
   - "Find recent elevation data" → search_datasets(keywords="elevation", start_date="2020-01-01")

5. Return results clearly:
   - Summarize the number of results found
   - List the top results with their EXACT IDs and titles FROM THE TOOL RESPONSE
   - Mention key information like provider, type, geographic coverage, and date range if available
   - For spatial searches, mention the search region

ERROR HANDLING:
- If no results found: Suggest trying different keywords, broader region, or removing filters
- If location not found: Suggest alternative spellings or try a broader region
- If tool error: Explain the error and suggest alternatives

CRITICAL RULES - MUST FOLLOW:
1. NEVER make up or hallucinate dataset IDs, titles, or any other information
2. ONLY report datasets that appear in the actual tool response
3. Copy dataset IDs and titles EXACTLY as they appear in the tool response
4. If a field is null/None in the tool response, say "N/A" or omit it - DO NOT guess
5. DO NOT use your training data knowledge about Earth Engine datasets
6. DO NOT fill in missing information from your knowledge
7. If unsure, say "Information not available in results"

Example of CORRECT behavior:
Tool returns: {"id": "AAFC/ACI", "title": "Canada AAFC Annual Crop Inventory"}
Your response: "Found dataset: AAFC/ACI - Canada AAFC Annual Crop Inventory"

Example of INCORRECT behavior (DO NOT DO THIS):
Tool returns: {"id": "AAFC/ACI", "title": "Canada AAFC Annual Crop Inventory"}
Your response: "Found dataset: USGS/NED - USGS Elevation Data"  ← WRONG! This ID wasn't in the tool response!"""

        super().__init__(
            name="Catalog Search Agent",
            model=model,
            tools=[
                self.catalog_tools.search_datasets,
                self.catalog_tools.search_by_region,
                self.catalog_tools.get_dataset_info,
                self.catalog_tools.geocode_location,
                self.catalog_tools.list_dataset_types,
                self.catalog_tools.list_providers,
                self.catalog_tools.get_catalog_stats,
            ],
            system_prompt=system_prompt,
            callback_handler=None,
        )

    def ask(self, prompt: str) -> str:
        """Send a single-turn prompt to the agent.

        Args:
            prompt: The text prompt to send to the agent.

        Returns:
            The agent's response as a string.
        """
        # Use strands' built-in __call__ method which now supports multiple calls
        result = self(prompt)
        return getattr(result, "final_text", str(result))

    def search_datasets(
        self,
        keywords: Optional[str] = None,
        dataset_type: Optional[str] = None,
        provider: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for datasets and return structured results.

        This method directly uses the CatalogTools without LLM inference for faster searches.

        Args:
            keywords: Keywords to search for.
            dataset_type: Filter by dataset type.
            provider: Filter by provider.
            max_results: Maximum number of results to return.

        Returns:
            List of dataset dictionaries.

        Example:
            >>> agent = CatalogAgent(catalog_url="...")
            >>> datasets = agent.search_datasets(keywords="landcover", provider="NASA")
            >>> for ds in datasets:
            ...     print(ds['id'], ds['title'])
        """
        result_json = self.catalog_tools.search_datasets(
            keywords=keywords,
            dataset_type=dataset_type,
            provider=provider,
            max_results=max_results,
        )

        result = json.loads(result_json)

        if "error" in result:
            print(f"Search error: {result['error']}")
            return []

        return result.get("datasets", [])
