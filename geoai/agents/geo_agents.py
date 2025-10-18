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

from .map_tools import MapSession, MapTools
from .stac_tools import STACTools


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
        status = widgets.HTML("<span style='color:#666'>Ready.</span>")

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
                out = self.ask(text)  # fresh Agent/model per call, silent
                _append("assistant", out)
                self._ui.status.value = "<span style='color:#0a7'>Done.</span>"
            except Exception as e:
                _append("assistant", f"[error] {type(e).__name__}: {e}")
                self._ui.status.value = (
                    "<span style='color:#c00'>Finished with an issue.</span>"
                )
            finally:
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
            system_prompt = """
            You are a STAC catalog search agent. Follow these steps EXACTLY for every search:

            STEP 1: Determine the collection ID
            Common collections:
            - "sentinel-2-l2a" for Sentinel-2 or Sentinel
            - "landsat-c2-l2" for Landsat
            - "naip" for NAIP or aerial imagery
            - "sentinel-1-grd" for Sentinel-1 or SAR
            - "aster-l1t" for ASTER
            - "cop-dem-glo-30" for DEM or elevation

            If the user asks for a collection NOT in the list above (e.g., "MODIS", "HLS", "building footprints", "land cover"):
            - Call list_collections(filter_keyword="<keyword>") using a relevant keyword from the user's query
            - Extract the most relevant collection ID from the response
            - Use that collection ID in the next step

            STEP 2: Check if user mentioned a location
            If the query contains ANY location name (e.g., "Paris", "San Francisco", "New York", "California", "Tokyo"):
            - REQUIRED: Call geocode_location("<location_name>") FIRST
            - Extract the bbox array from the response
            - This bbox MUST be included in search_items()

            Location words include: city names, state names, country names, region names, place names
            Examples: "Paris", "New York", "California", "Seattle", "San Francisco", "Tokyo", "London"

            STEP 3: Build search_items() parameters
            - collection: REQUIRED - the collection ID from Step 1
            - bbox: REQUIRED if Step 2 found a location, otherwise omit entirely
            - time_range: REQUIRED if query has dates/months/years, otherwise omit entirely
            - max_items: default to 1

            CRITICAL RULES:
            - If location mentioned → MUST call geocode_location() → MUST include bbox in search_items()
            - If dates mentioned → MUST include time_range in search_items()
            - If NO location → omit bbox parameter
            - If NO dates → omit time_range parameter

            Examples:
            - "Show me NAIP imagery for New York"
              → geocode_location("New York") → search_items(collection="naip", bbox=[...])

            - "Find Sentinel-2 imagery in August 2024"
              → search_items(collection="sentinel-2-l2a", time_range="2024-08-01/2024-08-31")

            - "Find Landsat over Paris from June to July 2023"
              → geocode_location("Paris") → search_items(collection="landsat-c2-l2", bbox=[...], time_range="2023-06-01/2023-07-31")

            - "Show me MODIS data for California"
              → list_collections(filter_keyword="modis") → geocode_location("California") → search_items(collection="modis-...", bbox=[...])

            STEP 4: After calling search_items(), extract the FIRST item from the response and return it as JSON.

            YOUR FINAL RESPONSE MUST BE VALID JSON ONLY. No explanatory text before or after.

            Format: Return the first item from the "items" array in the tool response:
            {
              "id": "item_id_from_response",
              "collection": "collection_from_response",
              "datetime": "datetime_from_response",
              "bbox": [west, south, east, north],
              "assets": [
                {"key": "asset_key", "title": "title"}
              ],
              "properties": {}
            }

            CRITICAL:
            - Return ONLY the JSON object, nothing else
            - Use actual values from the tool response
            - If no items found, return: {"error": "No items found"}
            """

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
                            if tool_result.get("status") == "success":
                                result_content = tool_result.get("content", [])
                                if isinstance(result_content, list) and result_content:
                                    text_content = result_content[0].get("text", "")
                                    try:
                                        payload = json.loads(text_content)
                                        if "items" in payload and payload.get("items"):
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
        else:
            # Try to find common asset names
            if "assets" in item:
                asset_keys = [
                    a.get("key") for a in item["assets"] if isinstance(a, dict)
                ]
                # Look for visual or RGB assets
                for possible in ["visual", "rendered_preview", "image", "data"]:
                    if possible in asset_keys:
                        assets = [possible]
                        break
                # If still no assets, use first few assets
                if not assets and asset_keys:
                    assets = asset_keys[:1]

        if not assets:
            return

        try:
            # Add the STAC layer to the map
            layer_name = f"{collection[:20]}_{item_id[:15]}"
            self.map_instance.add_stac_layer(
                collection=collection,
                item=item_id,
                assets=assets,
                name=layer_name,
                before_id=self.map_instance.first_symbol_layer_id,
            )
        except Exception as e:
            print(f"Could not visualize item on map: {e}")

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
        status = widgets.HTML("<span style='color:#666'>Ready to search.</span>")

        examples = widgets.Dropdown(
            options=[
                ("— Example Searches —", ""),
                (
                    "Sentinel-2 over SF",
                    "Find Sentinel-2 imagery over San Francisco in August 2024",
                ),
                ("NAIP for NYC", "Show me NAIP aerial imagery for New York City"),
                (
                    "Landsat over Paris",
                    "Find Landsat imagery over Paris from June to July 2023",
                ),
                ("MODIS for California", "Show me MODIS data for California"),
                ("Building footprints", "Find building footprints in Seattle"),
                ("Land cover data", "Get land cover data for Washington DC"),
                ("Sentinel-1 SAR", "Find Sentinel-1 SAR imagery over Tokyo in 2024"),
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
                # Get the structured search result directly
                item_data = self.search_and_get_first_item(text)

                if item_data is not None:
                    # Visualize on map
                    self._visualize_stac_item(item_data)

                    # Format response for display
                    formatted_response = (
                        f"Found item: {item_data['id']}\n"
                        f"Collection: {item_data['collection']}\n"
                        f"Date: {item_data.get('datetime', 'N/A')}\n"
                        f"✓ Added to map"
                    )
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
