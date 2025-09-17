from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Optional

import ipywidgets as widgets
import leafmap.maplibregl as leafmap
from ipyevents import Event
from IPython.display import display
from strands import Agent
from strands.models.ollama import OllamaModel

from .map_tools import MapSession, MapTools

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass


def _ensure_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class GeoAgent(Agent):
    """Geospatial AI agent with interactive mapping capabilities."""

    def __init__(
        self, *, model_id: str = "llama3.1", map_instance: Optional[leafmap.Map] = None
    ) -> None:
        """Initialize the GeoAgent.

        Args:
            model_id: Ollama model identifier (default: "llama3.1").
            map_instance: Optional existing map instance.
        """
        self.session: MapSession = MapSession(map_instance)
        self.tools: MapTools = MapTools(self.session)

        # --- save a model factory we can call each turn ---
        self._model_factory: Callable[[], OllamaModel] = lambda: OllamaModel(
            host="http://localhost:11434", model_id=model_id
        )

        # build initial model (first turn)
        ollama_model: OllamaModel = self._model_factory()

        super().__init__(
            name="Leafmap Visualization Agent",
            model=ollama_model,
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
            system_prompt="You are a map control agent. Call tools with MINIMAL parameters only.\n\n"
            + "CRITICAL: Treat all kwargs parameters as optional parameters.\n"
            + "CRITICAL: NEVER include optional parameters unless user explicitly asks for them.\n\n"
            + "TOOL CALL RULES:\n"
            + "- zoom_to(zoom=N) - ONLY zoom parameter, OMIT options completely\n"
            + "- add_cog_layer(url='X') - NEVER include bands, nodata, opacity, etc.\n"
            + "- fly_to(longitude=N, latitude=N) - NEVER include zoom parameter\n"
            + "- add_basemap(name='X') - NEVER include any other parameters\n"
            + "- add_marker(lng_lat=[lon,lat]) - NEVER include popup or options\n\n"
            + "- remove_layer(name='X') - call get_layer_names() to get the layer name closest to"
            + "the name of the layer you want to remove before calling this tool\n\n"
            + "- add_overture_3d_buildings(kwargs={}) - kwargs parameter required by tool validation\n"
            + "FORBIDDEN: Optional parameters, string representations like '{}' or '[1,2,3]'\n"
            + "REQUIRED: Minimal tool calls with only what's absolutely necessary",
            callback_handler=None,
        )

    def ask(self, prompt: str) -> str:
        """Send a single-turn prompt to the agent.

        Args:
            prompt: The text prompt to send to the agent.

        Returns:
            The agent's response as a string.
        """
        _ensure_loop()
        self.model = self._model_factory()

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
            [widgets.HTML("<h3 style='margin:0 0 8px 0'>Map</h3>"), m.container],
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
                    "Add vector layer: https://github.com/opengeos/datasets/releases/download/us/us_states.geojson",
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
        self._executor = ThreadPoolExecutor(max_workers=1)

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
