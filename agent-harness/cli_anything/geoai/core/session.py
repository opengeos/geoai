"""Session state management with undo/redo support.

Provides a singleton-like Session class that persists project state,
tracks modifications, and supports undo/redo via deep-copy snapshots.
"""

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from cli_anything.geoai.utils.serialization import atomic_write_json


class Session:
    """Manages project state and change history.

    Attributes:
        project: The current project dict (or None).
        project_path: Path to the project file on disk (or None).
    """

    MAX_UNDO = 50

    def __init__(self) -> None:
        """Initialize an empty session."""
        self.project: Optional[Dict[str, Any]] = None
        self.project_path: Optional[str] = None
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._modified: bool = False

    def has_project(self) -> bool:
        """Check whether a project is loaded.

        Returns:
            True if a project is loaded.
        """
        return self.project is not None

    def set_project(self, project: Dict[str, Any], path: Optional[str] = None) -> None:
        """Load a project into the session, clearing history.

        Args:
            project: Project dict.
            path: Optional file path the project was loaded from.
        """
        self.project = project
        self.project_path = path
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._modified = False

    def get_project(self) -> Dict[str, Any]:
        """Get the current project.

        Returns:
            The project dict.

        Raises:
            RuntimeError: If no project is loaded.
        """
        if self.project is None:
            raise RuntimeError(
                "No project loaded. Use 'project new' or 'project open' first."
            )
        return self.project

    def snapshot(self, description: str = "") -> None:
        """Save a snapshot of current state before a mutation.

        Args:
            description: Human-readable description of the upcoming change.
        """
        if self.project is None:
            return

        state = {
            "project": copy.deepcopy(self.project),
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }
        self._undo_stack.append(state)
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._modified = True

    def undo(self) -> str:
        """Undo the last operation.

        Returns:
            Description of what was undone.

        Raises:
            RuntimeError: If nothing to undo.
        """
        if not self._undo_stack:
            raise RuntimeError("Nothing to undo")

        state = self._undo_stack.pop()
        desc = state.get("description", "")
        self._redo_stack.append(
            {
                "project": copy.deepcopy(self.project),
                "description": desc,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self._redo_stack) > self.MAX_UNDO:
            self._redo_stack.pop(0)

        self.project = state["project"]
        self._modified = True
        return desc

    def redo(self) -> str:
        """Redo a previously undone operation.

        Returns:
            Description of what was redone.

        Raises:
            RuntimeError: If nothing to redo.
        """
        if not self._redo_stack:
            raise RuntimeError("Nothing to redo")

        state = self._redo_stack.pop()
        desc = state.get("description", "")
        self._undo_stack.append(
            {
                "project": copy.deepcopy(self.project),
                "description": desc,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)

        self.project = state["project"]
        self._modified = True
        return desc

    def history(self) -> List[Dict[str, str]]:
        """Get the undo history.

        Returns:
            List of dicts with description and timestamp for each snapshot.
        """
        return [
            {
                "index": i,
                "description": s.get("description", ""),
                "timestamp": s.get("timestamp", ""),
            }
            for i, s in enumerate(self._undo_stack)
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current session status.

        Returns:
            Status dict with project info, modification state, and history depth.
        """
        return {
            "has_project": self.has_project(),
            "project_name": (
                self.project.get("name", "untitled") if self.project else None
            ),
            "project_path": self.project_path,
            "modified": self._modified,
            "undo_depth": len(self._undo_stack),
            "redo_depth": len(self._redo_stack),
        }

    def save_session(self, path: Optional[str] = None) -> str:
        """Persist the current project to disk.

        Args:
            path: Output path. Defaults to project_path.

        Returns:
            Absolute path to the saved file.

        Raises:
            RuntimeError: If no project is loaded or no path specified.
        """
        if self.project is None:
            raise RuntimeError("No project to save")

        save_path = path or self.project_path
        if save_path is None:
            raise RuntimeError("No save path specified. Use 'project save -o <path>'.")

        new_modified = datetime.now().isoformat()
        payload = dict(self.project)
        payload["modified"] = new_modified

        abs_path = atomic_write_json(payload, save_path)

        self.project["modified"] = new_modified
        self.project_path = abs_path
        self._modified = False
        return abs_path
