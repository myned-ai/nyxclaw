"""JSONL conversation store for message history.

Appends each turn (user/assistant messages, rich content) to a JSONL file.
Serves recent history to the mobile app via REST endpoint.
"""

import json
import threading
import time
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

_store: "ConversationStore | None" = None
_lock = threading.Lock()


def get_conversation_store() -> "ConversationStore":
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = ConversationStore()
    return _store


class ConversationStore:
    """Append-only JSONL conversation log."""

    MAX_ENTRIES = 500  # trim after this many lines

    def __init__(self, path: str = "./data/conversations.jsonl") -> None:
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def append(
        self,
        role: str,
        text: str,
        rich_content: str | None = None,
        item_id: str | None = None,
    ) -> None:
        """Append a conversation entry.

        If an assistant transcript arrives after a rich_content entry with the
        same item_id, the text is merged into the existing entry so the JSONL
        keeps them in natural order (text first, rich_content attached).
        """
        with self._write_lock:
            # Try to merge assistant text into a preceding rich_content-only entry
            if role == "assistant" and text and item_id and self._path.exists():
                try:
                    lines = self._path.read_text().splitlines()
                    if lines:
                        last = json.loads(lines[-1])
                        if (
                            last.get("item_id") == item_id
                            and last.get("role") == "assistant"
                            and not last.get("text")
                            and last.get("rich_content")
                        ):
                            last["text"] = text
                            lines[-1] = json.dumps(last, ensure_ascii=False)
                            self._path.write_text("\n".join(lines) + "\n")
                            return
                except (json.JSONDecodeError, OSError):
                    pass  # fall through to normal append

            entry: dict = {
                "ts": time.time(),
                "role": role,
                "text": text,
            }
            if rich_content:
                entry["rich_content"] = rich_content
            if item_id:
                entry["item_id"] = item_id

            with open(self._path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent(self, limit: int = 50, before: float | None = None) -> list[dict]:
        """Read the last N entries, optionally before a timestamp.

        Args:
            limit: Max entries to return.
            before: If set, only return entries with ts < before (for pagination).

        Returns entries oldest-first.
        """
        if not self._path.exists():
            return []

        try:
            with open(self._path) as f:
                lines = f.readlines()
        except OSError:
            return []

        all_entries: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if before is not None and entry.get("ts", 0) >= before:
                continue
            all_entries.append(entry)

        return all_entries[-limit:]

    def trim(self) -> None:
        """Trim the file to MAX_ENTRIES if it's grown too large."""
        if not self._path.exists():
            return

        with self._write_lock:
            try:
                with open(self._path) as f:
                    lines = f.readlines()
                if len(lines) <= self.MAX_ENTRIES:
                    return
                # Keep the last MAX_ENTRIES lines
                with open(self._path, "w") as f:
                    f.writelines(lines[-self.MAX_ENTRIES :])
                logger.info(f"Trimmed conversation log: {len(lines)} → {self.MAX_ENTRIES}")
            except OSError as e:
                logger.warning(f"Failed to trim conversation log: {e}")
