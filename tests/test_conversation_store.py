"""Tests for services.conversation_store — JSONL append, get_recent, trim."""

import json

from services.conversation_store import ConversationStore


class TestConversationStoreAppend:
    def test_append_creates_file_and_writes_entry(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        store = ConversationStore(path=str(path))

        store.append("user", "hello")

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["role"] == "user"
        assert entry["text"] == "hello"
        assert "ts" in entry

    def test_append_with_rich_content_and_item_id(self, tmp_path):
        store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        store.append("assistant", "", rich_content='{"type":"image"}', item_id="item-1")

        entries = store.get_recent(10)
        assert len(entries) == 1
        assert entries[0]["rich_content"] == '{"type":"image"}'
        assert entries[0]["item_id"] == "item-1"

    def test_append_omits_rich_content_when_none(self, tmp_path):
        store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        store.append("user", "hi")

        entries = store.get_recent(10)
        assert "rich_content" not in entries[0]
        assert "item_id" not in entries[0]

    def test_append_multiple_entries(self, tmp_path):
        store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        store.append("user", "one")
        store.append("assistant", "two")
        store.append("user", "three")

        entries = store.get_recent(10)
        assert len(entries) == 3
        assert [e["text"] for e in entries] == ["one", "two", "three"]


class TestConversationStoreGetRecent:
    def test_returns_empty_when_file_missing(self, tmp_path):
        store = ConversationStore(path=str(tmp_path / "nonexistent.jsonl"))
        assert store.get_recent(10) == []

    def test_returns_last_n_entries_oldest_first(self, tmp_path):
        store = ConversationStore(path=str(tmp_path / "conv.jsonl"))
        for i in range(10):
            store.append("user", f"msg-{i}")

        entries = store.get_recent(3)
        assert len(entries) == 3
        assert [e["text"] for e in entries] == ["msg-7", "msg-8", "msg-9"]

    def test_before_filters_by_timestamp(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        store = ConversationStore(path=str(path))

        # Write entries with explicit timestamps
        for i, ts in enumerate([100.0, 200.0, 300.0, 400.0]):
            entry = {"ts": ts, "role": "user", "text": f"msg-{i}"}
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        entries = store.get_recent(10, before=300.0)
        assert len(entries) == 2
        assert [e["text"] for e in entries] == ["msg-0", "msg-1"]

    def test_before_and_limit_combined(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        store = ConversationStore(path=str(path))

        for i in range(5):
            entry = {"ts": float(i * 100), "role": "user", "text": f"msg-{i}"}
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        # before=300 keeps ts 0, 100, 200; limit=2 takes last 2
        entries = store.get_recent(2, before=300.0)
        assert len(entries) == 2
        assert [e["text"] for e in entries] == ["msg-1", "msg-2"]

    def test_skips_malformed_json_lines(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        path.write_text('{"ts":1,"role":"user","text":"ok"}\nnot-json\n{"ts":2,"role":"user","text":"also-ok"}\n')

        store = ConversationStore(path=str(path))
        entries = store.get_recent(10)
        assert len(entries) == 2

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        path.write_text('{"ts":1,"role":"user","text":"a"}\n\n\n{"ts":2,"role":"user","text":"b"}\n')

        store = ConversationStore(path=str(path))
        entries = store.get_recent(10)
        assert len(entries) == 2


class TestConversationStoreTrim:
    def test_trim_keeps_last_max_entries(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        store = ConversationStore(path=str(path))
        store.MAX_ENTRIES = 5  # lower for test

        for i in range(10):
            store.append("user", f"msg-{i}")

        store.trim()

        entries = store.get_recent(100)
        assert len(entries) == 5
        assert entries[0]["text"] == "msg-5"
        assert entries[-1]["text"] == "msg-9"

    def test_trim_noop_when_under_limit(self, tmp_path):
        path = tmp_path / "conv.jsonl"
        store = ConversationStore(path=str(path))
        store.MAX_ENTRIES = 100

        store.append("user", "only one")
        store.trim()

        entries = store.get_recent(100)
        assert len(entries) == 1

    def test_trim_noop_when_file_missing(self, tmp_path):
        store = ConversationStore(path=str(tmp_path / "nonexistent.jsonl"))
        store.trim()  # should not raise
