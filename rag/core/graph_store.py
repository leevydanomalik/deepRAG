"""SQLite-backed entity/relation graph using the sqlite-graph extension.

Drop-in replacement for the previous Neo4j store. The extension provides
graph_node_add / graph_edge_add for inserts and a queryable graph_nodes /
graph_edges virtual table for reads. Because v0.1.0-alpha's Cypher does not
support variable-length paths, we do multi-hop expansion in Python.

Storage layout:

    graph_nodes  (managed by sqlite-graph)   id INTEGER, properties JSON
    graph_edges  (managed by sqlite-graph)   id, source, target, edge_type, properties JSON
    entity_name_idx                          name TEXT PK, node_id INT, display_name TEXT

Entity props JSON shape:
    {name, display_name, type, description, source_chunks: [chunk_id, ...]}

Relation props JSON shape:
    {description, source_chunks: [chunk_id, ...], weight}
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from rag.core.config import get_settings


class GraphStore:
    def __init__(self, db_path: str | None = None, extension_path: str | None = None):
        s = get_settings()
        self.db_path = db_path or s.graph_db_path
        self.extension_path = extension_path or s.graph_extension_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.enable_load_extension(True)
            self._conn.load_extension(self.extension_path)
        return self._conn

    def init_schema(self) -> None:
        with self.conn:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS graph USING graph()")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_name_idx (
                    name TEXT PRIMARY KEY,
                    node_id INTEGER NOT NULL UNIQUE,
                    display_name TEXT
                )
                """
            )

    def wipe(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM graph_edges")
            self.conn.execute("DELETE FROM graph_nodes")
            self.conn.execute("DELETE FROM entity_name_idx")

    # ----- writes -----
    def merge_entity(
        self, name: str, type_: str, description: str, chunk_ids: list[str]
    ) -> None:
        norm = name.strip().lower()
        display = name.strip()
        with self.conn:
            row = self.conn.execute(
                "SELECT node_id FROM entity_name_idx WHERE name = ?", (norm,)
            ).fetchone()
            if row is None:
                node_id = (
                    self.conn.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM graph_nodes"
                    ).fetchone()[0]
                    + 1
                )
                props = {
                    "name": norm,
                    "display_name": display,
                    "type": type_ or "Other",
                    "description": description or "",
                    "source_chunks": list(dict.fromkeys(chunk_ids)),
                }
                self.conn.execute(
                    "SELECT graph_node_add(?, ?)", (node_id, json.dumps(props))
                )
                self.conn.execute(
                    "INSERT INTO entity_name_idx(name, node_id, display_name) VALUES (?, ?, ?)",
                    (norm, node_id, display),
                )
                return
            node_id = row[0]
            raw = self.conn.execute(
                "SELECT properties FROM graph_nodes WHERE id = ?", (node_id,)
            ).fetchone()[0]
            props = json.loads(raw) if raw else {}
            if description:
                cur = props.get("description", "")
                if not cur:
                    props["description"] = description
                elif description not in cur:
                    props["description"] = f"{cur} | {description}"
            if type_ and not props.get("type"):
                props["type"] = type_
            merged_chunks = list(props.get("source_chunks", []) or [])
            for c in chunk_ids:
                if c and c not in merged_chunks:
                    merged_chunks.append(c)
            props["source_chunks"] = merged_chunks
            props.setdefault("name", norm)
            props.setdefault("display_name", display)
            self.conn.execute(
                "UPDATE graph_nodes SET properties = ? WHERE id = ?",
                (json.dumps(props), node_id),
            )

    def merge_relation(
        self, src: str, dst: str, type_: str, description: str, chunk_ids: list[str]
    ) -> None:
        self.merge_entity(src, "Other", "", [])
        self.merge_entity(dst, "Other", "", [])
        s_norm = src.strip().lower()
        d_norm = dst.strip().lower()
        with self.conn:
            src_id = self.conn.execute(
                "SELECT node_id FROM entity_name_idx WHERE name = ?", (s_norm,)
            ).fetchone()[0]
            dst_id = self.conn.execute(
                "SELECT node_id FROM entity_name_idx WHERE name = ?", (d_norm,)
            ).fetchone()[0]
            existing = self.conn.execute(
                """
                SELECT id, properties FROM graph_edges
                WHERE source = ? AND target = ? AND edge_type = ?
                """,
                (src_id, dst_id, type_),
            ).fetchone()
            if existing is None:
                props = {
                    "description": description or "",
                    "source_chunks": list(dict.fromkeys(chunk_ids)),
                    "weight": 1,
                }
                self.conn.execute(
                    "SELECT graph_edge_add(?, ?, ?, ?)",
                    (src_id, dst_id, type_, json.dumps(props)),
                )
                return
            edge_id, raw = existing
            p = json.loads(raw) if raw else {}
            p["weight"] = (p.get("weight", 0) or 0) + 1
            chunks = list(p.get("source_chunks", []) or [])
            for c in chunk_ids:
                if c and c not in chunks:
                    chunks.append(c)
            p["source_chunks"] = chunks
            if description:
                cur = p.get("description", "")
                if not cur:
                    p["description"] = description
                elif description not in cur:
                    p["description"] = f"{cur} | {description}"
            self.conn.execute(
                "UPDATE graph_edges SET properties = ? WHERE id = ?",
                (json.dumps(p), edge_id),
            )

    # ----- reads -----
    def get_entity(self, name: str) -> dict[str, Any] | None:
        norm = name.strip().lower()
        row = self.conn.execute(
            """
            SELECT n.properties FROM entity_name_idx i
            JOIN graph_nodes n ON n.id = i.node_id
            WHERE i.name = ?
            """,
            (norm,),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def find_entities_by_names(self, names: list[str]) -> list[dict[str, Any]]:
        norm = [n.strip().lower() for n in names if n and n.strip()]
        if not norm:
            return []
        ph = ",".join(["?"] * len(norm))
        rows = self.conn.execute(
            f"""
            SELECT n.properties FROM entity_name_idx i
            JOIN graph_nodes n ON n.id = i.node_id
            WHERE i.name IN ({ph})
            """,
            norm,
        ).fetchall()
        return [json.loads(r[0]) for r in rows if r[0]]

    def find_relations(self, entity_names: list[str]) -> list[dict[str, Any]]:
        norm = [n.strip().lower() for n in entity_names if n and n.strip()]
        if not norm:
            return []
        ph = ",".join(["?"] * len(norm))
        id_rows = self.conn.execute(
            f"SELECT name, node_id FROM entity_name_idx WHERE name IN ({ph})", norm
        ).fetchall()
        ids = [nid for _, nid in id_rows]
        if not ids:
            return []
        idph = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT e.edge_type, e.properties, si.name, ti.name
            FROM graph_edges e
            JOIN entity_name_idx si ON si.node_id = e.source
            JOIN entity_name_idx ti ON ti.node_id = e.target
            WHERE e.source IN ({idph}) OR e.target IN ({idph})
            """,
            ids + ids,
        ).fetchall()
        out: list[dict[str, Any]] = []
        for type_, raw, src_name, dst_name in rows:
            p = json.loads(raw) if raw else {}
            out.append(
                {
                    "src": src_name,
                    "dst": dst_name,
                    "type": type_,
                    "description": p.get("description", ""),
                    "source_chunks": p.get("source_chunks", []) or [],
                }
            )
        return out

    def expand_subgraph(
        self, seed_names: list[str], hops: int = 2
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        norm = [n.strip().lower() for n in seed_names if n and n.strip()]
        if not norm:
            return [], []
        ph = ",".join(["?"] * len(norm))
        rows = self.conn.execute(
            f"SELECT node_id FROM entity_name_idx WHERE name IN ({ph})", norm
        ).fetchall()
        if not rows:
            return [], []
        visited: set[int] = {r[0] for r in rows}
        frontier = set(visited)
        for _ in range(max(0, hops)):
            if not frontier:
                break
            fph = ",".join(["?"] * len(frontier))
            params = list(frontier) * 2
            edges = self.conn.execute(
                f"SELECT source, target FROM graph_edges WHERE source IN ({fph}) OR target IN ({fph})",
                params,
            ).fetchall()
            new_nodes: set[int] = set()
            for s, t in edges:
                if s not in visited:
                    new_nodes.add(s)
                if t not in visited:
                    new_nodes.add(t)
            visited |= new_nodes
            frontier = new_nodes

        vph = ",".join(["?"] * len(visited))
        node_rows = self.conn.execute(
            f"SELECT id, properties FROM graph_nodes WHERE id IN ({vph})",
            list(visited),
        ).fetchall()
        id_to_name: dict[int, str] = {}
        nodes: list[dict[str, Any]] = []
        for nid, raw in node_rows:
            p = json.loads(raw) if raw else {}
            id_to_name[nid] = p.get("name", str(nid))
            nodes.append(p)

        edge_rows = self.conn.execute(
            f"""
            SELECT source, target, edge_type, properties
            FROM graph_edges
            WHERE source IN ({vph}) AND target IN ({vph})
            """,
            list(visited) + list(visited),
        ).fetchall()
        edges_out: list[dict[str, Any]] = []
        for s, t, et, raw in edge_rows:
            p = json.loads(raw) if raw else {}
            edges_out.append(
                {
                    "src": id_to_name.get(s, str(s)),
                    "dst": id_to_name.get(t, str(t)),
                    "type": et,
                    "description": p.get("description", ""),
                    "source_chunks": p.get("source_chunks", []) or [],
                }
            )
        return nodes, edges_out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def extension_available() -> bool:
    """Lightweight pre-flight check for the sqlite-graph extension."""
    s = get_settings()
    return os.path.exists(s.graph_extension_path)
