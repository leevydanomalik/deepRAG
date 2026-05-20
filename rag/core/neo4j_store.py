"""Neo4j adapter for entity/relation graph used by Graph RAG and Loop RAG."""
from __future__ import annotations

from typing import Any

from neo4j import Driver, GraphDatabase

from rag.core.config import get_settings


class Neo4jStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        s = get_settings()
        self.uri = uri or s.neo4j_uri
        self.user = user or s.neo4j_user
        self.password = password or s.neo4j_password.get_secret_value()
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def init_schema(self) -> None:
        with self.driver.session() as sess:
            sess.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            sess.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def wipe(self) -> None:
        with self.driver.session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")

    # ----- writes -----
    def merge_entity(self, name: str, type_: str, description: str, chunk_ids: list[str]) -> None:
        norm = name.strip().lower()
        with self.driver.session() as sess:
            sess.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $desc, e.source_chunks = $chunks, e.display_name = $display
                ON MATCH  SET e.type = coalesce(e.type, $type),
                              e.description = CASE WHEN coalesce(e.description, '') = '' THEN $desc
                                                   WHEN $desc = '' THEN e.description
                                                   ELSE e.description + ' | ' + $desc END,
                              e.source_chunks = [x IN coalesce(e.source_chunks, []) + $chunks WHERE x IS NOT NULL]
                """,
                name=norm, type=type_, desc=description, chunks=chunk_ids, display=name.strip(),
            )

    def merge_relation(self, src: str, dst: str, type_: str, description: str, chunk_ids: list[str]) -> None:
        with self.driver.session() as sess:
            sess.run(
                """
                MERGE (a:Entity {name: $src})
                MERGE (b:Entity {name: $dst})
                MERGE (a)-[r:RELATES {type: $type}]->(b)
                ON CREATE SET r.description = $desc, r.source_chunks = $chunks, r.weight = 1
                ON MATCH  SET r.weight = coalesce(r.weight, 1) + 1,
                              r.source_chunks = [x IN coalesce(r.source_chunks, []) + $chunks WHERE x IS NOT NULL],
                              r.description = CASE WHEN coalesce(r.description, '') = '' THEN $desc
                                                   WHEN $desc = '' THEN r.description
                                                   ELSE r.description + ' | ' + $desc END
                """,
                src=src.strip().lower(), dst=dst.strip().lower(),
                type=type_, desc=description, chunks=chunk_ids,
            )

    # ----- reads -----
    def get_entity(self, name: str) -> dict[str, Any] | None:
        with self.driver.session() as sess:
            rec = sess.run(
                "MATCH (e:Entity {name: $name}) RETURN e",
                name=name.strip().lower(),
            ).single()
            return dict(rec["e"]) if rec else None

    def find_entities_by_names(self, names: list[str]) -> list[dict[str, Any]]:
        norm = [n.strip().lower() for n in names]
        with self.driver.session() as sess:
            res = sess.run(
                "MATCH (e:Entity) WHERE e.name IN $names RETURN e",
                names=norm,
            )
            return [dict(r["e"]) for r in res]

    def find_relations(self, entity_names: list[str]) -> list[dict[str, Any]]:
        norm = [n.strip().lower() for n in entity_names]
        with self.driver.session() as sess:
            res = sess.run(
                """
                MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                WHERE a.name IN $names OR b.name IN $names
                RETURN a.name AS src, b.name AS dst, r.type AS type,
                       r.description AS description, r.source_chunks AS source_chunks
                """,
                names=norm,
            )
            return [r.data() for r in res]

    def expand_subgraph(self, seed_names: list[str], hops: int = 2) -> tuple[list[dict], list[dict]]:
        norm = [n.strip().lower() for n in seed_names]
        with self.driver.session() as sess:
            res = sess.run(
                f"""
                MATCH (seed:Entity) WHERE seed.name IN $names
                CALL {{
                    WITH seed
                    MATCH p=(seed)-[r:RELATES*1..{hops}]-(n)
                    RETURN nodes(p) AS ns, relationships(p) AS rs
                }}
                RETURN ns, rs
                """,
                names=norm,
            )
            nodes_by_name: dict[str, dict] = {}
            edges: list[dict] = []
            for row in res:
                for n in row["ns"]:
                    nodes_by_name[n["name"]] = dict(n)
                for r in row["rs"]:
                    edges.append({
                        "src": r.start_node["name"],
                        "dst": r.end_node["name"],
                        "type": r["type"],
                        "description": r.get("description", ""),
                        "source_chunks": r.get("source_chunks", []) or [],
                    })
            return list(nodes_by_name.values()), edges
