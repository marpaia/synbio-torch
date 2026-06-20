from __future__ import annotations

import httpx
import respx

from synbiotorch.sources.sbol_db import SbolDbClient

BASE = "http://sbol-db.test"


@respx.mock
def test_list_objects_follows_keyset_cursor(object_records):
    # First page returns a cursor; second page closes the stream.
    route = respx.get(f"{BASE}/objects/list")
    route.side_effect = [
        httpx.Response(200, json={"objects": [object_records[0]], "next_cursor": "cur1"}),
        httpx.Response(200, json={"objects": [object_records[1]], "next_cursor": None}),
    ]
    client = SbolDbClient(BASE, record_class="http://sbols.org/v3#Sequence")
    records = list(client.list_objects(record_class="http://sbols.org/v3#Sequence"))
    assert [r["iri"] for r in records] == [object_records[0]["iri"], object_records[1]["iri"]]
    assert route.call_count == 2


@respx.mock
def test_iter_yields_objects_with_labels(object_records):
    respx.get(f"{BASE}/objects/list").mock(
        return_value=httpx.Response(200, json={"objects": object_records, "next_cursor": None})
    )
    client = SbolDbClient(BASE, label_key="measure")
    objects = list(client)
    assert len(objects) == 2
    assert objects[0].label == 12.5
    assert objects[0].sequence.elements == "ACGTACGTACGT"


@respx.mock
def test_neighborhood_parses_graph_slice():
    payload = {
        "root_iri": "https://example.org/c1",
        "nodes": [{"id": "https://example.org/c1", "depth": 0, "sbol_class": "Component"}],
        "edges": [
            {
                "subject": "https://example.org/c1",
                "predicate": "http://sbols.org/v3#hasSequence",
                "object": "https://example.org/s1",
                "depth": 1,
            }
        ],
        "max_depth_reached": 1,
        "truncated": False,
    }
    respx.get(f"{BASE}/objects/neighborhood").mock(return_value=httpx.Response(200, json=payload))
    client = SbolDbClient(BASE)
    slice_ = client.neighborhood("https://example.org/c1", depth=1)
    assert slice_.root_iri == "https://example.org/c1"
    assert slice_.nodes[0].record_class == "Component"
    assert slice_.edges[0].object == "https://example.org/s1"


def test_fingerprint_depends_on_filters():
    a = SbolDbClient(BASE, role="r1").fingerprint()
    b = SbolDbClient(BASE, role="r2").fingerprint()
    assert a != b
