"""Netze (kleines Netz, Stadtnetz, Zufallsnetz), Kennzahlen, Bildfolge der Verschmelzung, Abbildungen und Messreihen."""

import networkx as nx
import numpy as np
import pytest

import hl_constants as C
import hl_evaluation as ev
import hl_labels as hl
import hl_scenario as sc
import hl_visualization as viz


def _connected(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(np.repeat(np.arange(g.n), g.degree()), g.indices))
    return nx.is_connected(G)


# --- Netze ---------------------------------------------------------------------------------------------------------------------------------------

def test_small_network_is_the_eight_places_with_a_fixed_pair():
    net = sc.small_network()
    g = net.graph
    assert g.n == 8 and g.m == 24 and not g.directed and _connected(g) and [g.names[v] for v in net.fixed_pair] == ["Altstadt", "Fabrik"] and net.unit == "min"


@pytest.mark.parametrize("seed", range(3))
def test_city_network_is_a_connected_grid_with_whole_number_costs(seed):
    net = sc.city_network(8, seed)
    g = net.graph
    assert g.n == 64 and g.m == 2 * 2 * 8 * 7 and _connected(g) and (g.weight >= 1).all() and np.array_equal(g.weight, np.rint(g.weight))


def test_random_network_is_connected_with_the_requested_degree():
    g = sc.random_network(150, 3.0, 4).graph
    assert _connected(g) and abs(g.m / g.n - 3.0) < 0.1 and (g.weight >= 1).all() and (g.weight <= 9).all() and not sc.random_network(150, 3.0, 4).geometric


def test_make_network_rejects_unknown_nets_and_is_reproducible():
    with pytest.raises(ValueError):
        sc.make_network("ring")
    a, b = sc.make_network("city", side=6, seed=3).graph, sc.make_network("city", side=6, seed=3).graph
    assert np.array_equal(a.weight, b.weight) and np.array_equal(a.xy, b.xy)


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
@pytest.mark.parametrize("order", tuple(C.ORDER_LABELS))
def test_analysis_invariants_for_every_net_and_order(key, order):
    net = sc.make_network(key, side=6, nodes=40)
    a = ev.analyse(net, order, 60)
    m = a.metrics
    assert m["reachable"] and m["exact"] and m["pairs"]["exact"] and ev.verdict(a) == "ok"
    assert m["steps"] == len(a.q.trace) <= m["size_s"] + m["size_t"] and m["total"] == a.L.total() and m["avg"] == pytest.approx(m["total"] / m["n"]) and m["table"] == m["n"] ** 2
    assert a.route[0] == a.s and a.route[-1] == a.t and m["hops"] == len(a.route) - 1 and a.L.order[int(a.L.pos[m["hub"]])] == m["hub"] and m["hub"] in a.route
    assert m["settled_dijkstra"] >= 1 and m["settled_ch"] >= 2 and m["raw_avg"] >= m["clean_avg"] - 1e-9 and (m["clean_equals_pll"] is True if order == "ch" else m["clean_equals_pll"] is None)
    assert m["removed"] >= 0 and m["pll_pruned"] <= m["pll_settled"] and m["pll_searches"] == m["n"]


def test_pick_pair_follows_the_distance_percentile():
    net = sc.city_network(8, 3)
    d = {}
    for pct in (0, 50, 100):
        s, t = ev.pick_pair(net, pct, 3)
        d[pct] = hl.dijkstra(net.graph, s)[0][t]
        assert s != t
    assert d[0] <= d[50] <= d[100] and ev.pick_pair(net, 50, 3) == ev.pick_pair(net, 50, 3)
    assert ev.pick_pair(sc.small_network()) == sc.small_network().fixed_pair


# --- Bildfolge ---------------------------------------------------------------------------------------------------------------------------------------

def test_frames_and_state_follow_the_merge_trace():
    a = ev.analyse(sc.small_network())
    n = len(a.q.trace)
    assert ev.frames(a) == list(range(n + 1)) and n == 5
    assert ev.state_at(a, 0) == (0, 0, [], float("inf"))
    i, j, common, best = ev.state_at(a, n)
    assert best == a.q.dist and (i == len(a.L.f_key[a.s]) or j == len(a.L.b_key[a.t])) and a.q.hub in [c[0] for c in common] and min(c[1] for c in common) == a.q.dist
    big = ev.analyse(sc.city_network(20, 7))
    fb = ev.frames(big)
    assert fb[0] == 0 and fb[-1] == len(big.q.trace) and len(fb) <= 61 and fb == sorted(set(fb))
    prev = (0, 0)
    for k in range(len(a.q.trace) + 1):
        i, j, _, _ = ev.state_at(a, k)
        assert i >= prev[0] and j >= prev[1] and i + j >= prev[0] + prev[1]
        prev = (i, j)


def test_tables_list_labels_and_merge_steps_for_the_small_network():
    a = ev.analyse(sc.small_network())
    rows = viz.label_table(a)
    assert [r["Ort"] for r in rows] == list(a.net.graph.names) and sum(r["Einträge"] for r in rows) == 24 and all(r["Hubs (Entfernung)"] for r in rows)
    steps = viz.merge_table(a, len(a.q.trace))
    assert [r["Schritt"] for r in steps] == list(range(1, 6)) and steps[-1]["Beste Summe"] == "13" and any("gleich" in r["Hub von s ↔ Hub von t"] for r in steps)
    assert len(viz.merge_table(ev.analyse(sc.city_network(20, 7)), 39)) == 8


def test_charts_render_for_every_net_and_frame():
    for key in C.NETS:
        a = ev.analyse(sc.make_network(key, side=6, nodes=40))
        frames = ev.frames(a)
        for k in (frames[0], frames[len(frames) // 2], frames[-1]):
            viz.build_network(a, k)
        viz.build_sizes(a)
    viz.build_size([{"key": "city", "n": 36, "avg": 6.0, "max": 12.0, "share": 0.2}, {"key": "random", "n": 50, "avg": 5.0, "max": 9.0, "share": 0.1}])
    viz.build_orders([{"key": "city", "size": 10, "ch": 1.0, "degree": 2.0, "random": 3.0}])
    viz.build_construction([{"key": "random", "size": 200, "raw": 3.0, "clean": 2.0, "pll": 2.0}])
    viz.build_query([{"key": "city", "n": 36, "dijkstra": 10.0, "ch": 5.0, "merge": 3.0}, {"key": "random", "n": 50, "dijkstra": 12.0, "ch": 5.0, "merge": 4.0}])
    viz.build_stale([{"fraction": 0.05, "wrong": 0.3}])


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------------

def test_size_rows_grow_with_the_network():
    rows = ev.size_rows(cases=[("city", 5), ("city", 8), ("random", 30), ("random", 60)], seeds=C.SWEEP_SEEDS[:2])
    assert rows[0]["avg"] < rows[1]["avg"] and rows[2]["avg"] < rows[3]["avg"] and all(r["max"] >= r["avg"] and 0 < r["share"] < 1 for r in rows)
    assert rows[1]["share"] < rows[0]["share"]                                        # Anteil an der Entfernungstabelle sinkt mit der Größe


def test_order_and_construction_rows():
    o = ev.order_rows(cases=[("city", 8), ("random", 60)], seeds=C.SWEEP_SEEDS[:2])
    assert all(r["ch"] < r["random"] for r in o)
    c = ev.construction_rows(cases=[("city", 8), ("random", 60)], seeds=C.SWEEP_SEEDS[:2])
    assert all(r["raw"] > r["clean"] == pytest.approx(r["pll"]) and r["same"] == 1.0 for r in c)


def test_query_rows_are_exact_and_the_labels_need_the_fewest_steps():
    rows = ev.query_rows(cases=[("city", 8), ("random", 60)], seeds=C.SWEEP_SEEDS[:2])
    assert all(r["merge"] < r["ch"] < r["dijkstra"] for r in rows)


def test_density_and_stale_rows():
    d = ev.density_rows(degrees=(2.0, 4.0), nodes=60, seeds=C.SWEEP_SEEDS[:2])
    assert d[0]["avg"] < d[1]["avg"]
    s = ev.stale_rows(fractions=(0.0, 0.1, 0.3), side=6, seeds=C.SWEEP_SEEDS[:2], pairs=60)
    assert s[0]["wrong"] == 0.0 and s[0]["wrong"] < s[1]["wrong"] < s[2]["wrong"] and s[0]["rebuild"] > 0


def test_reweight_changes_a_share_of_the_streets_symmetrically():
    g = sc.city_network(8, 3).graph
    g2 = ev.reweight(g, 0.2, 3.0, 1)
    assert np.array_equal(g.indices, g2.indices) and (g2.weight >= g.weight).all() and 0 < (g2.weight != g.weight).mean() < 0.5
    src = np.repeat(np.arange(g.n), g.degree())
    w = {(int(u), int(v)): x for u, v, x in zip(src, g2.indices, g2.weight)}
    assert all(w[(v, u)] == x for (u, v), x in w.items())                            # ungerichtet: beide Richtungen gleich
    assert np.array_equal(ev.reweight(g, 0.0, 3.0, 1).weight, g.weight)
