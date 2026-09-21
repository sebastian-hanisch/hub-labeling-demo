"""Hub Labeling gegen unabhängige Referenzen: networkx-Allpaar-Entfernungen für jede Konstruktion, jede Ordnung, gerichtete und ungerichtete Graphen; die kanonische Eigenschaft der Labels wird aus den Entfernungen nachgerechnet."""

import networkx as nx
import numpy as np
import pytest

import hl_ch as ch
import hl_labels as hl
from hl_graph import from_arcs, route_cost

INF = float("inf")


def _random_graph(n, m, seed, directed=False, zero=False):
    """Zusammenhängender (bei gerichteten Graphen: stark zusammenhängender) Zufallsgraph mit ganzzahligen Kosten 1 bis 9 (mit `zero`: auch 0)."""
    rng = np.random.default_rng(seed)
    lo = 0 if zero else 1
    arcs = {(int(rng.integers(0, i)), i): float(rng.integers(lo, 10)) for i in range(1, n)}
    if directed:
        arcs.update({(i, int(rng.integers(0, i))): float(rng.integers(lo, 10)) for i in range(1, n)})
    while len(arcs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs and (directed or (v, u) not in arcs):
            arcs[(u, v)] = float(rng.integers(lo, 10))
    return from_arcs(n, [(u, v, w) for (u, v), w in arcs.items()], rng.random((n, 2)), directed=directed)


def _apsp(g):
    G = nx.DiGraph() if g.directed else nx.Graph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            G.add_edge(u, v, weight=w)
    ref = dict(nx.all_pairs_dijkstra_path_length(G))
    return [[ref[s].get(t, INF) for t in range(g.n)] for s in range(g.n)]


CASES = [(n, m, seed) for n, m in ((6, 10), (12, 22), (25, 60)) for seed in range(4)]


def _all_labels(g, h, seed):
    out = {"raw": hl.ch_labels(g, h), "clean": hl.clean_labels(hl.ch_labels(g, h))}
    for kind in hl.ORDERS:
        out[f"pll_{kind}"] = hl.pll_labels(g, hl.make_order(g, kind, h, seed))
    return out


# --- Richtigkeit gegen alle Paare ------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("n,m,seed", CASES)
def test_every_construction_answers_every_pair_exactly(directed, n, m, seed):
    g = _random_graph(n, m, seed, directed)
    ref = _apsp(g)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    for name, L in _all_labels(g, h, seed).items():
        for s in range(n):
            for t in range(n):
                assert hl.label_query(L, s, t).dist == pytest.approx(ref[s][t]), (name, s, t)


@pytest.mark.parametrize("directed", (False, True))
def test_zero_cost_edges_and_parallel_arcs_do_not_disturb(directed):
    g = _random_graph(14, 26, 3, directed, zero=True)
    ref = _apsp(g)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    for name, L in _all_labels(g, h, 3).items():
        for s in range(g.n):
            for t in range(g.n):
                assert hl.label_query(L, s, t).dist == pytest.approx(ref[s][t]), (name, s, t)
    raw = from_arcs(3, [(0, 1, 5.0), (0, 1, 2.0), (1, 2, 0.0)], np.zeros((3, 2)), directed=True, clean=False)             # Parallelkanten bleiben (billigste gilt)
    L = hl.pll_labels(raw, [0, 1, 2])
    assert hl.label_query(L, 0, 2).dist == 2.0 and hl.label_query(L, 2, 0).dist == INF


def test_unreachable_pairs_and_the_same_node():
    g = from_arcs(4, [(0, 1, 1.0), (1, 2, 2.0)], np.zeros((4, 2)), directed=True)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    for name, L in _all_labels(g, h, 0).items():
        assert hl.label_query(L, 0, 2).dist == 3.0 and hl.label_query(L, 2, 0).dist == INF and hl.label_query(L, 0, 3).dist == INF, name
        assert hl.label_query(L, 3, 3).dist == 0.0 and hl.label_query(L, 3, 3).hub == 3, name
    q = hl.label_query(hl.pll_labels(g, [0, 1, 2, 3]), 2, 0)
    assert q.hub == -1 and hl.label_route(hl.pll_labels(g, [0, 1, 2, 3]), 2, 0) == []


# --- Aufbau der Labels ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("n,m,seed", CASES)
def test_cleaned_ch_labels_are_exactly_the_pll_labels_in_the_ch_order(directed, n, m, seed):
    g = _random_graph(n, m, seed, directed)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    clean = hl.clean_labels(hl.ch_labels(g, h))
    pll = hl.pll_labels(g, hl.make_order(g, "ch", h))
    for v in range(n):
        assert clean.f_key[v] == pll.f_key[v] and clean.f_dist[v] == pll.f_dist[v] and clean.b_key[v] == pll.b_key[v] and clean.b_dist[v] == pll.b_dist[v]
    assert hl.ch_labels(g, h).total() >= clean.total()


@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("kind", hl.ORDERS)
def test_pll_labels_are_canonical(directed, kind):
    """x steht im Vorwärtslabel von v genau dann, wenn kein wichtigerer Knoten y (kleinere Position) auf einer kürzesten Route v -> x liegt (also d(v, y) + d(y, x) = d(v, x)); Entfernungen exakt."""
    g = _random_graph(14, 26, 5, directed)
    ref = _apsp(g)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    L = hl.pll_labels(g, hl.make_order(g, kind, h, 5))
    pos = L.pos
    for v in range(g.n):
        expected_f = [x for x in range(g.n) if ref[v][x] < INF and not any(pos[y] < pos[x] and ref[v][y] + ref[y][x] == ref[v][x] for y in range(g.n) if y != x)]
        assert sorted(L.order[k] for k in L.f_key[v]) == sorted(expected_f), (v, kind)
        assert all(d == ref[v][L.order[k]] for k, d in zip(L.f_key[v], L.f_dist[v]))
        expected_b = [x for x in range(g.n) if ref[x][v] < INF and not any(pos[y] < pos[x] and ref[x][y] + ref[y][v] == ref[x][v] for y in range(g.n) if y != x)]
        assert sorted(L.order[k] for k in L.b_key[v]) == sorted(expected_b), (v, kind)


@pytest.mark.parametrize("directed", (False, True))
def test_labels_are_sorted_contain_the_node_itself_and_undirected_labels_are_shared(directed):
    g = _random_graph(20, 40, 2, directed)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    for L in _all_labels(g, h, 2).values():
        for v in range(g.n):
            for keys in ((L.f_key[v], L.b_key[v])):
                assert keys == sorted(set(keys))
            assert L.pos[v] in L.f_key[v] and L.pos[v] in L.b_key[v] and L.f_dist[v][L.f_key[v].index(L.pos[v])] == 0.0
        assert (L.f_key is L.b_key) != directed
        assert all(L.order[L.pos[v]] == v for v in range(g.n)) and sorted(L.order) == list(range(g.n))


def test_the_labels_are_never_larger_than_the_number_of_nodes_and_the_hub_is_at_least_as_important_as_the_node():
    g = _random_graph(25, 60, 1)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    L = hl.pll_labels(g, hl.make_order(g, "ch", h))
    assert all(len(L.f_key[v]) <= g.n and all(k <= L.pos[v] for k in L.f_key[v]) for v in range(g.n)) and L.sizes().sum() == L.total()


# --- Abfrage und Route ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("n,m,seed", CASES)
def test_route_is_a_real_route_with_the_reported_cost(directed, n, m, seed):
    g = _random_graph(n, m, seed, directed)
    ref = _apsp(g)
    L = hl.pll_labels(g, hl.make_order(g, "ch", ch.build_hierarchy(g, "lazy_edge_difference")))
    for s in range(n):
        for t in range(n):
            r = hl.label_route(L, s, t)
            assert r[0] == s and r[-1] == t and route_cost(g, r) == pytest.approx(ref[s][t]) and hl.label_query(L, s, t).hub in r


def test_merge_steps_are_bounded_by_the_two_label_sizes_and_the_trace_is_consistent():
    g = _random_graph(25, 60, 4)
    L = hl.pll_labels(g, hl.make_order(g, "ch", ch.build_hierarchy(g, "lazy_edge_difference")))
    for s in range(0, 25, 3):
        for t in range(1, 25, 4):
            q = hl.label_query(L, s, t, trace=True)
            assert q.steps == len(q.trace) <= len(L.f_key[s]) + len(L.b_key[t]) and q.steps >= 1
            assert [b for *_, b in q.trace] == sorted((b for *_, b in q.trace), reverse=True) and q.trace[-1][3] == q.dist
            assert sum(1 for *_, match, _ in q.trace if match) >= 1
    assert hl.label_query(L, 3, 9).trace == []


def test_route_needs_pll_labels():
    g = _random_graph(10, 18, 1)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    assert hl.label_route(hl.clean_labels(hl.ch_labels(g, h)), 0, 5) is None


def test_ordering_kinds_are_permutations_and_reproducible():
    g = _random_graph(20, 40, 2)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    for kind in hl.ORDERS:
        o = hl.make_order(g, kind, h, 3)
        assert sorted(o) == list(range(20)) and o == hl.make_order(g, kind, h, 3)
    assert hl.make_order(g, "ch", h) == list(h.order[::-1])
    deg = g.degree()
    o = hl.make_order(g, "degree", h, 0)
    assert all(deg[o[i]] >= deg[o[i + 1]] for i in range(19))
    with pytest.raises(ValueError):
        hl.make_order(g, "ring", h)


def test_a_wrong_order_only_changes_the_size_never_the_answers():
    g = _random_graph(25, 60, 6)
    ref = _apsp(g)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    sizes = {}
    for kind in hl.ORDERS:
        L = hl.pll_labels(g, hl.make_order(g, kind, h, 6))
        sizes[kind] = L.total()
        assert all(hl.label_query(L, s, t).dist == pytest.approx(ref[s][t]) for s in range(25) for t in range(25))
    assert sizes["random"] > sizes["ch"]


# --- Baseline und CH-Abfrage -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("directed", (False, True))
def test_dijkstra_and_ch_query_match_the_reference(directed):
    g = _random_graph(25, 60, 8, directed)
    ref = _apsp(g)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    for s in range(0, 25, 2):
        d, order, relax = hl.dijkstra(g, s)
        assert list(d) == pytest.approx(ref[s]) and order[0] == s and relax >= len(order) - 1
        for t in range(25):
            assert ch.ch_query(h, s, t).cost == pytest.approx(ref[s][t])
            dt, ot, _ = hl.dijkstra(g, s, t)
            assert dt[t] == pytest.approx(ref[s][t]) and ot[-1] == t and len(ot) <= len(order)
