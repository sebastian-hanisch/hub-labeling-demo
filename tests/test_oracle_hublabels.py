"""Unabhängiges Orakel: Floyd-Warshall (eigene Schleife, keine Bibliothek) auf Zufallsgraphen mit Nullkosten, Parallelkanten,
Selbstschleifen, gleichen Entfernungen und unerreichbaren Zielen. Geprüft für jede Konstruktion und JEDES Paar: Entfernung,
Hub liegt auf einer kürzesten Route, Route echt, kanonische Labels (auch bei Nullkosten), CH-Abfrage.
Die große Fassung (320 Graphen, 130 000 Paar-Abfragen) lief im Scratchpad."""

import random

import numpy as np

import hl_ch as ch
import hl_labels as hl
from hl_graph import from_arcs, route_cost

INF = float("inf")


def _floyd_warshall(n, arcs, directed):
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0.0
    for u, v, w in arcs:
        if u != v:
            d[u][v] = min(d[u][v], w)
            if not directed:
                d[v][u] = min(d[v][u], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def _graphs(count, seed):
    rng = random.Random(seed)
    for it in range(count):
        n = rng.choice([1, 2, 4, 7, 11])
        directed = rng.random() < 0.5
        zero = rng.random() < 0.4
        arcs = [(rng.randrange(n), rng.randrange(n), float(rng.randint(0 if zero else 1, rng.choice([1, 2, 5, 9]))))
                for _ in range(rng.randrange(0, 3 * n + 1) if n > 1 else 0)]
        g = from_arcs(n, arcs, np.random.default_rng(it).random((n, 2)), directed=directed, clean=rng.random() < 0.6)
        yield it, n, directed, zero, arcs, g, _floyd_warshall(n, arcs, directed)


def test_every_construction_matches_floyd_warshall_for_all_pairs():
    for it, n, directed, zero, arcs, g, ref in _graphs(40, 99):
        h = ch.build_hierarchy(g, "lazy_edge_difference")
        constructions = {"raw": hl.ch_labels(g, h), "clean": hl.clean_labels(hl.ch_labels(g, h))}
        for kind in hl.ORDERS:
            constructions["pll_" + kind] = hl.pll_labels(g, hl.make_order(g, kind, h, it))
        for name, L in constructions.items():
            for s in range(n):
                for t in range(n):
                    q = hl.label_query(L, s, t)
                    assert q.dist == ref[s][t], (it, name, s, t)
                    if q.dist < INF:
                        assert ref[s][q.hub] + ref[q.hub][t] == ref[s][t]  # der gemeldete Hub liegt auf einer kürzesten Route
                    else:
                        assert q.hub == -1
        for s in range(n):
            for t in range(n):
                assert ch.ch_query(h, s, t).cost == ref[s][t], (it, s, t)


def test_pll_labels_are_canonical_and_routes_are_real_also_with_zero_costs():
    for it, n, directed, zero, arcs, g, ref in _graphs(30, 7):
        h = ch.build_hierarchy(g, "lazy_edge_difference")
        for kind in hl.ORDERS:
            L = hl.pll_labels(g, hl.make_order(g, kind, h, it))
            pos = L.pos
            for v in range(n):
                exp_f = sorted(x for x in range(n) if ref[v][x] < INF and not any(
                    pos[y] < pos[x] and ref[v][y] + ref[y][x] == ref[v][x] for y in range(n) if y != x))
                assert sorted(L.order[k] for k in L.f_key[v]) == exp_f, (it, kind, v)
                exp_b = sorted(x for x in range(n) if ref[x][v] < INF and not any(
                    pos[y] < pos[x] and ref[x][y] + ref[y][v] == ref[x][v] for y in range(n) if y != x))
                assert sorted(L.order[k] for k in L.b_key[v]) == exp_b, (it, kind, v)
            for s in range(n):
                for t in range(n):
                    r = hl.label_route(L, s, t)
                    if ref[s][t] == INF:
                        assert r == []
                    else:
                        assert r[0] == s and r[-1] == t and route_cost(g, r) == ref[s][t]


def test_cleaned_ch_labels_equal_pll_for_positive_costs():
    checked = 0
    for it, n, directed, zero, arcs, g, ref in _graphs(60, 99):
        if any(w <= 0 for _, _, w in arcs):
            continue  # bei Nullkosten behält das Bereinigen den Eintrag (v, 0), den PLL streicht
        h = ch.build_hierarchy(g, "lazy_edge_difference")
        a, b = hl.clean_labels(hl.ch_labels(g, h)), hl.pll_labels(g, hl.make_order(g, "ch", h))
        for v in range(n):
            assert (a.f_key[v], a.f_dist[v], a.b_key[v], a.b_dist[v]) == (b.f_key[v], b.f_dist[v], b.b_key[v], b.b_dist[v])
        checked += 1
    assert checked >= 15
