"""Labels, Abfrage im Vergleich mit Dijkstra und der CH-Abfrage: Kennzahlen, Bildfolge der Verschmelzung, Verteilung über Zufallspaare und die Messreihen der Experimente (Größe, Ordnung, Konstruktion, Abfrage).
Aufwand als Zähler (Einträge, Vergleichsschritte, festgelegte Knoten); Laufzeiten stehen nur als Messwerte in der App."""

import time
from dataclasses import dataclass

import numpy as np

import hl_ch as ch
import hl_constants as C
import hl_labels as hl
from hl_scenario import make_network


@dataclass(frozen=True)
class Analysis:
    net: object
    h: ch.Hierarchy
    order_kind: str
    L: hl.Labels
    raw: hl.Labels
    clean: hl.Labels
    s: int
    t: int
    q: hl.LabelQuery
    route: list
    metrics: dict
    seconds: dict


def pick_pair(net, distance_pct=C.DEFAULT_DISTANCE, seed=C.DEFAULT_SEED):
    """Start und Ziel: bei Netzen mit fester Aufgabe diese; sonst ein Start (Netze mit Karte: nahe bei 30 % Breite und 50 % Höhe; sonst zufällig) und als Ziel der Knoten, dessen Entfernung vom Start in der Rangfolge
    aller erreichbaren Knoten bei `distance_pct` Prozent liegt."""
    if net.fixed_pair:
        return net.fixed_pair
    g = net.graph
    rng = np.random.default_rng([int(seed), 808])
    if net.geometric:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        s = int(np.argmin(np.hypot(*(g.xy - (lo + (hi - lo) * np.array([0.3, 0.5]))).T)))
    else:
        s = int(rng.integers(0, g.n))
    d = hl.dijkstra(g, s)[0]
    reachable = np.where(np.isfinite(d))[0]
    reachable = reachable[reachable != s]
    order = reachable[np.argsort(d[reachable], kind="stable")]
    t = int(order[min(len(order) - 1, int(round(distance_pct / 100.0 * (len(order) - 1))))])
    return s, t


def build_labels(net, order_kind=C.DEFAULT_ORDER, seed=C.DEFAULT_SEED):
    """CH, Labels in der gewünschten Ordnung (PLL) sowie, zum Vergleich, die rohen und bereinigten Labels aus den CH-Suchräumen. Rückgabe: (Hierarchie, Labels, roh, bereinigt, Sekunden)."""
    g = net.graph
    t0 = time.perf_counter()
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    t_ch = time.perf_counter() - t0
    t0 = time.perf_counter()
    L = hl.pll_labels(g, hl.make_order(g, order_kind, h, seed))
    t_pll = time.perf_counter() - t0
    t0 = time.perf_counter()
    raw = hl.ch_labels(g, h)
    clean = hl.clean_labels(hl.ch_labels(g, h))
    t_clean = time.perf_counter() - t0
    return h, L, raw, clean, {"ch": t_ch, "pll": t_pll, "clean": t_clean}


def pair_samples(g, h, L, n_pairs=C.PAIR_SAMPLES, seed=0):
    """Zufallspaare: Vergleichsschritte der Labels, festgelegte Knoten von CH und Dijkstra, und ob alle Antworten übereinstimmen."""
    rng = np.random.default_rng([int(seed), 909])
    su, sc, sm, exact = [], [], [], True
    for _ in range(n_pairs):
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if s == t:
            continue
        dist, order, _ = hl.dijkstra(g, s, t)
        cq = ch.ch_query(h, s, t)
        lq = hl.label_query(L, s, t)
        su.append(len(order))
        sc.append(cq.settled)
        sm.append(lq.steps)
        exact &= bool(abs(lq.dist - dist[t]) < 1e-9 and abs(cq.cost - dist[t]) < 1e-9) if np.isfinite(dist[t]) else bool(lq.dist == float("inf"))
    return {"dijkstra": float(np.mean(su)), "ch": float(np.mean(sc)), "merge": float(np.mean(sm)), "dijkstra_max": int(max(su)), "ch_max": int(max(sc)), "merge_max": int(max(sm)), "exact": exact, "n": len(su)}


def analyse(net, order_kind=C.DEFAULT_ORDER, distance_pct=C.DEFAULT_DISTANCE, seed=C.DEFAULT_SEED, built=None):
    g = net.graph
    h, L, raw, clean, seconds = built if built is not None else build_labels(net, order_kind, seed)
    s, t = pick_pair(net, distance_pct, seed)
    dist, order, relax = hl.dijkstra(g, s, t)
    cq = ch.ch_query(h, s, t)
    t0 = time.perf_counter()
    q = hl.label_query(L, s, t, trace=True)
    seconds = {**seconds, "query": time.perf_counter() - t0}
    route = hl.label_route(L, s, t) or []
    sizes = L.sizes()
    reachable = bool(np.isfinite(dist[t]))
    ps = pair_samples(g, h, L, seed=seed)
    same = all(clean.f_key[v] == L.f_key[v] for v in range(g.n)) if order_kind == "ch" else None
    m = {"n": g.n, "m": g.m // (1 if g.directed else 2), "reachable": reachable, "dist": float(dist[t]), "exact": bool(reachable and abs(q.dist - dist[t]) < 1e-9 and abs(cq.cost - dist[t]) < 1e-9),
         "hub": q.hub, "steps": q.steps, "settled_dijkstra": len(order), "settled_ch": cq.settled, "hops": len(route) - 1 if route else 0,
         "size_s": len(L.f_key[s]), "size_t": len(L.b_key[t]), "avg": float(sizes.mean()), "median": float(np.median(sizes)), "max": int(sizes.max()), "total": int(sizes.sum()), "table": g.n * g.n,
         "raw_avg": raw.total() / g.n, "clean_avg": clean.total() / g.n, "clean_equals_pll": same, "n_shortcuts": h.n_shortcuts,
         "ch_witness_settled": h.counters["witness_settled"], "pll_settled": L.counters["settled"], "pll_pruned": L.counters["pruned"], "pll_searches": L.counters["searches"],
         "raw_settled": raw.counters["settled"], "removed": clean.counters["removed"], "pairs": ps}
    return Analysis(net, h, order_kind, L, raw, clean, s, t, q, route, m, seconds)


def verdict(a):
    """Code für die App: unreachable / ok."""
    return "ok" if a.metrics["reachable"] else "unreachable"


def frames(a, max_frames=60):
    """Welche Zahlen von Vergleichsschritten gezeigt werden (0 bis Ende, höchstens `max_frames` + 1 Bilder)."""
    n = len(a.q.trace)
    if n <= max_frames:
        return list(range(n + 1))
    return sorted({round(i * n / max_frames) for i in range(max_frames + 1)} | {0, n})


def state_at(a, k):
    """Nach k Vergleichsschritten: (Zahl der schon betrachteten Hubs von s, von t, gemeinsame Hubs bis dahin als Liste (Hub-Knoten, Summe), beste Summe bisher)."""
    if k <= 0:
        return 0, 0, [], float("inf")
    tr = a.q.trace[:k]
    i, j, _, best = tr[-1]
    ks, kt = a.L.f_key[a.s], a.L.b_key[a.t]
    # nach dem letzten Schritt sind die Zeiger um eins weiter, wenn die Keys gleich waren bzw. der kleinere Zeiger vorrückte
    ka, kb = ks[i], kt[j]
    i2, j2 = (i + 1, j + 1) if ka == kb else ((i + 1, j) if ka < kb else (i, j + 1))
    common = []
    for ii, jj, match, _ in tr:
        if match:
            common.append((a.L.order[ks[ii]], a.L.f_dist[a.s][ii] + a.L.b_dist[a.t][jj]))
    return i2, j2, common, best


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------------------

def _mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


SIZE_CASES = [("city", 6), ("city", 10), ("city", 14), ("city", 20), ("random", 50), ("random", 100), ("random", 200), ("random", 300)]


def size_rows(cases=SIZE_CASES, seeds=C.SWEEP_SEEDS):
    """Größe der Labels (CH-Ordnung, PLL) gegen die Größe des Netzes: mittlere und größte Labelgröße und Einträge im Verhältnis zur Entfernungstabelle n²; Mittel über die Sweep-Netze."""
    rows = []
    for key, size in cases:
        acc = []
        for sd in seeds:
            net = make_network(key, side=size, nodes=size, seed=sd)
            g = net.graph
            L = hl.pll_labels(g, hl.make_order(g, "ch", ch.build_hierarchy(g, "lazy_edge_difference"), sd))
            sz = L.sizes()
            acc.append({"n": g.n, "avg": float(sz.mean()), "max": float(sz.max()), "share": L.total() / (g.n * g.n)})
        rows.append({"key": key, "size": size, **_mean(acc)})
    return rows


ORDER_CASES = [("city", 10), ("city", 20), ("random", 200)]


def order_rows(cases=ORDER_CASES, seeds=C.SWEEP_SEEDS):
    """Mittlere Labelgröße je Knotenordnung (CH, Grad, Zufall); Mittel über die Sweep-Netze."""
    rows = []
    for key, size in cases:
        acc = []
        for sd in seeds:
            net = make_network(key, side=size, nodes=size, seed=sd)
            g = net.graph
            h = ch.build_hierarchy(g, "lazy_edge_difference")
            acc.append({kind: hl.pll_labels(g, hl.make_order(g, kind, h, sd)).total() / g.n for kind in hl.ORDERS})
        rows.append({"key": key, "size": size, **_mean(acc)})
    return rows


def construction_rows(cases=ORDER_CASES, seeds=C.SWEEP_SEEDS):
    """Zwei Konstruktionen aus derselben CH-Ordnung: rohe CH-Suchräume, bereinigt, und PLL. Mittlere Labelgröße, Anteil der Netze, in denen bereinigt und PLL dieselben Labels ergeben, und der Aufwand (festgelegte Knoten der Suchen)."""
    rows = []
    for key, size in cases:
        acc = []
        for sd in seeds:
            net = make_network(key, side=size, nodes=size, seed=sd)
            g = net.graph
            h = ch.build_hierarchy(g, "lazy_edge_difference")
            raw = hl.ch_labels(g, h)
            clean = hl.clean_labels(hl.ch_labels(g, h))
            L = hl.pll_labels(g, hl.make_order(g, "ch", h))
            acc.append({"raw": raw.total() / g.n, "clean": clean.total() / g.n, "pll": L.total() / g.n, "same": float(all(clean.f_key[v] == L.f_key[v] and clean.f_dist[v] == L.f_dist[v] for v in range(g.n))),
                        "raw_settled": raw.counters["settled"], "pll_settled": L.counters["settled"], "n": g.n})
        rows.append({"key": key, "size": size, **_mean(acc)})
    return rows


def query_rows(cases=SIZE_CASES, seeds=C.SWEEP_SEEDS[:3]):
    """Abfrage gegen Netzgröße: festgelegte Knoten bei Dijkstra und CH, Vergleichsschritte der Labels (Mittel über 100 Zufallspaare je Netz)."""
    rows = []
    for key, size in cases:
        acc = []
        for sd in seeds:
            net = make_network(key, side=size, nodes=size, seed=sd)
            g = net.graph
            h = ch.build_hierarchy(g, "lazy_edge_difference")
            L = hl.pll_labels(g, hl.make_order(g, "ch", h, sd))
            ps = pair_samples(g, h, L, seed=sd)
            assert ps["exact"]
            acc.append({"n": g.n, "dijkstra": ps["dijkstra"], "ch": ps["ch"], "merge": ps["merge"]})
        rows.append({"key": key, "size": size, **_mean(acc)})
    return rows


DENSITY_DEGREES = (2.0, 3.0, 4.0, 6.0)


def density_rows(degrees=DENSITY_DEGREES, nodes=200, seeds=C.SWEEP_SEEDS):
    """Mittlere Labelgröße im Zufallsnetz gegen den mittleren Grad (dichtere Netze brauchen größere Labels); Mittel über die Sweep-Netze."""
    rows = []
    for deg in degrees:
        acc = []
        for sd in seeds:
            g = make_network("random", nodes=nodes, degree=deg, seed=sd).graph
            acc.append({"avg": hl.pll_labels(g, hl.make_order(g, "ch", ch.build_hierarchy(g, "lazy_edge_difference"), sd)).total() / g.n})
        rows.append({"degree": deg, **_mean(acc)})
    return rows


def reweight(g, fraction, factor, seed=0):
    """Neue Kosten (der "Verkehr" hat sich geändert): ein Anteil der Straßen wird mit `factor` multipliziert und auf ganze Zahlen gerundet (bei ungerichteten Graphen beide Richtungen gleich). Rückgabe: neuer Graph."""
    from hl_graph import Graph
    rng = np.random.default_rng([int(seed), 1313])
    src = np.repeat(np.arange(g.n), g.degree())
    key = np.minimum(src, g.indices) * g.n + np.maximum(src, g.indices) if not g.directed else np.arange(g.m)
    uniq = np.unique(key)
    hit = set(uniq[rng.random(len(uniq)) < fraction].tolist())
    mask = np.array([k in hit for k in key.tolist()], dtype=bool)
    w = g.weight.copy()
    w[mask] = np.maximum(1.0, np.rint(w[mask] * factor))
    return Graph(g.n, g.indptr, g.indices, w, g.xy, g.names, g.directed)


STALE_FRACTIONS = (0.02, 0.05, 0.1, 0.2)


def stale_rows(fractions=STALE_FRACTIONS, factor=3.0, side=10, seeds=C.SWEEP_SEEDS, pairs=200):
    """Labels nach einer Kostenänderung: Anteil der Zufallspaare, für die die alten Labels eine falsche Entfernung liefern (Stadtnetz, ein Anteil der Straßen wird `factor`-mal so teuer), und Aufwand des Neuaufbaus (festgelegte Knoten der PLL-Suchen)."""
    rows = []
    for frac in fractions:
        acc = []
        for sd in seeds:
            g = make_network("city", side=side, seed=sd).graph
            h = ch.build_hierarchy(g, "lazy_edge_difference")
            order = hl.make_order(g, "ch", h)
            L = hl.pll_labels(g, order)
            g2 = reweight(g, frac, factor, sd)
            L2 = hl.pll_labels(g2, order)
            rng = np.random.default_rng([int(sd), 5150])
            wrong = n = 0
            for _ in range(pairs):
                s, t = (int(x) for x in rng.integers(0, g.n, 2))
                if s == t:
                    continue
                n += 1
                wrong += abs(hl.label_query(L, s, t).dist - hl.dijkstra(g2, s, t)[0][t]) > 1e-9
                assert abs(hl.label_query(L2, s, t).dist - hl.dijkstra(g2, s, t)[0][t]) < 1e-9
            acc.append({"wrong": wrong / n, "rebuild": L2.counters["settled"]})
        rows.append({"fraction": frac, **_mean(acc)})
    return rows
