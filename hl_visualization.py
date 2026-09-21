"""Plotly-Abbildungen: Karte mit den Hubs von Start und Ziel und dem Verschmelzen ihrer Listen, Verteilung der Labelgrößen, Tabellen für kleine Netze, Experimente. Achsen sind gesperrt (fixedrange),
damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import hl_constants as C
from hl_evaluation import state_at


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.16), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segments(g):
    """Alle Kanten als eine Linienspur (None trennt die Segmente); bei ungerichteten Netzen Hin- und Rückrichtung nur einmal."""
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    keep = np.ones(len(src), dtype=bool)
    if not g.directed:
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        first = np.zeros(len(src), dtype=bool)
        _, idx = np.unique(lo * g.n + hi, return_index=True)
        first[idx] = True
        keep &= first
    u, v = src[keep], dst[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def name_of(g, v):
    return g.names[v] if g.names else f"{v}"


def build_network(a, k, height=480):
    """Die Karte nach `k` Vergleichsschritten beim Verschmelzen der Listen von Start und Ziel: die Hubs von s (blau) und von t (rot) als Ringe, schon verglichene ausgefüllt; gemeinsame Hubs orange; am Ende der Hub der
    kürzesten Route (Stern) und die Route selbst."""
    net, g, L = a.net, a.net.graph, a.L
    small = bool(g.names)
    final = k >= len(a.q.trace)
    i2, j2, common, best = state_at(a, k)
    ks, kt = L.f_key[a.s], L.b_key[a.t]
    hubs_s = [L.order[x] for x in ks]
    hubs_t = [L.order[x] for x in kt]
    fig = go.Figure()
    if net.geometric:
        ex, ey = _segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.45)", width=1), hoverinfo="skip", showlegend=False))
    size = 15 if small else (5 if g.n > 300 else 8)
    others = np.setdiff1d(np.arange(g.n), np.array(sorted(set(hubs_s) | set(hubs_t) | {a.s, a.t}), dtype=int))
    if small:
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers+text", text=[g.names[v] for v in range(g.n)], textposition="top center", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=size, color="white", line=dict(color="gray", width=1.5))))
    elif len(others):
        fig.add_trace(go.Scatter(x=g.xy[others, 0], y=g.xy[others, 1], mode="markers", showlegend=False, hoverinfo="skip", marker=dict(size=size, color="rgba(200,200,200,0.9)")))
    if final and a.route:
        pts = g.xy[a.route]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="kürzeste Route", line=dict(color=C.COLORS["route"], width=5), hoverinfo="skip"))
    for hubs, done, color, name in ((hubs_s, i2, C.COLORS["s"], "Hubs von s"), (hubs_t, j2, C.COLORS["t"], "Hubs von t")):
        seen, todo = hubs[:done], hubs[done:]
        if todo:
            fig.add_trace(go.Scatter(x=g.xy[todo, 0], y=g.xy[todo, 1], mode="markers", name=f"{name} (noch nicht verglichen)", hoverinfo="skip", marker=dict(size=size + 4, color="rgba(255,255,255,0)", line=dict(color=color, width=2))))
        if seen:
            fig.add_trace(go.Scatter(x=g.xy[seen, 0], y=g.xy[seen, 1], mode="markers", name=f"{name} (verglichen)", hoverinfo="skip", marker=dict(size=size + 2, color=color, opacity=0.65)))
    if common:
        cn = [c[0] for c in common]
        fig.add_trace(go.Scatter(x=g.xy[cn, 0], y=g.xy[cn, 1], mode="markers", name="gemeinsame Hubs", customdata=[c[1] for c in common], hovertemplate="Summe %{customdata:g}<extra></extra>",
                                 marker=dict(size=size + 6, color=C.COLORS["hub"], symbol="diamond", line=dict(color="white", width=1))))
    if final and a.q.hub >= 0:
        fig.add_trace(go.Scatter(x=[g.xy[a.q.hub, 0]], y=[g.xy[a.q.hub, 1]], mode="markers", name=f"Hub der Route: {name_of(g, a.q.hub)}", hoverinfo="skip", marker=dict(size=size + 12, color="#ffd700", symbol="star", line=dict(color="black", width=1))))
    for node, name, color, symbol in ((a.s, "Start", C.COLORS["start"], "square"), (a.t, "Ziel", C.COLORS["goal"], "x")):
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {name_of(g, node)}", hoverinfo="skip", marker=dict(size=size + 4, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if net.key == "city":
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + 1.2 * pad[0]])
        fig.update_yaxes(range=[lo[1] - 1.8 * pad[1], hi[1] + 1.8 * pad[1]])
    return _base(fig, height)


def label_table(a):
    """Die Labels aller Orte (kleines Netz): Ort, Hubs mit Entfernung in der Reihenfolge der Wichtigkeit, Größe."""
    g, L = a.net.graph, a.L
    rows = []
    for v in range(g.n):
        rows.append({"Ort": g.names[v], "Hubs (Entfernung)": ", ".join(f"{g.names[x]} ({d:g})" for x, d in L.hubs(v)), "Einträge": len(L.f_key[v])})
    return rows


def merge_table(a, k, last=8):
    """Die letzten `last` Vergleichsschritte bis Schritt k: (Schritt, Vergleich der beiden Hubs, beste Summe bisher)."""
    g, L = a.net.graph, a.L
    ks, ds = L.f_key[a.s], L.f_dist[a.s]
    kt, dt = L.b_key[a.t], L.b_dist[a.t]
    rows = []
    for idx in range(max(0, k - last), k):
        i, j, match, best = a.q.trace[idx]
        left = f"{name_of(g, L.order[ks[i]])} ({ds[i]:g})"
        right = f"{name_of(g, L.order[kt[j]])} ({dt[j]:g})"
        rows.append({"Schritt": idx + 1, "Hub von s ↔ Hub von t": f"{left} ↔ {right}" + (f" - gleich: Summe {ds[i] + dt[j]:g}" if match else ""), "Beste Summe": f"{best:g}" if np.isfinite(best) else "–"})
    return rows


def build_sizes(a, height=300):
    """Verteilung der Labelgrößen über alle Knoten; Start und Ziel markiert."""
    sizes = a.L.sizes()
    fig = go.Figure(go.Histogram(x=sizes, marker_color="#7f7f7f", name="Knoten"))
    for node, name, color in ((a.s, "Start", C.COLORS["s"]), (a.t, "Ziel", C.COLORS["t"])):
        fig.add_vline(x=int(sizes[node]), line=dict(color=color, dash="dash", width=2), annotation_text=name, annotation_position="top")
    fig.update_layout(xaxis_title="Einträge im Label eines Knotens", yaxis_title="Knoten", showlegend=False)
    return _base(fig, height)


def build_size(rows):
    """Mittlere Labelgröße gegen die Knotenzahl, Stadtnetz und Zufallsnetz."""
    fig = go.Figure()
    for key, name in (("city", "Stadtnetz"), ("random", "Zufallsnetz")):
        r = [x for x in rows if x["key"] == key]
        fig.add_trace(go.Scatter(x=[x["n"] for x in r], y=[x["avg"] for x in r], mode="lines+markers", name=f"{name}: Mittel"))
        fig.add_trace(go.Scatter(x=[x["n"] for x in r], y=[x["max"] for x in r], mode="lines+markers", name=f"{name}: größtes Label", line=dict(dash="dot")))
    fig.update_layout(xaxis_title="Knoten", yaxis_title="Einträge je Label")
    return _base(fig, 320)


def build_orders(rows):
    fig = go.Figure()
    x = [f"{'Stadtnetz' if r['key'] == 'city' else 'Zufallsnetz'} {r['size'] if r['key'] == 'random' else str(r['size']) + '×' + str(r['size'])}" for r in rows]
    for kind, name in (("ch", "CH-Ordnung"), ("degree", "Grad"), ("random", "Zufall")):
        fig.add_trace(go.Bar(x=x, y=[r[kind] for r in rows], name=name))
    fig.update_layout(barmode="group", yaxis_title="mittlere Einträge je Label")
    return _base(fig, 320)


def build_construction(rows):
    fig = go.Figure()
    x = [f"{'Stadtnetz' if r['key'] == 'city' else 'Zufallsnetz'} {r['size'] if r['key'] == 'random' else str(r['size']) + '×' + str(r['size'])}" for r in rows]
    for kind, name in (("raw", "CH-Suchräume (roh)"), ("clean", "bereinigt"), ("pll", "Pruned Landmark Labeling")):
        fig.add_trace(go.Bar(x=x, y=[r[kind] for r in rows], name=name))
    fig.update_layout(barmode="group", yaxis_title="mittlere Einträge je Label")
    return _base(fig, 320)


def build_query(rows):
    """Festgelegte Knoten bei Dijkstra und CH, Vergleichsschritte der Labels, gegen die Knotenzahl (logarithmisch)."""
    fig = go.Figure()
    for key, name, dash in (("city", "Stadtnetz", "solid"), ("random", "Zufallsnetz", "dot")):
        r = [x for x in rows if x["key"] == key]
        for col, label in (("dijkstra", "Dijkstra"), ("ch", "CH"), ("merge", "Labels")):
            fig.add_trace(go.Scatter(x=[x["n"] for x in r], y=[x[col] for x in r], mode="lines+markers", name=f"{label} ({name})", line=dict(dash=dash)))
    fig.update_layout(xaxis_title="Knoten", yaxis_title="Schritte einer Abfrage")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _base(fig, 340)


def build_stale(rows):
    fig = go.Figure(go.Bar(x=[f"{r['fraction']:.0%}" for r in rows], y=[100 * r["wrong"] for r in rows], marker_color="#d62728"))
    fig.update_layout(xaxis_title="teurere Straßen (dreifache Kosten)", yaxis_title="falsche Antworten alter Labels [%]")
    return _base(fig, 300)
