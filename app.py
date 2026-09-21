"""Hub Labeling - eine Abfrage ist nur noch das Verschmelzen zweier Listen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Hub Labeling - und lässt stattdessen das Beispiel wachsen.
Zehntes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Kind von Contraction Hierarchies: die CH-Abfrage sucht noch aufwärts im Graphen, mit Labels ist die Suche vorweggenommen - der Preis ist Speicher.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import hl_ch as ch
import hl_constants as C
import hl_evaluation as ev
import hl_labels as hl
from hl_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from hl_scenario import make_network
from hl_visualization import (
    build_construction,
    build_network,
    build_orders,
    build_query,
    build_size,
    build_sizes,
    build_stale,
    label_table,
    merge_table,
    name_of,
)

st.set_page_config(page_title="Hub Labeling – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _built(params, order):
    net = _network(params)
    return ev.build_labels(net, order, params[-1])


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params, order, distance):
    net = _network(params)
    return ev.analyse(net, order, distance, params[-1], built=_built(params, order))


@st.cache_data(show_spinner=False)
def _sizes():
    return ev.size_rows(), ev.density_rows()


@st.cache_data(show_spinner=False)
def _orders():
    return ev.order_rows(), ev.construction_rows()


@st.cache_data(show_spinner=False)
def _queries():
    return ev.query_rows()


@st.cache_data(show_spinner=False)
def _stale():
    return ev.stale_rows()


st.title("🏷️ Hub Labeling – jede Abfrage ist ein Listen-Merge")
st.markdown(
    """
Eine Contraction Hierarchy beschleunigt Dijkstra, indem sie nur noch **aufwärts** sucht - aber sie sucht noch im Graphen. Beim **Hub Labeling** wird diese Suche **vorweggenommen**: jeder Knoten bekommt eine kurze Liste von **Hubs** (wichtigen Knoten) samt Entfernung. Zu jedem Paar $(s, t)$
gibt es einen Hub auf einer kürzesten Route, der in **beiden** Listen steht - die Entfernung ist das Minimum von $d(s, h) + d(h, t)$ über die gemeinsamen Hubs. Die Abfrage ist nur noch das **Verschmelzen zweier sortierter Listen**, ohne einen Schritt im Graphen.
Der Preis: die Labels kosten **Speicher**, und sie müssen **neu berechnet werden, wenn sich die Kosten ändern**.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zehntes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Kind von Contraction Hierarchies - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Verfahren geht auf Cohen, Halperin, Kaplan und Zwick (2003) zurück, die Konstruktion aus der CH-Ordnung auf Abraham, Delling, Goldberg und Werneck (2011), das Pruned Landmark Labeling auf Akiba, Iwata und Yoshida (2013); "
    "alle Netze und Zahlen dieser Demo sind eigene Graphen und Messungen."
)

with st.expander("So funktioniert Hub Labeling", expanded=True):
    st.markdown(
        """
1. **Ordnung:** die Knoten werden nach Wichtigkeit geordnet - hier aus der Contraction Hierarchy (der zuletzt zusammengezogene Knoten ist der wichtigste), zum Vergleich nach Grad oder zufällig.
2. **Labels:** das Label von $v$ enthält Knoten $h$ mit der Entfernung $d(v, h)$: $h$ steht darin, wenn $h$ der **wichtigste Knoten auf einer kürzesten Route** von $v$ nach $h$ ist. Damit steht auf jeder kürzesten Route $s \\leadsto t$ ihr wichtigster Knoten in beiden Labels.
3. **Abfrage:** die Listen von $s$ und $t$ sind nach Hub sortiert; beide gleichzeitig durchlaufen, bei gleichem Hub die Summe bilden, das Minimum merken. Höchstens $|L(s)| + |L(t)|$ Vergleichsschritte.
4. **Zwei Wege zu denselben Labels:** die **Suchräume der CH-Aufwärtssuche** enthalten viel Überflüssiges (Einträge, die ein anderer Hub schon abdeckt) und werden bereinigt; das **Pruned Landmark Labeling** baut sie direkt auf, indem jeder Hub ein Dijkstra startet, das dort abbricht, wo die bisherigen Labels die Entfernung schon liefern.
5. **Route:** mit dem nächsten Knoten Richtung Hub je Eintrag lässt sich die Route ohne Suche zusammensetzen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (acht Orte, alle Labels sichtbar) oder erzeugt (Stadtnetz auf einem gestörten Raster, Zufallsnetz). Alle Kosten sind ganze Zahlen; höchstens 400 Knoten.",
    )
    order = st.selectbox("Ordnung der Hubs", tuple(C.ORDER_LABELS), key="order_select", format_func=lambda k: C.ORDER_LABELS[k],
                         help="Welche Knoten die wichtigsten Hubs sind. Die Antworten hängen nicht davon ab - nur die Größe der Labels: im 20 × 20-Stadtnetz (Mittel über fünf Netze) im Mittel 23.7 Einträge je Label mit der CH-Ordnung, 35.8 nach Grad, 40.3 zufällig.")
    if net_key == "city":
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider",
                         help="Größe des Rasters: n = Seite² Knoten. Mittlere Labelgröße (CH-Ordnung, Mittel über fünf Netze) bei 6 / 10 / 14 / 20 Kreuzungen je Seite: 6.8 / 12.4 / 17.0 / 23.7 Einträge - sie wächst viel langsamer als die Knotenzahl.")
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "random":
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=10,
                          help="Anzahl der Knoten n. Mittlere Labelgröße (CH-Ordnung, Mittel über fünf Netze) bei 50 / 100 / 200 / 300 Knoten: 6.1 / 8.7 / 12.8 / 16.0.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5,
                           help="Kanten je Knoten. Mittlere Labelgröße bei 200 Knoten und Grad 2 / 3 / 4 / 6 (Mittel über fünf Netze): 5.5 / 12.8 / 17.1 / 20.5 - dichtere Netze brauchen größere Labels.")
        st.session_state[KEPT["degree_slider"]] = degree
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
    if net_key in ("city", "random"):
        distance = st.slider("Entfernung des Paares [Perzentil]", *bounds("distance_slider"), key="distance_slider", step=5,
                             help="Das Ziel liegt so weit vom Start entfernt, wie es dem Perzentil aller Entfernungen von diesem Start entspricht: 0 = der nächste Knoten, 100 = der am weitesten entfernte.")
        st.session_state[KEPT["distance_slider"]] = distance
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        distance = int(st.session_state.get(KEPT["distance_slider"], C.DEFAULT_DISTANCE))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - Start Altstadt, Ziel Fabrik, es gibt nichts zu erzeugen.")

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, seed=C.DEFAULT_SEED)
if net_key == "city":
    d.update(side=int(side), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), seed=int(seed))
params = (net_key, d["side"], d["nodes"], d["degree"], d["seed"])
dist_pct = int(distance) if net_key in ("city", "random") else C.DEFAULT_DISTANCE
sync_query_params({"net_select": net_key, "order_select": order, "side_slider": int(side), "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1), "distance_slider": int(distance), "seed_input": int(seed)})
with st.spinner("Rechne ..."):
    a = _analysis(params, order, dist_pct)
net, m, g, L = a.net, a.metrics, a.net.graph, a.L
small = bool(g.names)
frame_list = ev.frames(a)
last_step = len(frame_list) - 1
view_key = (params, order, dist_pct)
if st.session_state.get("hl_step_owner") != view_key:
    st.session_state["hl_step_owner"] = view_key
    st.session_state["hl_step"] = last_step

# --- Hub Labeling in Aktion ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Hub Labeling in Aktion")
if small:
    st.markdown("**Die Labels aller Orte** (Hubs in der Reihenfolge der Wichtigkeit, mit Entfernung in Minuten):")
    st.dataframe(pd.DataFrame(label_table(a)), hide_index=True, width="stretch")
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Vergleichsschritte", 0, last_step, key="hl_step",
                     help="Wie viele Schritte des Verschmelzens von Start- und Ziel-Label schon ausgeführt sind: 0 = noch nichts verglichen, ganz rechts = fertig, die kürzeste Route steht. Bei vielen Schritten zeigt die Ansicht etwa 60 Zwischenstände.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(current):
    k = frame_list[current]
    with view_slot.container():
        st.plotly_chart(build_network(a, k, height=400 if small else 460), width="stretch", key=f"net_chart_{current}")
        if k > 0:
            st.dataframe(pd.DataFrame(merge_table(a, k)), hide_index=True, width="stretch")
        if k >= len(a.q.trace) and m["reachable"]:
            unit = net.unit
            st.markdown(f"**Ergebnis:** Entfernung {m['dist']:g} {unit} über den Hub {name_of(g, a.q.hub)} nach {a.q.steps} Vergleichsschritten"
                        + (f"; Route: {' → '.join(name_of(g, v) for v in a.route)}." if small and a.route else f"; die Route hat {m['hops']} Kanten."))


if auto_play:
    n_frames = min(last_step + 1, 40)
    for kk in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames)}):
        _render(kk)
        time.sleep(min(0.7, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note)

st.markdown("---")

# --- Der Preis der Labels ---------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der Preis der Labels")
st.caption("**Zähler** = Schritte und Einträge: Einträge in den Labels, Vergleichsschritte beim Verschmelzen, festgelegte Knoten bei Dijkstra und CH-Abfrage. Sie sind plattformfest. Laufzeiten stehen nur als Messwerte im Vergleich unten.")
if not m["reachable"]:
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar.")
else:
    ps = m["pairs"]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Einträge je Label", f"{m['avg']:.1f}", delta=f"größtes {m['max']}", delta_color="off", help="Mittlere Zahl der Hubs im Label eines Knotens (bei ungerichteten Netzen eine Liste je Knoten) und das größte Label.")
    p2.metric("Einträge insgesamt", _num(m["total"]), delta=f"{m['total'] / m['table']:.0%} von {_num(m['table'])} Entfernungen", delta_color="off", help="Speicher der Labels in Einträgen (Hub, Entfernung), gegen eine Tabelle aller Entfernungen zwischen allen Paaren (n²).")
    p3.metric("Abfrage dieses Paares", f"{m['steps']} Schritte", delta=f"Dijkstra {m['settled_dijkstra']}, CH {m['settled_ch']}", delta_color="off", help="Vergleichsschritte beim Verschmelzen gegen festgelegte Knoten bei Dijkstra und bei der CH-Abfrage.")
    p4.metric("Abfrage im Mittel", f"{ps['merge']:.1f} Schritte", delta=f"Dijkstra {ps['dijkstra']:.1f}, CH {ps['ch']:.1f}", delta_color="off", help=f"Mittel über {ps['n']} zufällige Paare dieses Netzes: Vergleichsschritte gegen festgelegte Knoten.")
    st.success(f"✅ **Ohne einen Schritt im Graphen:** {m['steps']} Vergleichsschritte für {name_of(g, a.s)} → {name_of(g, a.t)} ({m['dist']:g} {net.unit}); Dijkstra hätte {m['settled_dijkstra']} Knoten festgelegt, die CH-Abfrage {m['settled_ch']}. "
               f"Der Preis: {_num(m['total'])} Einträge ({m['avg']:.1f} je Knoten) - und {m['n_shortcuts']} Abkürzungen der CH für die Ordnung. In allen {ps['n']} Zufallspaaren stimmen die Labels mit Dijkstra überein.")
    st.plotly_chart(build_sizes(a), width="stretch", key="hist_chart")
    st.caption(f"Verteilung der Labelgrößen über alle {g.n} Knoten (Median {m['median']:.0f}); Start ({m['size_s']} Einträge) und Ziel ({m['size_t']}) sind markiert.")

st.markdown("---")

# --- Vergleich -------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Dijkstra, CH und Labels im Vergleich; zwei Konstruktionen"):
    if m["reachable"]:
        t0 = time.perf_counter()
        hl.dijkstra(g, a.s, a.t)
        t_dij = time.perf_counter() - t0
        t0 = time.perf_counter()
        ch.ch_query(a.h, a.s, a.t)
        t_ch = time.perf_counter() - t0
        st.table({"Verfahren": ["Dijkstra (keine Vorberechnung)", "Contraction Hierarchies", "Hub Labeling"],
                  "Abfrage (dieses Paar)": [f"{m['settled_dijkstra']} festgelegte Knoten", f"{m['settled_ch']} festgelegte Knoten", f"{m['steps']} Vergleichsschritte"],
                  "Laufzeit [ms]": [f"{t_dij * 1000:.3f}", f"{t_ch * 1000:.3f}", f"{a.seconds['query'] * 1000:.3f}"]})
        st.table({"Vorberechnung": ["CH (Zusammenziehen)", "Labels: CH-Suchräume + Bereinigen", "Labels: Pruned Landmark Labeling"],
                  "Zähler": [f"{_num(m['ch_witness_settled'])} festgelegte Knoten in Zeugensuchen, {m['n_shortcuts']} Abkürzungen", f"{_num(m['raw_settled'])} festgelegte Knoten, {_num(m['removed'])} Einträge gestrichen ({m['raw_avg']:.1f} → {m['clean_avg']:.1f} je Label)",
                             f"{_num(m['pll_settled'])} festgelegte Knoten in {m['pll_searches']} Suchen, {_num(m['pll_pruned'])} abgebrochen"],
                  "Laufzeit [ms]": [f"{a.seconds['ch'] * 1000:.1f}", f"{a.seconds['clean'] * 1000:.1f}", f"{a.seconds['pll'] * 1000:.1f}"]})
        eq = m["clean_equals_pll"]
        st.caption("Die Laufzeiten sind Messwerte dieses Laufs (reines Python, ein Lauf, schwankend). "
                   + ("Die bereinigten CH-Suchräume und das Pruned Landmark Labeling ergeben in dieser Ordnung **dieselben Labels** (Eintrag für Eintrag verglichen)." if eq else
                      "Die Zeile zu den CH-Suchräumen gilt für die CH-Ordnung; die Labels oben sind in einer anderen Ordnung gebaut.") +
                   " Die Route lässt sich nur aus den PLL-Labels zusammensetzen (sie speichern den nächsten Knoten je Eintrag).")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie groß werden die Labels?")
if st.button("Labelgröße gegen Netzgröße und Dichte messen (dauert einen Moment)", key="size_start"):
    st.session_state["size_on"] = True
if st.session_state.get("size_on"):
    with st.spinner("Rechne 8 Netzgrößen × 5 Netze und 4 Dichten × 5 Netze ..."):
        srows, drows = _sizes()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_size(srows), width="stretch", key="size_chart")
    c2.table({"Netz": [f"{'Stadt' if r['key'] == 'city' else 'Zufall'} {r['n']:.0f}" for r in srows], "Ø Einträge": [f"{r['avg']:.1f}" for r in srows], "von n²": [f"{r['share']:.1%}" for r in srows]})
    cs = [r for r in srows if r["key"] == "city"]
    st.caption(f"CH-Ordnung, Mittel über 5 Netze. Im Stadtnetz wächst die mittlere Labelgröße von {cs[0]['avg']:.1f} ({cs[0]['n']:.0f} Knoten) auf {cs[-1]['avg']:.1f} ({cs[-1]['n']:.0f} Knoten) - der Knotenzahl steht ein Faktor {cs[-1]['n'] / cs[0]['n']:.0f} gegenüber, den Labels ein Faktor {cs[-1]['avg'] / cs[0]['avg']:.1f}; "
               f"zusammen belegen sie im größten Netz nur {cs[-1]['share']:.1%} einer Entfernungstabelle. Dichte Netze brauchen mehr: im Zufallsnetz mit 200 Knoten wachsen die Labels von {drows[0]['avg']:.1f} (Grad {drows[0]['degree']:g}) auf {drows[-1]['avg']:.1f} (Grad {drows[-1]['degree']:g}).")

st.markdown("---")

st.subheader("🔬 Ordnung und Konstruktion")
if st.button("Ordnungen und Konstruktionen vergleichen (dauert einen Moment)", key="order_start"):
    st.session_state["order_on"] = True
if st.session_state.get("order_on"):
    with st.spinner("Rechne 3 Netztypen × 5 Netze × 3 Ordnungen und 3 Konstruktionen ..."):
        orows, crows = _orders()
    c1, c2 = st.columns(2)
    c1.markdown("**Ordnung der Hubs (PLL)**")
    c1.plotly_chart(build_orders(orows), width="stretch", key="order_chart")
    c2.markdown("**Konstruktion (CH-Ordnung)**")
    c2.plotly_chart(build_construction(crows), width="stretch", key="construction_chart")
    big = orows[1]
    cb = crows[1]
    st.caption(f"Mittel über 5 Netze. Die Ordnung entscheidet über die Größe: im 20 × 20-Stadtnetz {big['ch']:.1f} Einträge je Label mit der CH-Ordnung, {big['degree']:.1f} nach Grad, {big['random']:.1f} zufällig (Zufallsnetz mit 200 Knoten: {orows[2]['ch']:.1f} / {orows[2]['degree']:.1f} / {orows[2]['random']:.1f} - hier ist der Grad fast so gut wie die CH-Ordnung). "
               f"Die CH-Suchräume sind roh viel zu groß ({cb['raw']:.1f} im 20 × 20-Netz); nach dem Bereinigen bleiben {cb['clean']:.1f} - **genau die Labels des Pruned Landmark Labeling** (in {cb['same']:.0%} der Netze Eintrag für Eintrag gleich).")

st.markdown("---")

st.subheader("🔬 Wie viel spart die Abfrage?")
if st.button("Abfrage gegen Netzgröße messen (dauert einen Moment)", key="query_start"):
    st.session_state["query_on"] = True
if st.session_state.get("query_on"):
    with st.spinner("Rechne 8 Netzgrößen × 3 Netze × 100 Paare ..."):
        qrows = _queries()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_query(qrows), width="stretch", key="query_chart")
    cq = [r for r in qrows if r["key"] == "city"]
    c2.table({"Knoten": [f"{r['n']:.0f}" for r in cq], "Dijkstra": [f"{r['dijkstra']:.0f}" for r in cq], "CH / Labels": [f"{r['ch']:.0f} / {r['merge']:.0f}" for r in cq]})
    st.caption(f"Mittel über 100 Zufallspaare in 3 Netzen (alle Antworten mit Dijkstra verglichen). Im 20 × 20-Stadtnetz legt Dijkstra im Mittel {cq[-1]['dijkstra']:.0f} Knoten fest, die CH-Abfrage {cq[-1]['ch']:.0f}, die Labels brauchen {cq[-1]['merge']:.0f} Vergleichsschritte - "
               f"und keinen einzigen Schritt im Graphen. Die Schritte wachsen mit der Netzgröße nur noch langsam (Labels: {cq[0]['merge']:.1f} bei {cq[0]['n']:.0f} Knoten, {cq[-1]['merge']:.1f} bei {cq[-1]['n']:.0f}). Achten Sie auf die logarithmischen Achsen.")

st.markdown("---")

st.subheader("🔬 Wenn sich die Kosten ändern")
if st.button("Alte Labels nach einer Kostenänderung prüfen (dauert einen Moment)", key="stale_start"):
    st.session_state["stale_on"] = True
if st.session_state.get("stale_on"):
    with st.spinner("Rechne 4 Anteile × 5 Netze × 200 Paare ..."):
        trows = _stale()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_stale(trows), width="stretch", key="stale_chart")
    c2.table({"Teurere Straßen": [f"{r['fraction']:.0%}" for r in trows], "Falsch": [f"{r['wrong']:.0%}" for r in trows], "Neuaufbau (Knoten)": [_num(r["rebuild"]) for r in trows]})
    st.caption(f"Stadtnetz 10 × 10, ein Anteil der Straßen wird dreimal so teuer (Verkehr), Mittel über 5 Netze × 200 Paare. Die alten Labels liefern schon, wenn {trows[0]['fraction']:.0%} der Straßen teurer werden, für {trows[0]['wrong']:.0%} der Paare eine falsche Entfernung, bei {trows[-1]['fraction']:.0%} für {trows[-1]['wrong']:.0%}. "
               f"Der Neuaufbau (Pruned Landmark Labeling in derselben Ordnung) legt jedes Mal etwa {trows[0]['rebuild']:.0f} Knoten fest - und liefert danach wieder für jedes Paar die richtige Antwort. Labels sind eine Vorberechnung für **feste** Kosten.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Kosten ändern sich nicht** | Nach einer Kostenänderung liefern die alten Labels falsche Entfernungen: bei 2 / 5 / 10 / 20 % dreimal so teurer Straßen für 14 / 29 / 48 / 71 % der Paare (Stadtnetz 10 × 10). Die Labels müssen neu berechnet werden. | Customizable Contraction Hierarchies |
| **Der Speicher ist nebensächlich** | Jeder Knoten trägt eine Liste; im 20 × 20-Stadtnetz sind es im Mittel 23.7 Einträge, zusammen 5.9 % einer Entfernungstabelle - aber die Zahl wächst mit der Netzgröße und der Dichte (Grad 2 → 6 bei 200 Knoten: 5.5 → 20.5 Einträge). Für Straßennetze mit kleiner "Highway-Dimension" bleiben sie klein (Literatur, Abraham u. a.); in allgemeinen Graphen können sie linear wachsen (Literatur). | kompaktere Labels, Hub-Kompression |
| **Die Ordnung ist gut** | Eine schlechte Ordnung macht die Labels bis zum 1.7-Fachen größer (20 × 20-Stadtnetz: 40.3 zufällig gegen 23.7 mit der CH-Ordnung); die Antworten bleiben richtig. | Ordnung aus der CH, PLL-Ordnungen |
| **Nur die Entfernung zählt** | Für die Route braucht man zusätzlich den nächsten Knoten je Eintrag; mehrere Kostenarten (Zeit und CO₂) brauchen Labels mit mehreren Größen (Literatur). | Mehrkriterien-Demo |
| **Es gibt eine Karte, keine Fahrzeiten je Uhrzeit** | Zeitabhängige Kosten brauchen Labels aus Funktionen statt Zahlen (Literatur, nicht gebaut). | zeitabhängige Verfahren |
"""
)
st.caption("Der Contraction-Hierarchies-Ast der Kürzeste-Wege-Linie: Contraction Hierarchies (Stück 4), Hub Labeling (Stück 10).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit nichtnegativen Kosten $c_e$, Entfernung $d(u,v)$. Eine Ordnung $\pi$ weist jedem Knoten einen Rang zu (Position 0 = wichtigster Hub).

**Hub-Labels.** Jeder Knoten $v$ bekommt ein Vorwärtslabel $L_f(v)=\{(h,d(v,h))\}$ und ein Rückwärtslabel $L_b(v)=\{(h,d(h,v))\}$. Die **Deckungseigenschaft** verlangt: für alle $s,t$ gibt es ein $h\in L_f(s)\cap L_b(t)$ mit $d(s,h)+d(h,t)=d(s,t)$.

**Abfrage.** $d(s,t)=\min_{h\in L_f(s)\cap L_b(t)}\bigl(d(s,h)+d(h,t)\bigr)$. Sind beide Listen nach $\pi$ sortiert, genügt ein gleichzeitiger Durchlauf mit höchstens $|L_f(s)|+|L_b(t)|$ Schritten. Jede Summe ist die Länge einer echten Route, also mindestens $d(s,t)$; die Deckungseigenschaft sorgt für Gleichheit.

**Kanonische Labels.** $h\in L_f(v)$ genau dann, wenn $h$ auf einer kürzesten Route $v\leadsto h$ den größten Rang hat (kleinste Position). Sie sind die kleinsten Labels, die zu $\pi$ passen (Literatur: Abraham u. a. 2011). Deckung: der wichtigste Knoten $h$ einer kürzesten Route $s\leadsto t$ ist der wichtigste Knoten auf $s\leadsto h$ und auf $h\leadsto t$, steht also in beiden Labels.

**Pruned Landmark Labeling.** In der Reihenfolge von $\pi$ läuft von jedem Hub $x$ ein Dijkstra; erreicht es einen Knoten $u$ mit der Entfernung $\delta$, und liefern die bisherigen Labels schon $d'(x,u)\le\delta$, wird $u$ nicht beschriftet und nicht expandiert. Weil alle wichtigeren Hubs schon eingetragen sind, ist das genau die kanonische Regel.

**Labels aus der CH.** Die Aufwärtssuche von $v$ in der Hierarchie erreicht auf jeder kürzesten Route $v\leadsto t$ den wichtigsten Knoten (Spitze der CH-Abfrage); ihre Suchräume erfüllen also die Deckungseigenschaft. Ein Eintrag $(x,d)$ ist überflüssig, wenn ein anderer gemeinsamer Hub $y$ von $L_f(v)$ und $L_b(x)$ $d(v,y)+d(y,x)\le d$ liefert; nach dem Streichen (von den wichtigen zu den unwichtigen Knoten) bleiben die kanonischen Labels (in den Experimenten Eintrag für Eintrag bestätigt).

**Aufwand.** Speicher $\sum_v |L(v)|$ Einträge; Abfrage $O(|L(s)|+|L(t)|)$. In Straßennetzen mit kleiner Highway-Dimension $h$ sind die Labels $O(h\log D)$ groß (Literatur: Abraham u. a. 2010; gemessen: siehe Experimente); in allgemeinen Graphen kann die Labelgröße linear in $n$ wachsen.

Implementiert in `hl_graph.py` (CSR-Graph), `hl_ch.py` (Contraction Hierarchies als Ordnung und Vergleich), `hl_labels.py` (Labels, PLL, Abfrage, Route), `hl_scenario.py` (Netze), `hl_evaluation.py` (Kennzahlen, Bildfolge, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
