"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen
SPREAD = 0.3                       # Kosten einer Straße = Länge * (1 + SPREAD * Zufall): Ampeln, Steigung, Belag

NETS = ("small", "city", "random")
NET_LABELS = {
    "small": "🔀 Kleines Netz (acht Orte)",
    "city": "🏙️ Stadtnetz (erzeugt)",
    "random": "🕸️ Zufallsnetz (erzeugt)",
}
FIXED_NETS = ("small",)

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 20, 10               # n = Seite², höchstens 400 Knoten
NODES_MIN, NODES_MAX, DEFAULT_NODES = 20, 300, 100
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 2.0, 6.0, 3.0
DISTANCE_MIN, DISTANCE_MAX, DEFAULT_DISTANCE = 0, 100, 60  # Ziel bei diesem Prozentrang der Entfernungen vom Start
DEFAULT_SEED = 7
DEFAULT_NET = "small"

ORDER_LABELS = {"ch": "Contraction Hierarchies (Kantendifferenz)", "degree": "Grad (viele Nachbarn zuerst)", "random": "Zufall"}
DEFAULT_ORDER = "ch"
METHODS = ("pll", "ch")
METHOD_LABELS = {"pll": "Pruned Landmark Labeling", "ch": "CH-Suchräume, bereinigt"}
DEFAULT_METHOD = "pll"

SWEEP_SEEDS = tuple(range(100000, 100005))
PAIR_SAMPLES = 100

COLORS = {"s": "#1f77b4", "t": "#d62728", "hub": "#ff7f0e", "route": "#2ca02c", "start": "#111111", "goal": "#ff7f0e"}


_BASE = dict(side=DEFAULT_SIDE, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, distance=DEFAULT_DISTANCE, order=DEFAULT_ORDER, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random"},
    "🎲 Schlechte Ordnung": {**_BASE, "net": "city", "side": 20, "order": "random"},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Kleines Netz (8 Orte, 12 Straßen): alle Labels zusammen haben 24 Einträge (im Mittel 3 je Ort, das größte 5) statt 64 Entfernungen. Altstadt → Fabrik (13 min): gemeinsamer Hub ist Anschluss West, 5 Vergleichsschritte - Dijkstra legt 7 Knoten fest, die CH-Abfrage 8 (bei acht Orten lohnt sich nichts davon).",
    "🏙️ Stadtnetz": "Stadtnetz (10 × 10, CH-Ordnung): im Mittel 12.4 Einträge je Label (das größte 21), zusammen 1 241 - 12 % der 10 000 Entfernungen zwischen allen Paaren. Ein Paar in 628 m Entfernung: 16 Vergleichsschritte, Dijkstra legt 61 Knoten fest, die CH-Abfrage 19. Im Mittel über 100 Paare: 14.0 Schritte gegen 48.2 (Dijkstra) und 22.1 (CH).",
    "🕸️ Zufallsnetz": "Zufallsnetz (100 Knoten, Grad 3, CH-Ordnung): im Mittel 8.4 Einträge je Label (das größte 15), zusammen 835. Ein Paar in Entfernung 20: 10 Vergleichsschritte gegen 61 festgelegte Knoten bei Dijkstra und 14 bei der CH-Abfrage.",
    "🎲 Schlechte Ordnung": "Stadtnetz 20 × 20 mit zufälliger Hub-Reihenfolge: im Mittel 38.6 Einträge je Label statt 22.1 mit der CH-Ordnung (zusammen 15 448 statt 8 837), und das Paar braucht 60 statt 39 Vergleichsschritte. Die Antworten sind dieselben - nur die Labels werden größer.",
}
