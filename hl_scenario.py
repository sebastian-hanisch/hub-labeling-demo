"""Die Netze der Demo: kleines Netz (acht Orte, aus der CH-Demo übernommen), erzeugtes Stadtnetz und erzeugtes Zufallsnetz. Alle Kosten sind ganze Zahlen (Meter, Minuten, Einheiten): Gleichstände sind exakt, Zähler und Kosten plattformfest."""

from dataclasses import dataclass

import numpy as np

import hl_constants as C
from hl_graph import Graph, from_arcs, reverse_graph


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    title: str
    note: str = ""
    side: int = 0
    fixed_pair: tuple = ()         # (Start, Ziel) bei Netzen mit fester Aufgabe, sonst leer
    geometric: bool = True         # Lage der Knoten ist eine Karte (Kanten zeichnen), sonst nur Punkte
    unit: str = ""


# --- Stadtnetz -------------------------------------------------------------------------------------------------------------------------------------

def build_city(side, seed):
    """Gestörtes Raster: Kreuzungen im Abstand SPACING mit Lageabweichung, jede mit ihren Nachbarn rechts und darüber verbunden. Kosten einer Straße = ihre Länge in Metern mal (1 + SPREAD * Zufall), ganzzahlig."""
    rng = np.random.default_rng([int(seed), 101])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    arcs = []
    for i in range(side):
        for j in range(side):
            for di, dj in ((0, 1), (1, 0)):
                i2, j2 = i + di, j + dj
                if i2 < side and j2 < side:
                    u, v = i * side + j, i2 * side + j2
                    length = float(np.hypot(*(xy[u] - xy[v])))
                    arcs.append((u, v, max(1.0, float(np.rint(length * (1.0 + C.SPREAD * rng.random()))))))
    return from_arcs(n, arcs, xy)


def city_network(side, seed):
    return Network("city", build_city(int(side), int(seed)), "Stadtnetz", "Erzeugtes Stadtnetz: Kreuzungen auf einem gestörten Raster, jede mit den Nachbarn rechts und darüber verbunden; Kosten = Länge in Metern mit Zufallszuschlag.", int(side), unit="m")


# --- Zufallsnetz -----------------------------------------------------------------------------------------------------------------------------------

def build_random(n, degree, seed):
    """Zusammenhängender Zufallsgraph: jeder Knoten hängt an einem früheren, dann kommen zufällige Kanten bis zum mittleren Grad `degree`. Ganzzahlige Kosten 1 bis 9. Die Kugeln um einen Knoten wachsen hier exponentiell (wenige Schritte bis überall hin)."""
    rng = np.random.default_rng([int(seed), 707])
    arcs = {(int(rng.integers(0, i)), i): float(rng.integers(1, 10)) for i in range(1, n)}
    target_m = int(round(n * degree / 2))
    while len(arcs) < target_m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs and (v, u) not in arcs:
            arcs[(u, v)] = float(rng.integers(1, 10))
    xy = rng.random((n, 2)) * 1000.0
    return from_arcs(n, [(u, v, w) for (u, v), w in arcs.items()], xy)


def random_network(n, degree, seed):
    return Network("random", build_random(int(n), float(degree), int(seed)), "Zufallsnetz",
                   "Erzeugter Zufallsgraph: jeder Knoten hat im Mittel dieselbe Zahl Nachbarn, es gibt keine Karte - die Punkte sind zufällig verteilt und die Kanten nicht gezeichnet.", geometric=False, unit="Einheiten")


# --- Kleines Netz ----------------------------------------------------------------------------------------------------------------------------------

SMALL_STOPS = [("Altstadt", 0.0, 2.4), ("Bahnhof", 1.2, 1.2), ("Campus", 0.4, 0.0), ("Anschluss West", 2.8, 0.8), ("Anschluss Ost", 5.0, 0.8), ("Dorf", 6.6, 2.0), ("Eiche", 7.0, 0.2), ("Fabrik", 5.8, 3.0)]
SMALL_LINKS = [("Altstadt", "Bahnhof", 3), ("Bahnhof", "Campus", 2), ("Altstadt", "Anschluss West", 4), ("Campus", "Anschluss West", 3), ("Bahnhof", "Anschluss West", 5),
               ("Anschluss West", "Anschluss Ost", 5), ("Anschluss Ost", "Dorf", 3), ("Anschluss Ost", "Fabrik", 4), ("Dorf", "Eiche", 2), ("Eiche", "Fabrik", 3), ("Dorf", "Fabrik", 6), ("Bahnhof", "Eiche", 12)]


def small_network():
    names = [s[0] for s in SMALL_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], w) for a, b, w in SMALL_LINKS], [(s[1], s[2]) for s in SMALL_STOPS], names)
    return Network("small", g, "Kleines Netz", "Ein eigenes kleines Netz: zwei Gruppen von Orten mit Nebenstraßen und eine Verbindung zwischen den Anschlüssen (Fahrminuten an den Strecken). Jeder Ort bekommt eine kurze Liste von Hubs - "
                   "die beiden Gruppen treffen sich an den Anschlüssen.", fixed_pair=(idx["Altstadt"], idx["Fabrik"]), unit="min")


def make_network(net, side=C.DEFAULT_SIDE, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, seed=C.DEFAULT_SEED):
    if net == "small":
        return small_network()
    if net == "city":
        return city_network(side, seed)
    if net == "random":
        return random_network(nodes, degree, seed)
    raise ValueError(net)
