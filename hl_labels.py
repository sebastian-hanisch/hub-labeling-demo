"""Hub Labeling: jedem Knoten v wird eine kleine Liste von "Hubs" mitgegeben (Knoten x mit der Entfernung d(v, x)), so dass für jedes Paar (s, t) ein Hub auf einer kürzesten Route von s nach t in beiden Listen steht.
Die Abfrage ist dann nur noch das Verschmelzen zweier nach Hub sortierter Listen: d(s, t) = min über gemeinsame Hubs x von d(s, x) + d(x, t). Es wird nichts mehr im Graphen gesucht.

Zwei Konstruktionen aus derselben Knotenordnung (Ordnung = Wichtigkeit, Rang 0 = wichtigster Hub):
  * ch_labels + clean_labels: die Labels sind die Suchräume der CH-Aufwärtssuche (Abraham, Delling, Goldberg, Werneck 2011), danach werden überflüssige Einträge gestrichen.
  * pll_labels: Pruned Landmark Labeling (Akiba, Iwata, Yoshida 2013): von jedem Hub in der Reihenfolge der Wichtigkeit ein Dijkstra, das dort abbricht, wo die bisherigen Labels die Entfernung schon liefern.

Alles ist eigene Umsetzung auf dem CSR-Graphen aus hl_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe)."""

import heapq
from dataclasses import dataclass, field

import numpy as np

from hl_graph import reverse_graph

INF = float("inf")


@dataclass
class Labels:
    n: int
    order: list                     # Knoten von wichtigstem (Position 0) bis unwichtigstem Hub
    pos: np.ndarray                 # Position eines Knotens in `order`; das ist zugleich der Schlüssel, nach dem jedes Label sortiert ist
    f_key: list                     # f_key[v] = aufsteigende Hub-Positionen des Vorwärtslabels (Hubs, die v erreicht)
    f_dist: list                    # f_dist[v][i] = d(v, Hub)
    f_next: list                    # f_next[v][i] = nächster Knoten von v Richtung Hub (-1: Hub ist v selbst oder unbekannt)
    b_key: list                     # Rückwärtslabel: Hubs, die v erreichen (bei ungerichteten Graphen dieselben Listen wie vorwärts)
    b_dist: list                    # d(Hub, v)
    b_prev: list                    # Vorgänger von v auf der Route Hub -> v (-1: Hub ist v oder unbekannt)
    directed: bool
    method: str = ""
    counters: dict = field(default_factory=dict)

    def size(self, v):
        """Zahl der Einträge des Labels von v (bei gerichteten Graphen beide Richtungen)."""
        return len(self.f_key[v]) + (len(self.b_key[v]) if self.directed else 0)

    def sizes(self):
        return np.array([self.size(v) for v in range(self.n)])

    def total(self):
        return int(self.sizes().sum())

    def hubs(self, v, backward=False):
        """Die Hubs von v als (Hub-Knoten, Entfernung) in der Reihenfolge der Wichtigkeit."""
        keys, dists = (self.b_key[v], self.b_dist[v]) if backward else (self.f_key[v], self.f_dist[v])
        return [(self.order[k], d) for k, d in zip(keys, dists)]


@dataclass
class LabelQuery:
    dist: float
    hub: int                        # Hub-Knoten auf der kürzesten Route (-1, wenn kein gemeinsamer Hub)
    steps: int                      # Vergleichsschritte beim Verschmelzen
    trace: list = field(default_factory=list)      # je Schritt (i, j, gleich?, beste Summe bisher)


def dijkstra(g, s, t=None):
    """Einfaches Dijkstra: Entfernungen, Reihenfolge der festgelegten Knoten (bei gegebenem Ziel: Abbruch, sobald es festliegt) und Zahl der Kantenprüfungen."""
    n = g.n
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    dist = [INF] * n
    dist[s] = 0.0
    done = [False] * n
    heap = [(0.0, s)]
    order, relax = [], 0
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        order.append(u)
        if t is not None and u == t:
            break
        for k in range(ip[u], ip[u + 1]):
            relax += 1
            nd = d + w[k]
            v = ix[k]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return np.array(dist), order, relax


# --- Knotenordnungen -----------------------------------------------------------------------------------------------------------------------------------

ORDERS = ("ch", "degree", "random")


def make_order(g, kind, h=None, seed=0):
    """Knoten von wichtigstem bis unwichtigstem Hub: "ch" = umgekehrte CH-Zusammenziehreihenfolge (das zuletzt zusammengezogene ist am wichtigsten), "degree" = absteigender Grad (bei Gleichstand zufällig), "random" = zufällig."""
    if kind == "ch":
        return [int(v) for v in h.order[::-1]]
    if kind == "degree":
        deg = g.degree()
        tie = np.random.default_rng([int(seed), 4712]).random(g.n)
        return [int(v) for v in sorted(range(g.n), key=lambda v: (-int(deg[v]), tie[v]))]
    if kind == "random":
        return [int(v) for v in np.random.default_rng([int(seed), 4711]).permutation(g.n)]
    raise ValueError(kind)


def _new_labels(g, order, directed, method):
    n = g.n
    pos = np.empty(n, dtype=np.int64)
    pos[np.asarray(order)] = np.arange(n)
    lists = lambda: [[] for _ in range(n)]
    f_key, f_dist, f_next = lists(), lists(), lists()
    if directed:
        b_key, b_dist, b_prev = lists(), lists(), lists()
    else:
        b_key, b_dist, b_prev = f_key, f_dist, f_next
    return Labels(n, list(order), pos, f_key, f_dist, f_next, b_key, b_dist, b_prev, directed, method)


# --- Konstruktion 1: Suchräume der CH-Aufwärtssuche, dann bereinigen --------------------------------------------------------------------------------------

def ch_labels(g, h):
    """Rohe Labels aus einer Contraction Hierarchy `h`: das Vorwärtslabel von v sind alle Knoten, die die Aufwärtssuche (Kanten zu höherem Rang) von v aus erreicht, mit den dort gefundenen Entfernungen; das Rückwärtslabel
    die der Suche über `down_in` (Kanten von höherem Rang nach unten, rückwärts gelaufen). Die Ordnung ist die CH-Ordnung. Die Entfernungen sind die der Aufwärtssuche - nicht immer die echten (die kürzeste Route kann abwärts und wieder aufwärts führen)."""
    order = make_order(g, "ch", h)
    directed = g.directed
    L = _new_labels(g, order, directed, "ch_raw")
    pos = L.pos

    def space(v, adj):
        dist = {v: 0.0}
        heap = [(0.0, v)]
        done = {}
        while heap:
            d, x = heapq.heappop(heap)
            if x in done:
                continue
            done[x] = d
            for y, w, _ in adj[x]:
                nd = d + w
                if nd < dist.get(y, INF):
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        return done

    settled = 0
    for v in range(g.n):
        s = space(v, h.up)
        settled += len(s)
        for x in sorted(s, key=lambda x: pos[x]):
            L.f_key[v].append(int(pos[x]))
            L.f_dist[v].append(s[x])
            L.f_next[v].append(-1)
        if directed:
            s = space(v, h.down_in)
            settled += len(s)
            for x in sorted(s, key=lambda x: pos[x]):
                L.b_key[v].append(int(pos[x]))
                L.b_dist[v].append(s[x])
                L.b_prev[v].append(-1)
    L.counters = {"searches": g.n * (2 if directed else 1), "settled": settled, "raw_entries": L.total()}
    return L


def _covered(kv, dv, kx, dx, own_key):
    """Gibt es einen gemeinsamen Hub y != own_key der Listen (kv, dv) und (kx, dx) mit dv[y] + dx[y] <= Grenze? Rückgabe: kleinste Summe über gemeinsame Hubs ohne own_key."""
    i = j = 0
    best = INF
    while i < len(kv) and j < len(kx):
        a, b = kv[i], kx[j]
        if a == b:
            if a != own_key and dv[i] + dx[j] < best:
                best = dv[i] + dx[j]
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return best


def clean_labels(L):
    """Überflüssige Einträge streichen (Abraham et al. 2011): der Eintrag (x, d) im Vorwärtslabel von v ist überflüssig, wenn ein anderer gemeinsamer Hub y des Vorwärtslabels von v und des Rückwärtslabels von x die Entfernung
    d(v, y) + d(y, x) <= d liefert - dann liegt y auf einer mindestens ebenso kurzen Route v -> x. Knoten werden von den wichtigsten zu den unwichtigsten bearbeitet, damit die Labels der Hubs schon bereinigt sind
    (ein Label enthält nur Hubs, die mindestens so wichtig sind wie der Knoten selbst). Ändert keine Antwort (in den Tests gegen alle Paare geprüft)."""
    n = L.n
    removed = 0
    for v in L.order:
        for fwd in ((True, False) if L.directed else (True,)):
            keys, dists = (L.f_key[v], L.f_dist[v]) if fwd else (L.b_key[v], L.b_dist[v])
            nk, nd = [], []
            for k, d in zip(keys, dists):
                x = L.order[k]
                if x == v:
                    nk.append(k)
                    nd.append(d)
                    continue
                # Vorwärts: Route v -> y -> x, y im Vorwärtslabel von v und im Rückwärtslabel von x; Rückwärts: Route x -> y -> v, y im Rückwärtslabel von v und im Vorwärtslabel von x
                other_k, other_d = (L.b_key[x], L.b_dist[x]) if fwd else (L.f_key[x], L.f_dist[x])
                if _covered(keys, dists, other_k, other_d, k) <= d:
                    removed += 1
                else:
                    nk.append(k)
                    nd.append(d)
            if fwd:
                L.f_key[v], L.f_dist[v] = nk, nd
                L.f_next[v] = [-1] * len(nk)
            else:
                L.b_key[v], L.b_dist[v] = nk, nd
                L.b_prev[v] = [-1] * len(nk)
        if not L.directed:
            L.b_key, L.b_dist, L.b_prev = L.f_key, L.f_dist, L.f_next
    L.method = "ch_clean"
    L.counters = {**L.counters, "removed": removed, "clean_entries": L.total()}
    return L


# --- Konstruktion 2: Pruned Landmark Labeling ----------------------------------------------------------------------------------------------------------

def pll_labels(g, order):
    """Pruned Landmark Labeling in der Reihenfolge `order` (wichtigster Hub zuerst). Für jeden Hub x ein Dijkstra (vorwärts und, bei gerichteten Graphen, rückwärts); erreicht es einen Knoten u mit einer Entfernung, die die bisherigen Labels
    schon liefern (Hub y aus einer früheren Runde auf einer mindestens ebenso kurzen Route), wird u weder beschriftet noch von dort aus weitergesucht. Die Labels sind kanonisch: x steht in v, wenn x der wichtigste Knoten auf einer
    kürzesten Route v -> x ist. Enthält zu jedem Eintrag den nächsten Knoten Richtung Hub (für die Route)."""
    n = g.n
    directed = g.directed
    L = _new_labels(g, order, directed, "pll")
    rg = reverse_graph(g) if directed else g
    counters = {"searches": 0, "settled": 0, "pruned": 0, "checks": 0}

    def run(graph, root_key, root, root_labels, other_labels, write_labels):
        """Ein Dijkstra ab `root` auf `graph`. root_labels = (Schlüssel, Entfernungen) des Wurzelknotens, die für den Vergleich in ein Feld kommen; other_labels = Listen der besuchten Knoten, gegen die verglichen wird;
        write_labels = (Schlüssel, Entfernungen, Vorgänger) der besuchten Knoten, in die eingetragen wird."""
        counters["searches"] += 1
        ip, ix, w = graph.indptr.tolist(), graph.indices.tolist(), graph.weight.tolist()
        temp = {k: d for k, d in zip(*root_labels)}
        dist = {root: 0.0}
        prev = {root: -1}
        heap = [(0.0, root)]
        done = set()
        wk, wd, wp = write_labels
        ok_, od_ = other_labels
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            counters["settled"] += 1
            best = INF
            for k, du in zip(ok_[u], od_[u]):
                counters["checks"] += 1
                tv = temp.get(k)
                if tv is not None and tv + du < best:
                    best = tv + du
            if best <= d:
                counters["pruned"] += 1
                continue
            wk[u].append(root_key)
            wd[u].append(d)
            wp[u].append(prev[u])
            for e in range(ip[u], ip[u + 1]):
                v = ix[e]
                nd = d + w[e]
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

    for key, x in enumerate(order):
        if directed:
            # x -> u: Eintrag im Rückwärtslabel von u (Hub x erreicht u); Vergleich: Vorwärtslabel von x gegen Rückwärtslabel von u
            run(g, key, x, (L.f_key[x], L.f_dist[x]), (L.b_key, L.b_dist), (L.b_key, L.b_dist, L.b_prev))
            # u -> x: Eintrag im Vorwärtslabel von u; Vergleich: Rückwärtslabel von x gegen Vorwärtslabel von u
            run(rg, key, x, (L.b_key[x], L.b_dist[x]), (L.f_key, L.f_dist), (L.f_key, L.f_dist, L.f_next))
        else:
            run(g, key, x, (L.f_key[x], L.f_dist[x]), (L.f_key, L.f_dist), (L.f_key, L.f_dist, L.f_next))
    L.counters = counters | {"entries": L.total()}
    return L


# --- Abfrage -----------------------------------------------------------------------------------------------------------------------------------------

def label_query(L, s, t, trace=False):
    """Kürzeste Entfernung s -> t: Vorwärtslabel von s und Rückwärtslabel von t nach Hub-Position verschmelzen; Zähler = Vergleichsschritte. Ohne gemeinsamen Hub (unerreichbar): Entfernung unendlich."""
    ks, ds = L.f_key[s], L.f_dist[s]
    kt, dt = L.b_key[t], L.b_dist[t]
    i = j = steps = 0
    best, hub = INF, -1
    tr = []
    while i < len(ks) and j < len(kt):
        steps += 1
        a, b = ks[i], kt[j]
        match = a == b
        if match:
            if ds[i] + dt[j] < best:
                best, hub = ds[i] + dt[j], a
        if trace:
            tr.append((i, j, match, best))
        if a == b:
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return LabelQuery(best, L.order[hub] if hub >= 0 else -1, steps, tr)


def label_route(L, s, t):
    """Die kürzeste Route s -> t als Knotenfolge, über die gespeicherten nächsten Knoten Richtung Hub (nur bei PLL-Labels; sonst None). Leer, wenn t nicht erreichbar ist."""
    if L.method != "pll":
        return None
    q = label_query(L, s, t)
    if q.hub < 0:
        return []
    hk = int(L.pos[q.hub])
    up = [s]
    cur = s
    while cur != q.hub:
        i = L.f_key[cur].index(hk)
        cur = L.f_next[cur][i]
        up.append(cur)
    down = [t]
    cur = t
    while cur != q.hub:
        i = L.b_key[cur].index(hk)
        cur = L.b_prev[cur][i]
        down.append(cur)
    return up + down[-2::-1]
