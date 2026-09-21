"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Einträge, Vergleichsschritte, festgelegte Knoten und ganzzahlige Kosten sind plattformfest
(reine Python-Rechnung mit festen Seeds); Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import pytest

import hl_constants as C
import hl_evaluation as ev
import hl_scenario as sc

PRESET = {"small": "🔀 Kleines Netz", "city": "🏙️ Stadtnetz", "random": "🕸️ Zufallsnetz", "bad": "🎲 Schlechte Ordnung"}


def _preset(key):
    p = C.PRESETS[PRESET[key]]
    net = sc.make_network(p["net"], p["side"], p["nodes"], p["degree"], p["seed"])
    return p, ev.analyse(net, p["order"], p["distance"], p["seed"])


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    _, a = _preset("small")
    m, g = a.metrics, a.net.graph
    assert (m["n"], m["m"], m["total"], m["table"], m["max"]) == (8, 12, 24, 64, 5) and set(len(a.L.f_key[v]) for v in range(8)) == {1, 2, 3, 4, 5} and m["avg"] == 3.0
    assert (m["dist"], g.names[m["hub"]], m["steps"], m["settled_dijkstra"], m["settled_ch"]) == (13.0, "Anschluss West", 5, 7, 8)
    _has("small", "8 Orte, 12 Straßen", "24 Einträge", "im Mittel 3 je Ort", "das größte 5", "64 Entfernungen", "Altstadt → Fabrik (13 min)", "Anschluss West", "5 Vergleichsschritte", "7 Knoten", "die CH-Abfrage 8")


def test_city_preset_numbers():
    _, a = _preset("city")
    m = a.metrics
    assert (m["n"], m["max"], m["total"], m["table"], m["dist"], m["steps"], m["settled_dijkstra"], m["settled_ch"]) == (100, 21, 1241, 10000, 628.0, 16, 61, 19)
    assert m["avg"] == pytest.approx(12.41, abs=0.005) and m["total"] / m["table"] == pytest.approx(0.124, abs=0.0005)
    assert (m["pairs"]["merge"], m["pairs"]["dijkstra"], m["pairs"]["ch"]) == pytest.approx((14.0, 48.2, 22.1), abs=0.05) and m["pairs"]["exact"]
    _has("city", "10 × 10", "12.4 Einträge", "das größte 21", "1 241", "12 %", "10 000", "628 m", "16 Vergleichsschritte", "61 Knoten", "die CH-Abfrage 19", "14.0 Schritte gegen 48.2", "22.1 (CH)")


def test_random_preset_numbers():
    _, a = _preset("random")
    m = a.metrics
    assert (m["n"], m["max"], m["total"], m["dist"], m["steps"], m["settled_dijkstra"], m["settled_ch"]) == (100, 15, 835, 20.0, 10, 61, 14) and m["avg"] == pytest.approx(8.35, abs=0.005)
    _has("random", "100 Knoten, Grad 3", "8.4 Einträge", "das größte 15", "835", "Entfernung 20", "10 Vergleichsschritte", "61 festgelegte Knoten", "14 bei der CH-Abfrage")


def test_bad_order_preset_numbers():
    p, a = _preset("bad")
    good = ev.analyse(sc.city_network(20, 7), "ch", 60, 7)
    assert p["order"] == "random" and a.net.graph.n == 400
    assert (a.metrics["total"], good.metrics["total"]) == (15448, 8837) and a.metrics["avg"] == pytest.approx(38.62, abs=0.005) and good.metrics["avg"] == pytest.approx(22.09, abs=0.005)
    assert (a.metrics["steps"], good.metrics["steps"]) == (60, 39) and a.metrics["dist"] == good.metrics["dist"]                                  # dieselbe Antwort, größere Labels
    _has("bad", "20 × 20", "38.6 Einträge", "22.1", "15 448", "8 837", "60 statt 39")


# --- Sidebar-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_size_help_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.size_rows()}
    assert [rows[("city", s)]["avg"] for s in (6, 10, 14, 20)] == pytest.approx([6.8, 12.4, 17.0, 23.7], abs=0.05)
    assert [rows[("random", s)]["avg"] for s in (50, 100, 200, 300)] == pytest.approx([6.1, 8.7, 12.8, 16.0], abs=0.05)
    assert rows[("city", 20)]["share"] == pytest.approx(0.059, abs=0.0005) and rows[("city", 20)]["avg"] / rows[("city", 6)]["avg"] == pytest.approx(3.5, abs=0.05) and rows[("city", 20)]["n"] / rows[("city", 6)]["n"] == pytest.approx(11.1, abs=0.05)
    assert all(rows[("city", b)]["avg"] / rows[("city", a)]["avg"] < rows[("city", b)]["n"] / rows[("city", a)]["n"] for a, b in ((6, 10), (10, 14), (14, 20)))          # sublinear
    assert all(rows[("city", s)]["share"] > rows[("city", t)]["share"] for s, t in ((6, 10), (10, 14), (14, 20)))


def test_density_help_numbers():
    rows = ev.density_rows()
    assert [r["avg"] for r in rows] == pytest.approx([5.5, 12.8, 17.1, 20.5], abs=0.05) and [r["degree"] for r in rows] == [2.0, 3.0, 4.0, 6.0]


def test_order_help_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.order_rows()}
    big = rows[("city", 20)]
    assert (big["ch"], big["degree"], big["random"]) == pytest.approx((23.7, 35.8, 40.3), abs=0.05) and big["random"] / big["ch"] == pytest.approx(1.7, abs=0.01)
    assert (rows[("city", 10)]["ch"], rows[("city", 10)]["degree"], rows[("city", 10)]["random"]) == pytest.approx((12.35, 15.17, 17.48), abs=0.005)
    rnd = rows[("random", 200)]
    assert (rnd["ch"], rnd["degree"], rnd["random"]) == pytest.approx((12.8, 13.0, 31.2), abs=0.05)
    assert rnd["degree"] < 1.02 * rnd["ch"]                                                       # im Zufallsnetz ist der Grad fast so gut wie die CH-Ordnung


# --- Experimente und die Tabelle "Wo die Annahmen enden" ------------------------------------------------------------------------------------------

def test_construction_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.construction_rows()}
    big = rows[("city", 20)]
    assert big["raw"] == pytest.approx(49.05, abs=0.005) and big["clean"] == pytest.approx(23.72, abs=0.005) and big["same"] == 1.0
    assert all(r["same"] == 1.0 and r["clean"] == pytest.approx(r["pll"]) and r["raw"] > 1.4 * r["clean"] for r in rows.values())               # bereinigt = PLL, Eintrag für Eintrag
    assert big["raw_settled"] > big["pll_settled"]


def test_query_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.query_rows()}
    big, small = rows[("city", 20)], rows[("city", 6)]
    assert (big["dijkstra"], big["ch"], big["merge"]) == pytest.approx((209.9, 62.3, 32.0), abs=0.05) and (small["merge"], small["n"]) == (pytest.approx(7.6, abs=0.05), 36)
    assert all(r["merge"] < r["ch"] < r["dijkstra"] for r in rows.values())                       # in jedem gemessenen Netz: Labels < CH < Dijkstra


def test_stale_numbers():
    rows = {r["fraction"]: r for r in ev.stale_rows()}
    assert [rows[f]["wrong"] for f in (0.02, 0.05, 0.1, 0.2)] == pytest.approx([0.136, 0.291, 0.481, 0.711], abs=0.0005)                            # Tabelle: 14 / 29 / 48 / 71 %
    assert [rows[f]["rebuild"] for f in (0.02, 0.05, 0.1, 0.2)] == pytest.approx([2033.8, 2049.2, 2032.4, 2037.8], abs=0.5)
    assert all(rows[a]["wrong"] < rows[b]["wrong"] for a, b in ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2)))


def test_labels_answer_every_pair_exactly_on_the_presets():
    for key in PRESET:
        _, a = _preset(key)
        assert a.metrics["exact"] and a.metrics["pairs"]["exact"], key
