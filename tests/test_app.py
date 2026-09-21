"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Ordnungen, ausgeblendete Regler, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hl_constants as C
from hl_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_states_the_result():
    at = _run()
    assert any("Hub Labeling in Aktion" in m.value for m in at.markdown)
    assert len(at.success) == 1 and "Ohne einen Schritt im Graphen" in at.success[0].value and "5 Vergleichsschritte" in at.success[0].value and not at.warning and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_one_verdict(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert len(at.success) == 1 and not at.warning and not at.error


@pytest.mark.parametrize("order", tuple(C.ORDER_LABELS))
@pytest.mark.parametrize("net", C.NETS)
def test_every_order_renders_on_every_net(order, net):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["order_select"] = order
        at.session_state["side_slider"] = 6
        at.session_state["nodes_slider"] = 40
    at = _run(setup)
    assert len(at.success) == 1 and not at.warning


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN

    def big(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MAX
        at.session_state["distance_slider"] = C.DISTANCE_MAX

    def near(at):
        at.session_state["net_select"] = "random"
        at.session_state["nodes_slider"] = C.NODES_MIN
        at.session_state["degree_slider"] = C.DEGREE_MIN
        at.session_state["distance_slider"] = C.DISTANCE_MIN
    for setup in (small, big, near):
        at = _run(setup)
        assert at.slider(key="hl_step").value == at.slider(key="hl_step").max


def test_hidden_controls_follow_the_net():
    common = {"Netz", "Ordnung der Hubs"}
    small, city, rnd = (_labels(_run(net=n)) for n in ("small", "city", "random"))
    assert small == common                                                                            # feste Aufgabe: kein Paar, kein Seed
    assert city == common | {"Kreuzungen je Seite", "Entfernung des Paares [Perzentil]", "Zufalls-Seed"}
    assert rnd == common | {"Knoten", "Mittlerer Grad", "Entfernung des Paares [Perzentil]", "Zufalls-Seed"}


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="city")
    at.session_state["side_slider"] = 7
    at.run()
    at.session_state["net_select"] = "small"
    at.run()
    at.session_state["net_select"] = "city"
    at.run()
    assert not at.exception and at.slider(key="side_slider").value == 7


def test_step_slider_returns_to_the_last_step_when_the_pair_or_the_order_changes():
    at = _run(net="city")
    at.slider(key="hl_step").set_value(3)
    at.run()
    assert at.slider(key="hl_step").value == 3
    at.slider(key="distance_slider").set_value(30)
    at.run()
    assert not at.exception and at.slider(key="hl_step").value == at.slider(key="hl_step").max
    at.slider(key="hl_step").set_value(2)
    at.run()
    at.session_state["order_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="hl_step").value == at.slider(key="hl_step").max


def test_every_step_of_the_small_network_renders():
    at = _run()
    for k in range(0, int(at.slider(key="hl_step").max) + 1):
        at.slider(key="hl_step").set_value(k)
        at.run()
        assert not at.exception, k


def test_play_renders_several_frames_without_duplicate_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    for setup in (lambda a: None, lambda a: a.session_state.__setitem__("net_select", "city"), lambda a: a.session_state.__setitem__("net_select", "random")):
        at = _run(setup)
        _play(at)
        assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_select_the_net_and_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "random"
    at.query_params["nodes"] = "999"
    at.query_params["deg"] = "3.3"
    at.query_params["order"] = "degree"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "random" and at.selectbox(key="order_select").value == "degree"
    assert at.slider(key="nodes_slider").value == C.NODES_MAX and at.slider(key="degree_slider").value == 3.5


def test_unknown_values_in_the_permalink_fall_back_to_the_defaults():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.query_params["order"] = "alphabet"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET and at.selectbox(key="order_select").value == C.DEFAULT_ORDER


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über 5 Netze" in c.value for c in at.caption)
    for key in ("size_start", "order_start", "query_start", "stale_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("wächst die mittlere Labelgröße", "genau die Labels des Pruned Landmark Labeling", "keinen einzigen Schritt im Graphen", "Labels sind eine Vorberechnung für **feste** Kosten"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    # Die Karte steht in der Play-Schleife: ihr Schlüssel trägt den Schritt
    assert sorted(keys) == sorted(["net_chart", "hist_chart", "size_chart", "order_chart", "construction_chart", "query_chart", "stale_chart"]), keys
    assert sum('key=f"' in c for c in calls) == 1 and all('_{current}"' in c for c in calls if 'key=f"' in c)
    viz = (ROOT / "hl_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
