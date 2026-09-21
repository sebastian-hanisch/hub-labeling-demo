# Hub Labeling – jede Abfrage ist ein Listen-Merge – Streamlit-Demo

Zehntes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind von [Contraction Hierarchies](../contraction-hierarchies-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Hub Labeling** – an einem wachsenden Beispiel.
Eine Contraction Hierarchy beschleunigt Dijkstra, indem sie nur noch aufwärts sucht – aber sie sucht noch im Graphen. Beim Hub Labeling wird diese Suche **vorweggenommen**: jeder Knoten bekommt eine kurze Liste von **Hubs** (wichtigen Knoten) samt Entfernung, so dass für jedes Paar
(s, t) ein Hub auf einer kürzesten Route in **beiden** Listen steht. Die Abfrage ist das **Verschmelzen zweier sortierter Listen** – kein Schritt im Graphen. Der Preis: **Speicher**, und die Labels müssen **neu berechnet werden, wenn sich die Kosten ändern**.

**Einordnung in die Reihe (die Kanten des Graphen):**
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       │                         └─ hub-labeling-demo (die Suche vorwegnehmen)       [dieses Stück]
       ├─ bellman-ford-demo ─┐                                                       [gebaut]
       │   floyd-warshall-demo ─┴→ johnson-demo (Konvergenz: Umgewichtung)           [gebaut]
       ├─ multicriteria-demo (Zeit gegen CO₂, Pareto)                                [gebaut]
       └─ time-dependent-demo (Kosten hängen von der Uhrzeit ab)                     [gebaut]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (Hub Labeling, Abfrage durch Verschmelzen) | Cohen, Halperin, Kaplan und Zwick (2003); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) behandeln Hub Labeling nicht, es gibt kein Buchbeispiel zu spiegeln |
| Konstruktion aus der CH-Ordnung (Suchräume der Aufwärtssuche, Bereinigen) | Abraham, Delling, Goldberg und Werneck (2011) |
| Pruned Landmark Labeling | Akiba, Iwata und Yoshida (2013) |
| Umsetzung, Contraction Hierarchies als Ordnung (Kopie aus `contraction-hierarchies-demo`), Vergleiche, Messreihen | eigen |
| Alle Netze | **eigene Graphen und Erzeuger**: kleines Netz mit acht Orten (aus der CH-Demo), Stadtnetz auf einem gestörten Raster, Zufallsnetz |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Kein OpenStreetMap-Auszug**, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (8 Orte, 12 Straßen) | ✅ alle Labels zusammen haben **24 Einträge** (im Mittel 3 je Ort, das größte 5) statt 64 Entfernungen; Altstadt → Fabrik (13 min): gemeinsamer Hub Anschluss West, **5 Vergleichsschritte** – Dijkstra legt 7 Knoten fest, die CH-Abfrage 8 (bei acht Orten lohnt sich nichts davon) |
| Stadtnetz (10 × 10, CH-Ordnung) | ✅ im Mittel **12.4 Einträge** je Label (das größte 21), zusammen 1 241 = 12 % der 10 000 Entfernungen; Paar in 628 m: **16 Schritte** gegen 61 festgelegte Knoten (Dijkstra) und 19 (CH); Mittel über 100 Paare 14.0 gegen 48.2 und 22.1 |
| Zufallsnetz (100 Knoten, Grad 3) | ✅ im Mittel 8.4 Einträge (das größte 15), zusammen 835; Paar in Entfernung 20: 10 Schritte gegen 61 (Dijkstra) und 14 (CH) |
| Labelgröße gegen Netzgröße (Mittel über 5 Netze) | ✅ Stadtnetz mit 36 / 100 / 196 / 400 Knoten: **6.8 / 12.4 / 17.0 / 23.7** Einträge – der Knotenzahl steht ein Faktor 11 gegenüber, den Labels 3.5; im größten Netz belegen die Labels **5.9 %** einer Entfernungstabelle. Zufallsnetz mit 50 / 100 / 200 / 300 Knoten: 6.1 / 8.7 / 12.8 / 16.0 |
| Dichte | ⚠️ Zufallsnetz mit 200 Knoten, Grad 2 / 3 / 4 / 6: **5.5 / 12.8 / 17.1 / 20.5** Einträge – dichtere Netze brauchen größere Labels |
| Abfrage gegen Netzgröße (100 Paare je Netz) | ✅ im 20 × 20-Stadtnetz legt Dijkstra im Mittel **210** Knoten fest, die CH-Abfrage **62**, die Labels brauchen **32 Vergleichsschritte** und keinen Schritt im Graphen; in **jedem** gemessenen Netz: Labels < CH < Dijkstra |
| Ordnung der Hubs (20 × 20-Stadtnetz) | ⚠️ CH-Ordnung / Grad / Zufall: **23.7 / 35.8 / 40.3** Einträge je Label (Faktor 1.7 zwischen bester und schlechtester); im Zufallsnetz (200 Knoten) 12.8 / 13.0 / 31.2 – dort ist der Grad fast so gut wie die CH-Ordnung. Die Antworten sind immer dieselben |
| Zwei Konstruktionen | ✅ die rohen Suchräume der CH-Aufwärtssuche sind zu groß (im 20 × 20-Stadtnetz rund 49 Einträge je Label); nach dem **Bereinigen** bleiben 23.7 – **genau die Labels des Pruned Landmark Labeling**, Eintrag für Eintrag gleich in allen gemessenen Netzen |
| Kostenänderung (Stadtnetz 10 × 10) | ❌ die alten Labels sind veraltet: wenn 2 / 5 / 10 / 20 % der Straßen dreimal so teuer werden, liefern sie für **14 / 29 / 48 / 71 %** der Paare eine falsche Entfernung; der Neuaufbau legt jedes Mal etwa 2 034 Knoten fest |
| Korrektheit | ✅ jede Konstruktion (roh, bereinigt, PLL in drei Ordnungen) beantwortet **jedes Paar** exakt wie networkx (gerichtete und ungerichtete Graphen, Nullkosten, Parallelkanten, unerreichbare Ziele); die PLL-Labels sind **kanonisch** (aus den Entfernungen unabhängig nachgerechnet: x steht in v genau dann, wenn kein wichtigerer Knoten auf einer kürzesten Route v → x liegt); Routen sind echte Routen mit den berichteten Kosten |

Die Zähler (Einträge, Vergleichsschritte, festgelegte Knoten) sind Schritte des Verfahrens und plattformfest. Laufzeiten stehen in der App nur als Messwerte (reines Python) und werden nirgends behauptet oder getestet.

## Was die Demo zeigt

1. **Hub Labeling in Aktion:** beim kleinen Netz die **Labels aller Orte** als Tabelle; ein Regler über die **Vergleichsschritte** beim Verschmelzen der Listen von Start und Ziel (+ Abspielen): die Karte mit den Hubs von s (blau) und t (rot) als Ringe, schon verglichene gefüllt, gemeinsame Hubs orange, am Ende der Hub der Route (Stern) und die Route; dazu die letzten Vergleichsschritte als Tabelle.
2. **Der Preis der Labels:** Einträge je Label, Einträge insgesamt gegen eine Entfernungstabelle, Abfrage dieses Paares und im Mittel über 100 Paare (Vergleichsschritte gegen festgelegte Knoten bei Dijkstra und CH), Verteilung der Labelgrößen; alle Antworten werden mit Dijkstra verglichen.
3. **Vergleich** (Expander: Dijkstra, CH und Labels; Vorberechnung von CH, Labels aus CH-Suchräumen und Pruned Landmark Labeling); **Experimente auf Knopfdruck**: Labelgröße gegen Netzgröße und Dichte; Ordnung und Konstruktion; Abfrage gegen Netzgröße; alte Labels nach einer Kostenänderung.
4. **Wo die Annahmen enden** (Tabelle; feste Kosten, Speicher, Ordnung, nur die Entfernung, keine Uhrzeit) und **Mathematische Formulierung** (Deckungseigenschaft, kanonische Labels, PLL, Labels aus der CH, Aufwand als Lehrbuchwert gekennzeichnet).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, Ordnung der Hubs und – bei erzeugten Netzen – Größe, Dichte, Entfernung des Paares und Seed wählen. Die Adresszeile spiegelt die Konfiguration (Permalink). Höchstens 400 Knoten.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `hl_graph.py`, `hl_ch.py` | Graph in CSR-Form; Contraction Hierarchies (Kopie aus `contraction-hierarchies-demo`) als Ordnung und Vergleich |
| `hl_labels.py` | Labels: Suchräume der CH, Bereinigen, Pruned Landmark Labeling, Abfrage durch Verschmelzen, Route, Ordnungen, Dijkstra als Baseline |
| `hl_scenario.py` | Netze: kleines Netz, Stadtnetz, Zufallsnetz |
| `hl_evaluation.py` | Kennzahlen, Bildfolge, Messreihen, Kostenänderung |
| `hl_visualization.py`, `hl_presets.py`, `hl_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen networkx für **alle Paare** (Netze mit Parallelkanten, Nullkosten, unerreichbaren Zielen und identischem Start und Ziel); ein Regressionstest klickt "▶️ Abspielen" auf Netzen mit mehreren Bildern.
