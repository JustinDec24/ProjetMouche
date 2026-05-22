# Submission — CoBAR 2026 Miniproject

Contrôleur biomimétique permettant à la mouche d'atteindre une source d'odeur
(banane) en évitant les obstacles, sur 5 niveaux de difficulté.

## ⚡ Démarrage rapide

```sh
# Installer les dépendances (une fois)
uv sync

# Lancer une simulation (niveau 2, seed 42)
.\.venv\Scripts\python.exe miniproject/run_with_controller.py --level 2 --seed 42
```

Sur macOS/Linux remplacer `.\.venv\Scripts\python.exe` par `.venv/bin/python`.

## 🎮 Options de lancement

```sh
.\.venv\Scripts\python.exe miniproject/run_with_controller.py \
    --level <0-4> --seed <int> [options]
```

| Option | Effet |
|---|---|
| `--level N` | Difficulté : 0 plat / 1 +terrain / 2 +gazon / 3 +vent / 4 +libellule |
| `--seed N` | Graine aléatoire (défaut 42) |
| `--max-steps N` | Cap sur le nombre de pas physiques (défaut 100 000 ≈ 10 s simulés) |
| `--no-display` | Mode headless (pas de fenêtre, pour benchmarker) |
| `--render-fly-vision` | Affiche la vision brute de la mouche au-dessus des caméras |
| `--debug-vision` | Affiche l'overlay de détection (silhouette + bbox du pic ciblé) |
| `--progress-every N` | Imprime une ligne de progrès tous les N pas (0 = désactivé) |

Touches dans la fenêtre : `ESPACE` = reset, `ÉCHAP` = quitter.

## 🐝 Aperçu du contrôleur

Le contrôleur combine **vision** + **olfaction** pour naviguer :

### Navigation (olfaction uniquement)
Les coordonnées de la banane ne sont jamais lues. La mouche s'oriente via
l'**asymétrie L/R** des 4 capteurs olfactifs (2 palpes + 2 antennes) :
- `asym = (odor_R − odor_L) / (odor_R + odor_L + ε)` ∈ [-1, +1]
- `bearing = -π × asym` détermine le côté à tourner
- `mean_odor` croît à l'approche → choisit le gain de steering (proche vs loin)
- Stop déclenché par `mean_odor > STOP_ODOR_THRESHOLD`

### Détection des obstacles (vision)
À partir du niveau 2 (gazon présent) :
1. Segmentation foreground vert sur le panorama binoculaire raw
2. Calcul de la courbe silhouette (row du pixel vert le plus haut par colonne)
3. Baseline = rolling-min sur fenêtre large (≈ niveau du terrain local)
4. `excess = silhouette_height − baseline` (combien le pic dépasse le sol)
5. `score = excess² × centralité⁴` → on retient **un seul pic** : le plus gênant
6. Filtre aspect-ratio (h/w ≥ 1.5) pour rejeter les collines

### Stratégies d'évitement
| Cas | Action |
|---|---|
| Pic central (\|p\| ≤ 0.30) et taille ≥ 2000 px | Pivot serré (drives 1:8) |
| Pic moyen (1200–3000 px) ou latéral | Esquive douce par arc (drives asymétriques continus) |
| Approche finale (`mean_odor > ODOR_SPRINT_THRESHOLD`) | Sprint direct, pas de répulsion |

### Robustesse vent / chute (niveaux 3-4)
**Tilt freeze v2** — détection précoce de basculement :
- Si `uprightness < 0.75` → drives à 0 + grip max sur toutes les pattes
- Stay en freeze ≥ 15 décisions, exit uniquement si `uprightness > 0.92`
- `TILT_LEAN_GAIN × 2` pendant le freeze pour active roll comp renforcée

## 📊 Performances (validation)

Sweep `level 4` (le plus dur : terrain + gazon + vent + libellule), 50 seeds aléatoires :
- **48 / 50 reached** (96 %)

Sur les seeds obligatoires (1, 67, 777) × niveaux 2-3-4 : **9 / 9** réussis.

## 🗂 Structure de la soumission

```
miniproject/
├── SUBMISSION.md             ← ce fichier
├── README.md                 ← README du prof (général)
├── run_with_controller.py    ← exécutable principal
├── run_interactive.py        ← mode interactif (WASD au clavier)
├── run_controller.ipynb      ← équivalent notebook
└── submission/
    ├── controller.py         ← LE contrôleur
    └── __init__.py
```

Le contrôleur utilise `MiniprojectSimulation` (défini dans `src/miniproject/`)
qui gère l'arène, le rendu, le vent et la libellule.

## 🛠 Reproductibilité

Pour rejouer un sweep représentatif :

```sh
# 10 seeds aléatoires sur level 4
for s in $(python -c "import random; random.seed(0); print(*[random.randint(1,9999) for _ in range(10)])"); do
    .\.venv\Scripts\python.exe miniproject/run_with_controller.py \
        --level 4 --seed $s --no-display --max-steps 120000
done
```

Une simulation `--no-display` prend ~60 secondes wall-clock (~1700 sim-steps/s).
