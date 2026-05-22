# Miniproject CoBAR 2026 — Controlling a fly via olfaction

Contrôleur biomimétique pour la mouche du projet BIOENG-456. La navigation vers
la banane est **100 % olfactive** (les coordonnées de la cible ne sont jamais
utilisées pour décider du cap ; seul un critère d'arrêt physique à 2 mm les lit).

Pour la description détaillée du pipeline interne, voir [`SUBMISSION.md`](SUBMISSION.md).

## 1. Installation

Dépendances gérées par `uv` (voir le README racine pour installer `uv`, Git et
FFmpeg).

```sh
# Depuis la racine du repo
uv sync
```

Cette commande crée `.venv/` et installe tous les paquets nécessaires (`flygym`,
`mujoco`, `pygame`, `numpy`, etc.).

## 2. Lancer le contrôleur sur un niveau

### Windows (PowerShell / cmd)

```sh
.\.venv\Scripts\python.exe miniproject/run_with_controller.py --level 2 --seed 42
```

### macOS / Linux

```sh
.venv/bin/python miniproject/run_with_controller.py --level 2 --seed 42
```

Une fenêtre `pygame` s'ouvre avec la vue caméra et l'overlay choisi. Elle est
automatiquement bornée à ~90 % du moniteur (ratio préservé).

**Touches dans la fenêtre :**
- `ESPACE` → reset la simulation
- `ÉCHAP` → quitter

## 3. Options CLI

```sh
.venv/bin/python miniproject/run_with_controller.py --level <N> --seed <N> [options]
```

| Option | Effet |
|---|---|
| `--level N` | Difficulté : `0` plat, `1` +terrain, `2` +gazon, `3` +vent, `4` +libellule |
| `--seed N` | Graine aléatoire (défaut 42) |
| `--max-steps N` | Cap sur le nombre de pas physiques (défaut 100 000 ≈ 10 s simulés) |
| `--no-display` | Mode headless (pas de fenêtre), utile pour benchmarker |
| `--render-fly-vision` | Affiche la vision brute (œil gauche + œil droit) au-dessus des caméras |
| `--debug-vision` | Affiche l'overlay de détection vision (silhouette + bbox du pic ciblé) |
| `--progress-every N` | Imprime une ligne de progrès tous les N pas (0 désactive) |

`--debug-vision` a la priorité sur `--render-fly-vision` si les deux sont passés.

## 4. Exemples

```sh
# L0 (arène plate), seed 1, fenêtre standard
.venv/bin/python miniproject/run_with_controller.py --level 0 --seed 1

# L2 (gazon), avec overlay vision pour voir ce que la mouche détecte
.venv/bin/python miniproject/run_with_controller.py --level 2 --seed 67 --debug-vision

# L4 (le plus dur), headless, max 200 000 steps, log tous les 10 000
.venv/bin/python miniproject/run_with_controller.py \
    --level 4 --seed 777 --no-display --max-steps 200000 --progress-every 10000
```

## 5. Mode interactif (clavier)

Pour piloter manuellement la mouche au clavier (utile pour explorer un niveau) :

```sh
.venv/bin/python miniproject/run_interactive.py --level 2 --seed 42
```

Touches : `W A S D` pour bouger, `Q` pour arrêter, `ÉCHAP` pour quitter.

## 6. Notebook d'évaluation

Le notebook `run_controller.ipynb` reproduit la même boucle que
`run_with_controller.py` mais dans un environnement Jupyter — c'est l'équivalent
de ce qui sera utilisé pour évaluer la soumission.

## 7. Structure du dossier

```
miniproject/
├── README.md             ← ce fichier
├── SUBMISSION.md         ← description détaillée du contrôleur
├── run_with_controller.py ← exécutable principal (pygame + CLI)
├── run_interactive.py    ← mode interactif (clavier)
├── run_controller.ipynb  ← équivalent notebook
└── submission/
    ├── controller.py     ← LE contrôleur
    └── __init__.py
```

## 8. Dépannage

- **Fenêtre noire (Linux)** : ajouter `--dont-use-pygame-rendering` (nécessite `uv pip install pynput`).
- **MuJoCo sans display** (serveur headless) : exporter `MUJOCO_GL=egl` et `PYOPENGL_PLATFORM=egl` avant de lancer.
- **FPS très bas en pygame** : utiliser `--no-display` pour benchmark, sinon réduire la taille fenêtre n'est pas exposé en CLI (modifier la valeur de scale dans `run_with_controller.py`).
