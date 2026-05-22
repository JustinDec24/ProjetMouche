# Submission — CoBAR 2026 Miniproject

Contrôleur biomimétique permettant à la mouche d'atteindre une source d'odeur
(banane) sur 5 niveaux de difficulté. La navigation est **100 % olfactive** :
les coordonnées de la banane ne sont **jamais** lues pour décider du cap. La
seule exception est un critère d'arrêt physique de sécurité à 2 mm.

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

La fenêtre s'adapte automatiquement à ~90 % du moniteur (ratio préservé).

## 🐝 Architecture du contrôleur

Le contrôleur combine **chimiotaxie olfactive** (pour le cap) et **répulsion
vision** (pour l'évitement). Boucle de décision à 40 Hz (toutes les 25 ms).

```
   sim.get_olfaction()  →  _read_olfaction (EMA + noise floor + formule notebook)
                                │
                                ▼
                        (mean_odor, effective_norm, raw_asym)
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ Stop physique  (||thorax − banana|| ≤ 2 mm  OR  mean_odor > 5e-3)│
   └─────────────────────────────────────────────────────────────────┘
                                │  (sinon)
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  ALIGN initial (pivot dur jusqu'à dépassement de l'axe banane)  │
   └─────────────────────────────────────────────────────────────────┘
                                │  (puis)
                                ▼
   _compute_target_bias  →  (target_bias saturé, bearing smooth)
                                │
   panorama bi-oculaire         │
   _vision_step  →  spikes  ────┤
                                ▼
   _compute_drives  →  fusion (target_bias × BLEND + vision_repulsion + slope_bias)
                                │
                                ▼
                       drives = [left, right]
                                │
                                ▼
                       TurningController (CPG locomotion)
```

## 1️⃣ Lecture olfactive — `_read_olfaction`

Calque 1:1 sur `notebooks/week4/solutions_olfaction.ipynb` :

```python
# 4 capteurs : [palp_L, palp_R, antenna_L, antenna_R], canal 0 = attractif
attractive_intensities = np.average(
    odor[:, 0].reshape(2, 2),                    # [[palp_L, palp_R], [ant_L, ant_R]]
    axis=0, weights=[9, 1],                       # palpes 9× plus que antennes
)                                                 # → [L_combined, R_combined]
mean_odor       = attractive_intensities.mean()
raw_asym        = (L − R) / mean_odor             # ∈ [-2, +2]
attractive_bias = -500 × raw_asym                 # gain énorme du notebook
effective_norm  = tanh(attractive_bias²) × sign(attractive_bias)   # ∈ [-1, +1]
```

Le **gain -500** est crucial : il sature `effective_norm` à `±1` dès que
`|raw_asym| ≳ 0.01`, donc le côté de la source est détecté de manière fiable
même quand l'asymétrie est minuscule.

### Pré-traitements

- **EMA temporel conditionnel au vent** (`ODOR_EMA_ALPHA_WIND = 0.017`,
  τ ≈ 1.5 s à 40 Hz) : appliqué UNIQUEMENT sur L3/L4. Le sim change la
  direction du vent toutes les 100 ms (uniform 0-360°) ; moyenner ~10-15
  cycles de rotation du plume récupère le signal **time-averaged** qui est
  isotrope, comme sans vent. Sans vent (L0-L2) : aucun lissage, brut direct.
- **Noise floor** (`NOISE_FLOOR_ASYM = 0.001`) : si `|raw_asym| < 0.001`,
  force `raw_asym = 0`. Évite la réaction au jitter numérique (~1e-4)
  près de la source.

## 2️⃣ Calcul du steering — `_compute_target_bias`

```python
# Pull vers la source (saturé pour décision tranchée)
target_bias = effective_norm × OLF_TARGET_BIAS_SCALE     # 3.0
target_bias *= damping_derivatif                          # voir ci-dessous

# Pseudo-bearing pour ALIGN / GO_PIVOT (smooth, proportionnel à l'erreur)
bearing = -clip(raw_asym × OLF_BEARING_GAIN, -1, 1) × π/2     # OLF_BEARING_GAIN = 1.0
```

### Damping dérivatif (A4)

Réduit le pull quand `raw_asym` change vite (= la mouche est en train de
croiser l'axe banane) → amortit l'overshoot :

```python
d_asym = raw_asym − raw_asym_prev
damping = 1 / (1 + 50 × |d_asym|)        # ASYM_DAMPING_GAIN = 50
target_bias = effective_norm × 3.0 × damping
```

### Séparation `target_bias` / `bearing`

C'est la clé pour éviter les deux modes pathologiques d'ALIGN :

| Signal | Formule | Saturation | Usage |
|--------|---------|------------|-------|
| `target_bias` | `effective_norm × 3` | Saturé ±3 | Steering instantané (tranchant) |
| `bearing` | `−raw_asym × π/2` | Proportionnel à l'erreur | Seuils ALIGN / GO_PIVOT |

Convention de signes (préservée partout) :
- `target_bias < 0` → tourne à GAUCHE
- `bearing     > 0` → banane à GAUCHE (signe opposé à `target_bias`)

## 3️⃣ ALIGN initial — forced pivot avec fallback search

```python
# Direction = -sign(effective_norm) (saturé, fiable même quand raw_asym tiny)
if |effective_norm| ≥ 0.5:
    align_dir = -sign(effective_norm)
    pivot dur jusqu'à sign_flip(effective_norm) ou timeout 60 décisions
else:
    # ALIGN-SEARCH : avance doucement (drives ÷ 2) jusqu'à voir un signal exploitable
    return [base_drive × 0.5, base_drive × 0.5]
```

**Pourquoi cette séparation `target_bias` / `bearing`** : un bearing dérivé
directement d'`effective_norm` (saturé à ±π/2) maintiendrait ALIGN actif
indéfiniment (la mouche ferait des 360° sans s'arrêter). En basant `bearing`
sur `raw_asym` (proportionnel à l'erreur), ALIGN se désengage dès que la
mouche est ~face à la source.

**Pourquoi ALIGN-SEARCH** : quand la mouche spawn perpendiculaire à la banane,
`raw_asym` est sous le noise floor, donc `effective_norm = 0` et ALIGN ne sait
pas dans quelle direction pivoter. Au lieu de skip et partir tout droit dans
une direction random, elle **avance lentement** jusqu'à percevoir un côté
clair, puis ALIGN s'engage proprement.

## 4️⃣ Vision — détection et répulsion

À partir du niveau 2 (gazon présent) :

1. **Segmentation foreground vert** sur le panorama bi-oculaire raw.
2. **Courbe silhouette** : row du pixel vert le plus haut par colonne.
3. **Baseline** = rolling-min sur fenêtre large (≈ niveau du terrain local).
4. **`excess = silhouette − baseline`** (combien le pic dépasse le sol).
5. **`score = excess² × centralité⁴`** → on retient un seul pic, le plus gênant.
6. **Filtre aspect-ratio** (h/w ≥ 1.5) pour rejeter les collines / pentes.

### Fusion vision × olfaction

Trois modes selon la proximité (déterminée par `mean_odor`) :

| Cas | Condition | Action |
|-----|-----------|--------|
| **Sprint final** | `mean_odor > 5e-6` (ODOR_SPRINT_THRESHOLD) | Bypass vision, suit `target_bias` direct |
| **REPUL-PIVOT** | Pic central ≥ 3000 px | Pivot serré (`[max, min]`) côté opposé au pic |
| **REPUL doux** | Pic latéral ≥ 2000 px | `bias = vision_repulsion + REPULSION_BANANA_BLEND × target_bias` |

`REPULSION_BANANA_BLEND = 1.0` : le pull olfactif compte **autant** que la
répulsion vision (au lieu d'1/2 dans la version précédente), pour éviter que
la mouche soit massivement déviée par les herbes à mi-chemin de la banane.

## 5️⃣ Robustesse vent / chute — L3 / L4

### Tilt freeze v2
Détection précoce de basculement (`uprightness < 0.75`) :
- Drives à 0 + grip max sur toutes les pattes.
- Reste en freeze ≥ 15 décisions, exit uniquement si `uprightness > 0.92`.
- `TILT_LEAN_GAIN × 2` pendant le freeze pour compensation roll renforcée.

### Head-collision recovery (3 phases)
Si choc frontal détecté (force tête > 3 N) :
1. **BACKUP** : recule en ligne droite ~45 décisions (≈ 1 s).
2. **PIVOT** : pivote vers le côté présumé de la banane (signe du dernier `target_bias`).
3. **ARC** : courbe contournante autour de l'obstacle.

## 6️⃣ Critères d'arrêt

```python
if ||thorax − banana_xy|| ≤ STOP_DIST (= 2 mm):   stop physique
if mean_odor > STOP_ODOR_THRESHOLD (= 5e-3):       stop olfactif (fallback)
```

**`banana_xy` n'est utilisé QUE pour le stop physique** — jamais pour le cap.
Le steering reste purement olfactif tout au long de la course.

## 📊 Constantes clé

| Constante | Valeur | Rôle |
|-----------|--------|------|
| `OLF_ATTRACTIVE_GAIN` | -500 | Gain notebook week4 (sature `effective_norm`) |
| `OLF_TARGET_BIAS_SCALE` | 3.0 | Échelle du pull olfactif |
| `OLF_BEARING_GAIN` | 1.0 | Échelle `raw_asym → bearing` |
| `NOISE_FLOOR_ASYM` | 0.001 | Plancher de bruit sur `raw_asym` (D1) |
| `ASYM_DAMPING_GAIN` | 50 | Amortissement dérivatif sur `target_bias` (A4) |
| `ODOR_EMA_ALPHA_WIND` | 0.017 | Lissage olfactif conditionnel au vent (τ ≈ 1.5 s) |
| `ODOR_SPRINT_THRESHOLD` | 5e-6 | Bypass de la vision en approche finale |
| `STOP_ODOR_THRESHOLD` | 5e-3 | Stop olfactif quand on est sur la source |
| `STOP_DIST` | 2 mm | Stop physique (le seul usage de `banana_xy`) |
| `REPULSION_BANANA_BLEND` | 1.0 | Poids du pull olfactif dans le mode REPUL |
| `ALIGN_BEARING_OK` | 0.20 | Seuil de sortie d'ALIGN (en bearing) |
| `ALIGN_MAX_DECISIONS` | 60 | Timeout d'ALIGN (~1.5 s) |

## 📊 Performances

Validations historiques (avant les derniers réglages D1+A4+ALIGN-SEARCH) :
- Seeds publics (1, 67, 777) × niveaux 2-3-4 : 9 / 9 atteints.
- Sweep level 4 (50 seeds aléatoires) : ~48 / 50 atteints.

Validations actuelles :
- Level 0, seeds {0, 1, 2, 67, 777} : 5 / 5 atteints (~1.7-1.9 s par seed).

Le code a évolué depuis les sweep complets — re-benchmark recommandé avant
rendu final.

## 🗂 Structure de la soumission

```
miniproject/
├── SUBMISSION.md             ← ce fichier
├── README.md                 ← README du prof (général)
├── run_with_controller.py    ← exécutable principal (pygame + CLI)
├── run_interactive.py        ← mode interactif (WASD au clavier)
├── run_controller.ipynb      ← équivalent notebook
└── submission/
    ├── controller.py         ← LE contrôleur (~1500 lignes)
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

Une simulation `--no-display` prend ~5-25 s wall-clock par seed (~3000 sim-steps/s
en mode headless).
