"""Generate all_evaluation_immucan.ipynb from all_evaluation_covid.ipynb."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COVID = ROOT / "all_evaluation_covid.ipynb"
IMMUCAN = ROOT / "all_evaluation_immucan.ipynb"


def transform_source(src: str) -> str:
    if not src.strip():
        return src

    # --- global replacements ---
    repl = [
        ("from covid_data.load_covid import load_covid", ""),
        (
            "from scipy.stats import spearmanr\n",
            "from scipy.stats import spearmanr\n\n"
            "from immucan_data import immucan_model_config, load_immucan\n",
        ),
        (
            "from src.enums import Likelihood, WPrior, ZPrior\n"
            "from src.model_config import CohortPriorConfig, ModelConfig, "
            "SimpleViewConfig\n",
            "from src.enums import ZPrior\n",
        ),
        ("CACHE_DIR = '.fit_cache'", "CACHE_DIR = '.fit_cache_immucan'"),
        (
            "print(f'HEADLINE_K={HEADLINE_K}  PI={PI}  MAX_ITER={MAX_ITER}  "
            "SEED={SEED}')",
            "IMMUCAN_ROOT = '/Volumes/T7/immucan/results/IF/05_IF_table_extraction'\n"
            "IMMUCAN_PANELS = ('IF1',)\n"
            "IMMUCAN_COHORTS = None\n"
            "IMMUCAN_SAMPLE_IDS = None\n"
            "FIRST_N_PER_COHORT = 10\n"
            "REBUILD_CACHE = False   # ustaw True po zmianie FIRST_N_PER_COHORT / "
            "TUMOR_ROI_ONLY\n"
            "TUMOR_ROI_ONLY = True   # in.ROI.tumor_tissue == TRUE; False = wszystkie "
            "komórki z TSV\n\n"
            "print(f'HEADLINE_K={HEADLINE_K}  PI={PI}  MAX_ITER={MAX_ITER}  "
            "SEED={SEED}')\n"
            "print(f'panels={IMMUCAN_PANELS}  first_n_per_cohort={FIRST_N_PER_COHORT}  "
            "tumor_roi_only={TUMOR_ROI_ONLY}')",
        ),
        (
            "def fit_cohort(views, K, pi=PI, max_iter=MAX_ITER, seed=SEED):\n"
            "    '''CohortFACTM: WPrior.ARD_SS + ZPrior.COHORT, spike-and-slab prior "
            "`pi`.'''\n"
            "    cfg = ModelConfig(\n"
            "        simple_view_configs=[SimpleViewConfig(likelihood=Likelihood.NORMAL"
            ",\n"
            "                                              w_prior=WPrior.ARD_SS)\n"
            "                             for _ in range(views.num_simple)],\n"
            "        structured_view_configs=[],\n"
            "        z_priors=[ZPrior.COHORT] * K,\n"
            "        cohort_prior_config=CohortPriorConfig(pi=pi),\n"
            "    )\n"
            "    m = FACTModel(views=views, K=K, model_config=cfg, seed=seed)\n"
            "    m.fit(max_iter=max_iter, pretrain=True, elbo_tres=0.0)\n"
            "    return m\n",
            "def fit_cohort(views, K, pi=PI, max_iter=MAX_ITER, seed=SEED, "
            "n_topics=10):\n"
            "    '''CohortFACTM: simple (Normal+ARD) + structured CTM + "
            "ZPrior.COHORT.'''\n"
            "    cfg = immucan_model_config(\n"
            "        views, K, z_prior=ZPrior.COHORT, pi=pi, n_topics=n_topics\n"
            "    )\n"
            "    m = FACTModel(views=views, K=K, model_config=cfg, seed=seed)\n"
            "    m.fit(max_iter=max_iter, pretrain=True, elbo_tres=0.0)\n"
            "    return m\n",
        ),
        (
            "def fit_baseline(views, K, max_iter=MAX_ITER, seed=SEED):\n"
            "    '''Standardowy FACTM z izotropowym priorem N(0,1) — bez struktury "
            "kohortowej.'''\n"
            "    cfg = ModelConfig(\n"
            "        simple_view_configs=[SimpleViewConfig(likelihood=Likelihood.NORMAL"
            ",\n"
            "                                              w_prior=WPrior.ARD_SS)\n"
            "                             for _ in range(views.num_simple)],\n"
            "        structured_view_configs=[],\n"
            "        z_priors=[ZPrior.STD_NORMAL] * K,\n"
            "    )\n"
            "    m = FACTModel(views=views, K=K, model_config=cfg, seed=seed)\n"
            "    m.fit(max_iter=max_iter, pretrain=True, elbo_tres=0.0)\n"
            "    return m\n",
            "def fit_baseline(views, K, max_iter=MAX_ITER, seed=SEED, n_topics=10):\n"
            "    '''Baseline FACTM: simple + structured CTM, Z ~ N(0,1).'''\n"
            "    cfg = immucan_model_config(\n"
            "        views, K, z_prior=ZPrior.STD_NORMAL, n_topics=n_topics\n"
            "    )\n"
            "    m = FACTModel(views=views, K=K, model_config=cfg, seed=seed)\n"
            "    m.fit(max_iter=max_iter, pretrain=True, elbo_tres=0.0)\n"
            "    return m\n",
        ),
        (
            "def severity_from_label(label):\n"
            "    '''Etykieta kohorty -> wartość WHO (lub środek przedziału).'''\n"
            "    s = str(label).strip().lower()\n"
            "    if s in ('mild', 'moderate', 'severe'):\n"
            "        return {'mild': 0.5, 'moderate': 3.0, 'severe': 6.0}[s]\n"
            "    if s.startswith('sev') and s[3:].isdigit():\n"
            "        return float(s[3:])\n"
            "    if '-' in s:\n"
            "        a, b = s.split('-'); return (float(a) + float(b)) / 2\n"
            "    if s == 'non_hosp':\n"
            "        return 1.0\n"
            "    if s == 'hosp':\n"
            "        return 5.0\n"
            "    return float('nan')",
            "def severity_from_label(label):\n"
            "    '''Etykieta kohorty -> pseudo-ordinal (analog WHO). grouped/binary "
            "mają sens; by_type nominale.'''\n"
            "    s = str(label).strip().lower()\n"
            "    if s in ('breast', 'lung', 'kidney', 'head_neck'):\n"
            "        return {'breast': 1.0, 'lung': 2.0, 'kidney': 3.0, 'head_neck': "
            "4.0}[s]\n"
            "    if s in ('immucan', 'synergy'):\n"
            "        return {'immucan': 0.0, 'synergy': 1.0}[s]\n"
            "    type_ord = {'bc1': 0.0, 'nsclc': 1.0, 'rcc': 2.0, 'scchn1': 3.0, "
            "'syg_bc1': 4.0, 'scchn3': 5.0}\n"
            "    return type_ord.get(s, float('nan'))",
        ),
        (
            "        DATA[mode] = load_covid(cohort_mode=mode, standardize=True)",
            "        DATA[mode] = load_immucan(\n"
            "            globals().get('IMMUCAN_ROOT', "
            "'/Volumes/T7/immucan/results/IF/05_IF_table_extraction'),\n"
            "            panels=globals().get('IMMUCAN_PANELS', ('IF1',)),\n"
            "            cohorts=globals().get('IMMUCAN_COHORTS', None),\n"
            "            sample_ids=globals().get('IMMUCAN_SAMPLE_IDS', None),\n"
            "            first_n_per_cohort=globals().get('FIRST_N_PER_COHORT', 10),\n"
            "            cohort_mode=mode,\n"
            "            tumor_roi_only=globals().get('TUMOR_ROI_ONLY', True),\n"
            "            rebuild_cache=globals().get('REBUILD_CACHE', False),\n"
            "            standardize=True,\n"
            "            progress=True,\n"
            "        )",
        ),
        (
            "os.makedirs(CACHE_DIR, exist_ok=True)\n\n\n",
            "os.makedirs(CACHE_DIR, exist_ok=True)\n\n"
            "_DEFAULT_IMMUCAN_ROOT = "
            "'/Volumes/T7/immucan/results/IF/05_IF_table_extraction'\n\n\n",
        ),
        (
            "def get_data(mode):\n"
            "    if mode not in DATA:\n"
            "        DATA[mode] = load_immucan(\n"
            "            globals().get('IMMUCAN_ROOT', "
            "'/Volumes/T7/immucan/results/IF/05_IF_table_extraction'),\n"
            "            panels=globals().get('IMMUCAN_PANELS', ('IF1',)),\n"
            "            cohorts=globals().get('IMMUCAN_COHORTS', None),\n"
            "            sample_ids=globals().get('IMMUCAN_SAMPLE_IDS', None),\n"
            "            first_n_per_cohort=globals().get('FIRST_N_PER_COHORT', 10),\n"
            "            cohort_mode=mode,\n"
            "            tumor_roi_only=globals().get('TUMOR_ROI_ONLY', True),\n"
            "            rebuild_cache=globals().get('REBUILD_CACHE', False),\n"
            "            standardize=True,\n"
            "            progress=True,\n"
            "        )\n"
            "    return DATA[mode]",
            "def _data_cache_tag():\n"
            "    '''Krótki tag — RAM/dysk fit cache unieważnia się po zmianie próbki "
            "lub filtrów.'''\n"
            "    n = globals().get('FIRST_N_PER_COHORT', 10)\n"
            "    tumor = int(globals().get('TUMOR_ROI_ONLY', True))\n"
            "    panels = '-'.join(globals().get('IMMUCAN_PANELS', ('IF1',)))\n"
            "    return f'n{n}_roi{tumor}_{panels}'\n\n\n"
            "def get_data(mode):\n"
            "    tag = _data_cache_tag()\n"
            "    key = (mode, tag)\n"
            "    if key not in DATA:\n"
            "        DATA[key] = load_immucan(\n"
            "            globals().get('IMMUCAN_ROOT', _DEFAULT_IMMUCAN_ROOT),\n"
            "            panels=globals().get('IMMUCAN_PANELS', ('IF1',)),\n"
            "            cohorts=globals().get('IMMUCAN_COHORTS', None),\n"
            "            sample_ids=globals().get('IMMUCAN_SAMPLE_IDS', None),\n"
            "            first_n_per_cohort=globals().get('FIRST_N_PER_COHORT', 10),\n"
            "            cohort_mode=mode,\n"
            "            tumor_roi_only=globals().get('TUMOR_ROI_ONLY', True),\n"
            "            rebuild_cache=globals().get('REBUILD_CACHE', False),\n"
            "            standardize=True,\n"
            "            progress=True,\n"
            "        )\n"
            "    return DATA[key]",
        ),
        (
            "    key = (mode, kind, K, pi_key, max_iter, seed)",
            "    tag = _data_cache_tag()\n"
            "    key = (tag, mode, kind, K, pi_key, max_iter, seed)",
        ),
        (
            "    views = get_data(mode).views\n"
            "    t0 = time.time()\n"
            "    if kind == 'cohort':\n"
            "        m = fit_cohort(views, K=K, pi=pi, max_iter=max_iter, seed=seed)\n"
            "    elif kind == 'baseline':\n"
            "        m = fit_baseline(views, K=K, max_iter=max_iter, seed=seed)\n",
            "    data = get_data(mode)\n"
            "    views = data.views\n"
            "    n_topics = data.n_topics\n"
            "    t0 = time.time()\n"
            "    if kind == 'cohort':\n"
            "        m = fit_cohort(views, K=K, pi=pi, max_iter=max_iter, seed=seed, "
            "n_topics=n_topics)\n"
            "    elif kind == 'baseline':\n"
            "        m = fit_baseline(views, K=K, max_iter=max_iter, seed=seed, "
            "n_topics=n_topics)\n",
        ),
        (
            "    print(f'=== {mode} ===  N={d.views.N}  D_metab={d.views.simple[0].D}  "
            "D_prot={d.views.simple[1].D}')",
            "    v = d.views\n"
            "    dims = ', '.join(f'simple{vi}={v.simple[vi].D}' for vi in "
            "range(v.num_simple))\n"
            "    if v.num_structured:\n"
            "        dims += f'  structured={v.num_structured} (L={d.n_topics}, "
            "G={v.structured[0].G})'\n"
            "    print(f'=== {mode} ===  N={v.N}  {dims}')",
        ),
        (
            "view_names = ['metabolom', 'proteom']\n\n"
            "fig, axes = plt.subplots(1, views.num_simple, figsize=(5.2 * "
            "views.num_simple, 6))\n"
            "for v in range(views.num_simple):",
            "_panels = globals().get('IMMUCAN_PANELS', ('IF1',))\n"
            "view_names = [\n"
            "    _panels[i] if i < len(_panels) else f'simple_{i}'\n"
            "    for i in range(views.num_simple)\n"
            "]\n\n"
            "n_v = views.num_simple\n"
            "fig, axes = plt.subplots(1, n_v, figsize=(5.2 * max(n_v, 1), 6), "
            "squeeze=False)\n"
            "axes = axes.ravel()\n"
            "for v in range(n_v):",
        ),
        ("COVID-19 (Su et al.)", "Immucan IF"),
        (
            "(metabolom + proteom, `covid_data.load_covid`)",
            "(proporcje celltype / markery IF + okna spatial CTM, "
            "`immucan_data.load_immucan`)",
        ),
        ("na danych COVID", "na danych Immucan"),
        ("danych COVID", "danych Immucan"),
        ("8 kohortach", "5 kohortach"),
        (
            "Jedna kohorta na wartość WHO (`sev0`…`sev7`).",
            "Jedna kohorta na typ nowotworu (`BC1`, `NSCLC`, …) — alias `per_severity`.",  # noqa: E501
        ),
        (
            "Podział ordynalny na WHO = 3 (`0-2` vs `3-7`), tj. niehospitalizowani vs "
            "hospitalizowani.",
            "Podział binarny: kohorty IMMU vs SYNG (trial Synergy).",
        ),
        (
            "Tak jak w `covid_cohort_tests`: dwa widoki proste (metabolom + proteom), "
            "wyrównane po\n"
            "`sample_id`, z-score'owane per cecha. Etykieta kohorty pochodzi z WHO "
            "Ordinal Scale w\n"
            "trzech wariantach podziału. Poniżej liczności każdej partycji — niektóre "
            "poziomy\n"
            "nasilenia (`sev2`) są bardzo małe, więc ich efekt kohortowy będzie z "
            "natury niepewny.",
            "Widoki proste (MOFA-style per panel IF) + structured CTM (okna kNN z "
            "`tables/`).\n"
            "Próbki z-score'owane per cecha. Trzy partycje kohort (analog COVID):\n"
            "`per_severity` = typ nowotworu, `grouped` = grupa tkankowa, `binary` = "
            "IMMU vs SYNG.",
        ),
        (
            "**Część A (jak w covid_cohort_tests Exp 1).** Jedna kohorta na wartość "
            "WHO (`sev0`…`sev7`).\n"
            "Patrzymy na `E[γ]` (które czynniki aktywują się jako kohortowe) i "
            "`E[γ·δ]` (pełny profil\n"
            "przesunięć). Oczekiwanie: małe kohorty (`sev2`) mają niskie `γ` (prior je "
            "tłumi).",
            "**Część A (jak w covid_cohort_tests Exp 1).** Jedna kohorta na typ "
            "nowotworu.\n"
            "Patrzymy na `E[γ]` i `E[γ·δ]`. Oczekiwanie: małe kohorty mają niższe `γ` "
            "(prior je tłumi).",
        ),
        (
            "Duże `K` (15, 20) są kosztowne na\n378 próbkach — stąd cache.",
            "Duże `K` (15, 20) są kosztowne (structured CTM) — stąd cache.",
        ),
        ("378 próbkach", "N próbkach"),
        ("378 pacjentów", "N próbek"),
        (
            "WHO Ordinal Scale (0–7)",
            "pseudo-ordinal z typu nowotworu / grupy tkankowej",
        ),
        ("WHO", "pseudo-ordinal"),
        ("nasileniem WHO", "różnicą pseudo-ordinal"),
        ("nasilenia WHO", "pseudo-ordinal"),
        ("nasilenie WHO", "pseudo-ordinal"),
        ("nasileniem", "różnicą pseudo-ordinal"),
        ("nasilenia", "pseudo-ordinal"),
        ("nasilenie", "ordinal"),
        ("mild / moderate / severe", "breast / lung / kidney / head_neck"),
        ("non_hosp / hosp", "immucan / synergy"),
        ("metabolity/białka", "cechy IF (celltype / marker)"),
        ("ciężkości COVID", "typu nowotworu"),
        (
            "dla każdego **widoku** (metabolom / proteom) oraz dla każdej **kohorty**",
            "dla każdego **widoku prostego IF** oraz dla każdej **kohorty**",
        ),
        ("rośnie nasilenie", "kohorta"),
        ("|ρ(Z, WHO)| Spearman", "|ρ(Z, ordinal)| Spearman"),
        ("powiązany z nasileniem WHO", "powiązany z pseudo-ordinal"),
        (
            "Faktor najsilniej powiązany z nasileniem WHO",
            "Faktor najsilniej powiązany z pseudo-ordinal",
        ),
        (
            "## 6 — Test `pi` (z `01_pi_tests`) na danych COVID dla `K ∈ {3, 5, 10}`",
            "## 6 — Test `pi` (z `01_pi_tests`) na danych Immucan dla `K ∈ {3, 5, 10}`",
        ),
        (
            "**Exp 5 — odległość vs nasilenie WHO (kluczowy test).** Ponieważ skala "
            "WHO jest\n"
            "porządkowa, korelację odległości kohort w przestrzeni utajonej z |różnicą "
            "nasilenia|\n"
            "liczymy **Spearmanem** (rangami), nie Pearsonem.",
            "**Exp 5 — odległość vs pseudo-ordinal (analog WHO).** Dla `grouped` "
            "ordinal ma sens\n"
            "(grupy tkankowe). Dla `per_severity` typy nowotworu są **nominalne** — "
            "wynik Spearmana\n"
            "traktujemy tylko eksploracyjnie. Liczymy **Spearmanem** (rangami).",
        ),
        (
            "`binary` pomijamy w Exp 5 (tylko 1 para → korelacja nieokreślona).",
            "`binary` ma tylko 2 kohorty (1 para) — korelacja słabo zdefiniowana, ale "
            "raportujemy.",
        ),
        (
            "**Wniosek 6b.** `pi=0.5` to sweet spot: na `per_severity` korelacja rang "
            "z WHO",
            "**Wniosek 6b.** `pi=0.5` to sweet spot: na `per_severity` korelacja rang "
            "z pseudo-ordinal",
        ),
        (
            "> **Korelacje z nasileniem WHO liczymy Spearmanem (rangi), nie "
            "Pearsonem.** WHO Ordinal",
            "> **Korelacje z pseudo-ordinal liczymy Spearmanem (rangi), nie "
            "Pearsonem.** Dla `by_type`",
        ),
        ("~30 min.", "~20–35 min (structured CTM)."),
        ("(`sev0`/`sev1`)", "(sąsiednie typy / grupy)"),
        ("(`sev0`/`sev7`)", "(odległe typy / grupy)"),
        ("`sev0→sev7`", "grup tkankowych"),
        ('„oś ciężkości"', '„oś kohortowa"'),
        ("oś ciężkości", "oś kohortowa"),
        ("korelacja z różnicą pseudo-ordinal", "korelacja z pseudo-ordinal"),
        ("odległość vs ordinal pseudo-ordinal", "odległość vs pseudo-ordinal"),
        (
            "skala pseudo-ordinal jest\nordynalna, odległości kohort *powinny* "
            "korelować z `|sev_i − sev_j|`. Liczymy korelację\nPearsona",
            "dla `grouped` pseudo-ordinal ma sens (grupy tkankowe). Liczymy "
            "korelację\nSpearmana",
        ),
        (
            '|corr(Z, pseudo-ordinal)|` to „oś kohortowa" — jego centroidy powinny '
            "rosnąć/maleć monotonicznie z\n`grup tkankowych`",
            "|corr(Z, ordinal)|` to potencjalna oś kohortowa — centroidy w 7c pokazują "
            "separację typów",
        ),
    ]
    for old, new in repl:
        src = src.replace(old, new)

    # Section 9c — replace covid feature-name loader
    if "covid_data.load_covid as _lc" in src:
        src = (
            "def feature_names_for_view(long_df, view_name):\n"
            "    sub = long_df[long_df['view'] == view_name]\n"
            "    pivot = sub.pivot_table(index='sample', columns='feature', "
            "values='value', aggfunc='first')\n"
            "    return pivot.columns.tolist()\n\n"
            "# 9c — top cechy per faktor (które cechy IF najsilniej go budują)\n"
            "d = get_data(mode)\n"
            "feat_names = {}\n"
            "for vi, vname in enumerate(view_names):\n"
            "    panel = sorted(d.long_df['view'].unique())[vi]\n"
            "    feat_names[vname] = feature_names_for_view(d.long_df, panel)\n"
            "TOPN = 10\n\n"
            "rows = []\n"
            "for k in range(HEADLINE_K):\n"
            "    for v, vname in enumerate(view_names):\n"
            "        W = get_W(mc, v)\n"
            "        col = W[:, k]\n"
            "        order_pos = np.argsort(col)[::-1][:TOPN]\n"
            "        order_neg = np.argsort(col)[:TOPN]\n"
            "        rows.append({'faktor': f'Z{k}', 'widok': vname, 'kierunek': "
            "'top+',\n"
            "                     'cechy': ', '.join(feat_names[vname][i] for i in "
            "order_pos)})\n"
            "        rows.append({'faktor': f'Z{k}', 'widok': vname, 'kierunek': "
            "'top-',\n"
            "                     'cechy': ', '.join(feat_names[vname][i] for i in "
            "order_neg)})\n"
            "df_top = pd.DataFrame(rows)\n"
            "print(df_top.to_string(index=False))\n"
        )

    return src


def transform_markdown(src: str) -> str:
    title = (
        "# Ewaluacja CohortFACTM na danych Immucan IF — kompletny notebook\n\n"
        "Ten notebook jest **strukturalnym odpowiednikiem** "
        "`all_evaluation_covid.ipynb`:\n"
        "te same sekcje 0–10, te same testy, ale dane z **Immucan IF** "
        "(`load_immucan`).\n"
        "Porównujemy **CohortFACTM** (`ZPrior.COHORT`) z **baseline FACTM** "
        "(`ZPrior.STD_NORMAL`).\n\n"
        "**Dane.** Widoki proste (MOFA-style: proporcje celltype / tissue / markery "
        "IF) oraz\n"
        "structured CTM (okna kNN ze spatial `tables/`, domyślnie L=10 tematów). "
        "Komórki filtrowane\n"
        "do **`in.ROI.tumor_tissue == TRUE`** (`TUMOR_ROI_ONLY=True` w setupie; "
        "`False` = cały TSV).\n\n"
        "Mapowanie partycji kohort (analog COVID):\n\n"
        "| COVID | Immucan | Opis |\n"
        "|-------|---------|------|\n"
        "| `per_severity` | `per_severity` (= `by_type`) | jedna kohorta na typ "
        "nowotworu |\n"
        "| `grouped` | `grouped` | breast / lung / kidney / head_neck |\n"
        "| `binary` | `binary` | immucan vs synergy (SYNG_BC1) |\n\n"
        "**Brak ground-truth.** Jako miękki analog WHO używamy **pseudo-ordinal** z "
        "etykiet\n"
        "kohort (grupy tkankowe mają sens porządkowy; typy nowotworu — tylko "
        "eksploracyjnie).\n\n"
        "> **Czas.** Fit-y cache'owane w `.fit_cache_immucan/`; agregaty i okna "
        "spatial w `data/`.\n"
        "> Domyślnie `FIRST_N_PER_COHORT=10` (~50 TSV, 5 kohort; po filtrze tumor ROI "
        "zwykle ~35+ próbek). Structured CTM + siatka K×pi —\n"
        "> pełny przebieg ~20–35 min przy `MAX_ITER=60`. Po zmianie `TUMOR_ROI_ONLY` "
        "ustaw\n"
        "> `REBUILD_CACHE=True`.\n"
    )
    if src.startswith("# Ewaluacja CohortFACTM"):
        return title
    return transform_source(src)


def clear_cell_outputs(cell: dict) -> dict:
    cell = copy.deepcopy(cell)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def main() -> None:
    nb = json.loads(COVID.read_text())
    out_cells = []
    for cell in nb["cells"]:
        cell = clear_cell_outputs(cell)
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            src = transform_markdown(src)
        else:
            src = transform_source(src)
        cell["source"] = [src] if src.endswith("\n") else [src + "\n"] if src else []
        if src and not src.endswith("\n"):
            cell["source"] = [src]
        out_cells.append(cell)

    nb["cells"] = out_cells
    nb["metadata"]["kernelspec"] = nb.get("metadata", {}).get("kernelspec", {})
    IMMUCAN.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"Wrote {IMMUCAN} ({len(out_cells)} cells)")


if __name__ == "__main__":
    main()
