"""Guard: the 2026-06-13 backtest REALISM regeneration (next-open fills + real
Fyers CNC charges + PIT universe fixes) must stay reflected everywhere the
backtest numbers are displayed/documented — Settings strategy cards, README
results table, exports/models docs — and the exports summary.json copies must
stay in sync with the authoritative tools/models/<m>/summary.json.

If a future re-run changes summary.json, refresh the docs via
tools/analysis/refresh_export_docs.py and update the Settings/README cards,
then this test's expected headline strings.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "emerging_momentum",
    "momentum_n100_top5_max1",
    "momentum_pseudo_n100_adv",
    "momentum_retest_n500",
    "n40",
]

# cagr_pct / max_dd_pct / calmar straight from the regenerated summary.json
EXPECTED = {
    "emerging_momentum": (129.35, 35.17, 3.68),  # 2026-06-13 POOL 100->80, window 2021-03->2026-06
    "momentum_n100_top5_max1": (49.51, 49.13, 1.01),
    "momentum_pseudo_n100_adv": (65.57, 44.89, 1.46),  # 2026-06-13 nosml rework
    "momentum_retest_n500": (57.64, 34.03, 1.69),
    "n40": (32.01, 38.86, 0.82),  # 2026-06-13 stop re-tune 12%->10%
}


def _summary(model: str) -> dict:
    return json.loads((ROOT / "tools" / "models" / model / "summary.json").read_text())


def test_summaries_carry_realism_convention_markers():
    for m in MODELS:
        d = _summary(m)
        assert d.get("fill_at_next_open") is True or d.get("fill_convention") == "next_open", \
            f"{m}: summary.json missing next-open fill marker"
        assert d.get("charges_model") == "fyers_cnc", f"{m}: charges_model missing"
        assert d.get("charges_total") or d.get("total_charges_inr"), \
            f"{m}: no charges figure in summary.json"


def test_summaries_match_expected_headline():
    for m, (cagr, dd, calmar) in EXPECTED.items():
        d = _summary(m)
        assert abs(d["cagr_pct"] - cagr) < 0.01, f"{m}: cagr {d['cagr_pct']} != {cagr}"
        assert abs(d["max_dd_pct"] - dd) < 0.01, f"{m}: dd {d['max_dd_pct']} != {dd}"
        assert abs(d["calmar"] - calmar) < 0.01, f"{m}: calmar {d['calmar']} != {calmar}"


def test_exports_summary_json_in_sync_with_tools():
    for m in MODELS:
        tools_j = (ROOT / "tools" / "models" / m / "summary.json").read_text()
        exp_j = (ROOT / "exports" / "models" / m / "summary.json").read_text()
        assert tools_j == exp_j, f"{m}: exports/models summary.json out of sync — re-copy + run refresh_export_docs.py"


def test_settings_cards_quote_new_net_numbers():
    html = (ROOT / "src" / "web" / "templates" / "v2" / "settings.html").read_text()
    for needle in (
        "+129.3% CAGR",   # emerging full (2026-06-13 pool 100->80)
        "+49.5% CAGR",    # n100 full
        "+65.6% CAGR",    # pseudo full (nosml rework 2026-06-13)
        "+57.6% CAGR",    # retest full
        "+32.0% CAGR",    # n40 full (2026-06-13 stop re-tune 12%->10%)
        "net of charges",
        "next-open fills",
    ):
        assert needle in html, f"settings.html missing: {needle}"
    # the collapsed/biased pseudo headline must not survive as current
    assert "+12.7% CAGR" not in html, "stale collapsed pseudo headline still in settings.html"


def test_pseudo_nosml_flagged_in_settings():
    html = (ROOT / "src" / "web" / "templates" / "v2" / "settings.html").read_text()
    assert "nosml" in html, "pseudo nosml rework note missing from settings.html"
    assert "survivorship" in html, "pseudo survivorship-bias explanation missing from settings.html"


def test_readme_table_updated():
    md = (ROOT / "README.md").read_text()
    for needle in ("+129.35%", "+57.64%", "+49.51%", "+32.01%", "+65.57%",
                   "net of real Fyers CNC charges", "next-open fills"):
        assert needle in md, f"README.md missing: {needle}"


def test_exports_index_regenerated():
    md = (ROOT / "exports" / "models" / "SUMMARY.md").read_text()
    for needle in ("+129.3%", "+57.6%", "+49.5%", "+32.0%", "+65.6%",
                   "net of real Fyers CNC charges"):
        assert needle in md, f"exports/models/SUMMARY.md missing: {needle}"


def test_settings_template_parses():
    """Render-check: the edited template must still be valid Jinja2."""
    import jinja2
    src = (ROOT / "src" / "web" / "templates" / "v2" / "settings.html").read_text()
    jinja2.Environment().parse(src)  # raises TemplateSyntaxError on breakage
