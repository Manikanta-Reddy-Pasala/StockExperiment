# Cap-Filter A/B Test — Skip Small / Add Mid?

User question: take one model, skip small-cap or add mid-cap, see if gains improve.

Tested both directions on 2 models with full 3-year backtest, ₹10L start, all trades comparable.

## Test 1 — Add Midcap150 to momentum_n100_top5_max1

Baseline: real NSE Nifty 100 (Large only, 104 stocks)
Variant: Nifty 100 ∪ Midcap 150 (Large + Mid, ~254 stocks)
Strategy: monthly rotation, lb=30, top-1 by 30d return (same as baseline)

| Variant | Final NAV | CAGR | Max DD | Trades | WR | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (Large only)** | ₹58.69 L | **+80.38%** | 29.71% | 31 | 74.2% | **2.71** |
| + Midcap150 (Large + Mid) | ₹58.44 L | +80.12% | 34.33% | 30 | 76.7% | 2.33 |

**Verdict: Adding Mid does NOT help.** CAGR essentially flat (+80.12% vs +80.38%), Max DD WORSE (+4.62pp). Larger universe pulls in mid-caps that compete with large-cap winners but don't deliver — picks like NETWEB/COCHINSHIP/GRSE divert capital from MAZDOCK/IRFC/ADANIPOWER.

Recommend: **keep momentum_n100_top5_max1 as Large-only.**

---

## Test 2 — Exclude cap categories from n20_daily_30d_mc1_uptrend

Baseline: top-20 ADV + uptrend filter (Large + Mid + Small mix)
Strategy: daily rotation, lb=30, top-1 by 30d return

| Variant | Final NAV | CAGR | Max DD | Trades | WR | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (all caps)** | ₹1.70 Cr | **+157.11%** | 50.61% | 134 | 47.8% | 3.11 |
| Excl Small | ₹1.29 Cr | +134.47% | 56.02% | 140 | 47.1% | 2.40 |
| **Excl Small + Mid (Large only)** | ₹1.40 Cr | **+140.78%** | **25.52%** | 139 | 43.1% | **5.52** |

**Verdict: Excluding ONLY Small is WORSE** — strategy loses access to mid-cap winners that compensate for small-cap losses. CAGR -23pp, DD +5pp.

**Excluding BOTH Small + Mid (Large only) is BEST risk-adjusted** — CAGR -16pp from baseline but Max DD HALVED (50.6% → 25.5%). Calmar 5.52 vs 3.11 = much better risk profile.

Recommend: **add `n20_daily_30d_mc1_uptrend_large_only` variant** if reduced-DD version wanted. Or apply Large filter to existing live signal as toggle.

---

## Cross-model insight

| Model | Best universe | CAGR | DD |
|---|---|---:|---:|
| momentum_n100_top5_max1 | Large only (NSE Nifty 100) — baseline | +80.38% | 29.71% |
| n20_daily_30d_mc1_uptrend | **Large only filter** | +140.78% | **25.52%** |

Pattern: **Large-cap is sufficient at the apex of strategies.** Mid+Small add noise + DD without proportional return. This contradicts the intuition that "wider universe = more winners" — wider universe = more noise.

**Why Large-only n20_daily wins risk-adjusted:**
- Strategy still picks top-1 by 30d momentum
- Confines selection to high-quality large-caps with strong 30d move
- Avoids speculative pumps in mid/small that mean-revert hard
- Same daily rotation discipline keeps it responsive

## Recommendation

1. Keep `momentum_n100_top5_max1` as-is (Large only) — confirmed optimal universe choice.
2. Consider **adding `n20_daily_large_only` variant** if a Calmar-5.5 strategy is desired (CAGR +140% / DD 25%). Could co-exist with full n20_daily as risk-conservative alternative.
3. **Do NOT add mid-cap to N100 model** — A/B shows no benefit.

## Reproduce

```bash
docker cp /Users/manip/Documents/codeRepo/StockExperiment/src/data/symbols/nifty_smallcap250.csv trading_system_app:/app/src/data/symbols/
docker exec trading_system_app python /tmp/cap_filter_tests.py
```

See `/tmp/cap_filter_results.json` for full trade ledgers per variant.
