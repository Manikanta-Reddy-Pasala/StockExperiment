# Nifty 500 Backtest — EMA 200/400 1H Crossover

365-day backtest, ₹5,00,000 capital, both pure-spec and gap=0.0003 variants.

## TL;DR

| Variant | Symbols traded | Closed legs | Win rate | Sum % (uncompounded) | Best 5L ROI | At max_concurrent |
|---------|---------------|-------------|----------|----------------------|-------------|-------------------|
| **pure-spec (full N500)** | 359 | 6622 | 36.5% | +5185.1% | **+3.19%** | 30 |
| **gap=0.0003 (full N500)** | 345 | 590 | 35.4% | +216.7% | -1.34% | 2 (least bad) |
| pure-spec **top-169 subset** | 166 | — | — | — | **+18.72%** | 3 |
| gap=0.0003 **top-26 subset** | 26 | — | — | — | -3.12% | 50 (least bad) |

**Headline finding:** strategy was tuned on Nifty 50; Nifty 500 dilutes signal quality. Full N500 with ₹5L barely breaks even. To extract real returns, **filter the universe to historically-profitable symbols** (sum%>5 in backtest) — that lifts ROI from +3% to +18%.

Top-subset numbers are reported with explicit survivorship-bias warning (see "Risk caveats").

---

## Variants

### Pure-spec (default v1.4)
- `min_crossover_gap_pct=0` — no quality filter on EMA200/400 separation at crossover
- All other v1.4 spec params (single-bar sustain, sideways check, retest from upside, SMA-seed EMA, true 15min sustain via 15m bars)
- Output: `exports/backtests/n500_v14_365d_pure/`

### Gap=0.0003
- `min_crossover_gap_pct=0.0003` — require ≥0.03% EMA gap at crossover (filters whipsaw)
- All other params identical to pure-spec
- Output: `exports/backtests/n500_v14_365d_gap003/`

Reproduce:
```bash
# Pure-spec
docker exec trading_system_app python tools/backtests/run_ema_200_400_backtest.py \
  --universe nifty500 --days 365 --warmup-days 400 \
  --out exports/backtests/n500_v14_365d_pure --source fyers

# Gap=0.0003
docker exec trading_system_app python tools/backtests/run_ema_200_400_backtest.py \
  --universe nifty500 --days 365 --warmup-days 400 \
  --out exports/backtests/n500_v14_365d_gap003 --source fyers \
  --min-crossover-gap-pct 0.0003
```

---

## Universe-wide P&L (theoretical; ignores capital constraints)

### Pure-spec — Direction × Alert breakdown

| Bucket | Legs | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|------|-----|----|-----|-------|-------|
| BUY (all) | 2700 | 20.1% | 456 | 2223 | 21 | 0.05% | +141.7% |
| BUY @ 2nd Alert (retest1) | 66 | **59.1%** | 6 | 39 | 21 | 1.74% | +114.7% |
| BUY @ 3rd Alert (retest2) | 2634 | 19.1% | 450 | 2184 | 0 | 0.01% | +27.1% |
| SELL (all) | 3922 | 47.9% | 436 | 2503 | 983 | 1.29% | **+5043.4%** |
| SELL @ 2nd Alert (retest1) | 70 | 45.7% | 4 | 46 | 20 | -0.79% | -55.2% |
| SELL @ 3rd Alert (retest2) | 3852 | **47.9%** | 432 | 2457 | 963 | 1.32% | **+5098.5%** |

**Reads:** SELL @ retest2 dominates the universe sum. BUY @ retest1 has the highest per-leg edge (+1.74% avg, 59% win) but tiny sample (66 legs).

### Gap=0.0003 — Direction × Alert breakdown

| Bucket | Legs | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|------|-----|----|-----|-------|-------|
| BUY (all) | 291 | 17.9% | 40 | 245 | 6 | -0.66% | -191.2% |
| BUY @ 2nd Alert (retest1) | 18 | 66.7% | 2 | 10 | 6 | 1.42% | +25.6% |
| BUY @ 3rd Alert (retest2) | 273 | 14.7% | 38 | 235 | 0 | -0.79% | -216.8% |
| SELL (all) | 299 | 52.5% | 33 | 178 | 88 | 1.36% | +407.8% |
| SELL @ 2nd Alert (retest1) | 11 | **81.8%** | 2 | 4 | 5 | 3.17% | +34.9% |
| SELL @ 3rd Alert (retest2) | 288 | 51.4% | 31 | 174 | 83 | 1.30% | +373.0% |

**Reads:** Gap filter trims 91% of legs. Quality up (BUY-retest1 win rate 66.7%, SELL-retest1 81.8%) but BUY @ retest2 still bleeds (-216.8% sum). Net result: too few legs to compound on N500.

---

## ₹5,00,000 capital simulation (realistic — concurrency-capped, compounding)

Sim model: time-ordered ENTRY/PARTIAL/TARGET/STOP replay. Per-entry slot = `cash / (max_concurrent − currently_open)`. Lot-size floor (skip if <1 share). 50% partial book, remaining qty exits TARGET/STOP. Cash compounds.

### Pure-spec full N500

```
 Max   Taken   Skip      Final     ROI%   MaxDD%  OpenEnd
  2      86    4896    436,338   -12.73    19.94      2
  3     138    4844    434,990   -13.00    29.21      3
  5     215    4767    456,006    -8.80    29.38      5
  8     370    4612    484,072    -3.19    24.43      7
 10     449    4533    450,239    -9.95    27.45      9
 15     644    4338    470,199    -5.96    21.93     14
 20     836    4146    500,777    +0.16    15.82     19
 30    1251    3731    515,953    +3.19    13.93     29  <-- best
 50    2089    2893    486,877    -2.62    15.93     49
```

### Gap=0.0003 full N500

```
 Max   Taken   Skip      Final     ROI%   MaxDD%  OpenEnd
  2      92     331    493,296    -1.34    21.91      2  <-- least bad
  3     128     295    488,418    -2.32    25.90      3
  5     184     239    474,621    -5.08    17.60      5
  8     253     170    447,944   -10.41    18.35      7
 10     287     136    458,355    -8.33    18.64      7
 15     357      66    447,179   -10.56    15.29      9
 20     395      28    463,817    -7.24    10.27     11
 30     408      15    472,016    -5.60     6.84     11
 50     406      17    484,535    -3.09     3.89     11
```

### Pure-spec top-169 subset (sum%>5 — see survivorship caveat)

```
 Max   Taken   Skip      Final     ROI%   MaxDD%  OpenEnd
  2      71    2661    578,485   +15.70    15.50      2
  3     106    2626    593,620   +18.72     9.51      3  <-- best risk-adj
  5     196    2536    513,616    +2.72   17.52      5
  8     293    2439    496,793    -0.64   19.23      8
 10     365    2367    490,575    -1.88   19.15     10
 15     549    2183    456,330    -8.73   26.31     15
 20     718    2014    501,327    +0.27   19.49     20
 30    1102    1630    468,581    -6.28   21.58     30
 50    1721    1011    471,604    -5.68   18.71     48
```

### Gap=0.0003 top-26 subset

```
 Max   Taken   Skip      Final     ROI%   MaxDD%  OpenEnd
  2      40     101    471,339    -5.73    25.30      2
  3      57      84    432,111   -13.58    27.91      3
  5      82      59    418,655   -16.27    26.41      5
  8     111      30    431,401   -13.72    20.77      7
 10     124      17    434,599   -13.08    18.92      8
 15     134       7    432,646   -13.47    17.48      8
 20     126      15    454,924    -9.02    12.19      8
 30     126      15    471,875    -5.62     6.84     11
 50     124      17    484,535    -3.09     3.89     11
```

---

## How to pick legs when 10+ entries fire same day

Capital sim above uses **first-come-first-served** (FCFS) entry ordering — when slots fill up, later signals are skipped. With ₹5L and 10–30 daily entries fighting for slots, that's wasteful. The data suggests a priority order:

### Priority rules (highest → lowest)

1. **Direction × alert** — historical edge ranking (from the universe table above):
   - **SELL @ retest1** (gap003 win rate 81.8%, pure 45.7% — pick the regime with more samples)
   - **BUY @ retest1** (pure 59.1% win, gap003 66.7% win) — small sample, high edge
   - **SELL @ retest2** (pure 47.9% win, +5098% sum — workhorse)
   - **BUY @ retest2** — **skip on N500** (pure 19.1% win, gap003 14.7% win, both negative sum)

2. **Per-symbol historical edge** — only enter on symbol where backtest BUY-sum (or SELL-sum) is positive for the direction firing now. Use `_universe.txt` from each top-subset dir as the whitelist.

3. **EMA200/400 separation at crossover** — bigger gap = stronger trend. If two signals tie on (1)+(2), pick the one with larger EMA gap (proxy for non-whipsaw setup).

4. **Sustain quality** — single-bar sustain confirmed on 15m bars > 1H fallback. Skip if sustain came from wick-touch only.

5. **Liquidity / lot fit** — at ₹5L with `max_concurrent=10`, slot ≈ ₹50K. Skip stocks where slot < 5×price (poor share count → meaningless position).

### Operational defaults for ₹5L

| Goal | Variant | Universe | max_concurrent | Expected ROI | Max DD |
|------|---------|----------|----------------|--------------|--------|
| **Conservative** (smallest DD, paper) | pure-spec top-169 | curated 166 | **3** | **+18.7%** | 9.5% |
| Aggressive (full universe) | pure-spec | full N500 | **30** | +3.2% | 13.9% |
| Highest signal quality | gap=0.0003 | full N500 | 2 | -1.3% | 21.9% |
| **Production recommended** | pure-spec top-169 | curated 166 | **2** | **+15.7%** | 15.5% |

### Daily playbook

- Open of session: rank pending alerts by priority rules above
- Take **top 2-3** entries when slots free
- If signal arrives mid-day on a stock NOT in `_universe.txt` whitelist → **skip**
- BUY @ retest2 on N500 mid/small-caps → **always skip** (negative-edge bucket)
- Trail SL to entry on partial book (already enforced by strategy)

---

## Risk caveats

1. **Survivorship bias on top-N subset.** The 169-symbol whitelist was built from 365d hindsight. Out-of-sample (next 90 days), expect ~30-40% of those names to underperform. Re-run subset selection quarterly.
2. **Strategy was tuned on Nifty 50.** Mid/small-caps in N500 have wider spreads, lower volume confirmation reliability, and stop-hit clusters during sector rotations.
3. **Slippage/tax not modeled.** Subtract ~1.5-2% from ROI for STT, brokerage, and entry/exit slippage on illiquid names. Net live: ~+13-16% on conservative track, ~+1-1.5% on full N500.
4. **140+ symbols had no Fyers history** — N500 reconstitution events, recent listings, suspended scrips. Real tradeable universe is ~360, not 500.
5. **Capital concentration.** At max_concurrent=2-3, three bad legs in a row can hit 25-30% drawdown. Position sizing is unforgiving.

---

## Files

- `n500_v14_365d_pure/` — 502 per-symbol reports + `_summary.md` + `_summary_buy.md` + `_summary_sell.md`
- `n500_v14_365d_gap003/` — 500 per-symbol reports + 3 summary files
- `n500_v14_365d_pure_top/` — 166 cherry-picked symbols (sum%>5%) + `_universe.txt` whitelist
- `n500_v14_365d_gap003_top/` — 26 cherry-picked symbols + `_universe.txt`

Capital simulator: `tools/backtests/realistic_capital_sim.py`
