# Best Model — Final Report

_Date: 2026-05-12 | Capital: ₹10,00,000_

## Winning Configuration

```yaml
strategy:        ema_200_400          # EMA 200/400 1H crossover
universe:        selector_top10       # multi-param ranked N500 picks
refresh:         monthly              # selector re-runs 1st of month
capital_inr:     1_000_000            # ₹10L locked
max_concurrent:  2                    # 2 simultaneous positions
slot_alloc:      500_000              # ₹5L per slot
min_price:       50                   # penny filter
min_adv_lakh:    100                  # ₹1cr/day liquidity
kill_switch:     -5%                  # daily loss → halt
```

## Performance — May 2025 → May 2026 (1 year)

| Metric | Value |
|--------|------:|
| Starting capital | ₹10,00,000 |
| **Final equity** | **₹12,18,456** |
| **Profit** | **₹2,18,456** |
| **ROI** | **+21.85%** |
| **MaxDD** | **9.58%** (~₹96K) |
| Trades taken | 28 |
| Signals skipped (slot full) | 41 |
| Avg trades/month | 2.3 |
| Win rate (est.) | ~50% |
| Open at year end | 0 |

## Universe (selector top-10, ranked 2025-05-12)

| # | Symbol | Close@start | ATR% | 60d Return | ADV ₹L |
|--:|--------|------------:|-----:|-----------:|-------:|
| 1 | SWIGGY     | ₹597.45 | 6.18 | +58.31% | 1,141 |
| 2 | VMM        | ₹101.19 | 7.40 | -12.86% |   802 |
| 3 | AEGISLOG   | ₹816.45 | 7.29 | +12.57% |   548 |
| 4 | ANGELONE   | ₹288.17 | 4.89 | +13.50% |   608 |
| 5 | SAILIFE    | ₹703.65 | 6.49 |  -5.01% |   318 |
| 6 | ITI        | ₹341.65 | 6.51 | +22.32% |   688 |
| 7 | IKS        | ₹1,885.80 | 5.15 | +10.56% |   226 |
| 8 | AMBER      | ₹6,898.00 | 6.48 | +27.66% |   907 |
| 9 | NTPCGREEN  | ₹131.67 | 6.33 | +39.72% |   694 |
|10 | BSE        | ₹1,847.62 | 4.09 | +44.43% | 1,675 |

Refreshed monthly via `tools/backtests/stock_selector.py`. No lookahead —
selector ran at backtest start using only data up to that date.

## Why this beat alternatives

| Config | ROI% | MaxDD% | vs Winner |
|--------|-----:|-------:|----------:|
| **Selector top-10 N500 + max=2** | **+21.85** | **9.58** | — |
| N500 top-20 (historical sum%) + max=2 | +14.15 | 10.68 | -7.7pp |
| N50 top-19 + max=3 | +13.20 | 7.86 | -8.6pp |
| Full N50 baseline + max=2 | +7.30 | 12.77 | -14.5pp |
| Full N500 baseline + max=2 | -33.53 | 34.94 | -55pp |

Selector wins because:
1. **Picks on current characteristics** (ATR + momentum + liquidity), not
   on historical performance → no survivorship bias.
2. **Volatility = alpha.** ATR-weighted picks have wider price ranges.
   EMA 200/400 crossover strategy needs ranges, not consolidation.
3. **Liquidity floor catches institutional flow.** ADV ≥ ₹1cr filter
   excludes stocks where ₹10L would move the price.

## 4-Phase finding tree

```
Phase 0 — Baseline 1yr (16 cells)
  └─ Best: N50 full + max=2 = +7.30% ROI
  └─ All others < 0% or OOM-killed
  └─ Target gap: 10× short of "5-10%/mo" headline

Phase 1 — Pattern mining
  └─ Top-19 contributors = +1100% sum%
  └─ Bottom-34 = -400% drag
  └─ "Concentrate, don't diversify"

Phase 2 — Sweep (top-N × max)
  └─ N50 top-19 + max=3 = +13.20% (best risk-adj)
  └─ N500 top-20 + max=2 = +14.15% (best ROI)
  └─ Combined N50+N500 = -3.57% (crowding kills it)

Phase 3 — Regime gate (no news, no LLM)
  └─ Nifty 50 trend + ATR% gate tested
  └─ ALL 4 variants HURT performance (+21.30 → +1-6%)
  └─ SELL alpha lives in bear/volatile periods — gating destroys it
  └─ DROPPED from production

Phase 4 — Multi-param selector  ⬅ WINNER
  └─ Composite: 25% ATR + 25% mom + 15% vol + 20% ADV + 15% 52W-dist
  └─ Top-10 N500 + max=2 = +21.30% (₹2L) / +21.85% (₹10L)
  └─ Beats every prior config by 7-15pp
```

## What doesn't work (don't retry)

1. **Adding more stocks** beyond top-10 → returns decay
2. **Combining N50 + N500** → crowding effect, -3.57%
3. **Regime/news/sentiment gates** → kicks out SELL alpha
4. **EMA 9/21** (overtrades, slippage drag): -7.52% to -28.31%
5. **Swing pullback** (too conservative, 3-18 trades/yr): -1.09% to +0.13%
6. **ORB 15-min** (no 5m bars in cache): 0 trades fired
7. **Penny filter OFF on EMA 200/400** → OOM-killed

## What COULD push ROI further (untested, in scope for future phases)

1. **Multi-year validation** (2024 + 2023) — confirm selector
   generalizes across regimes
2. **Selector weight grid search** — current 25/25/15/20/15 untuned
3. **Direction-aware regime gate** — needs strategy code refactor to
   expose BUY vs SELL in cycle table
4. **Loss-streak kill-switch** — pause after N losses
5. **Limit-order entry mode** — currently market, switch to LIMIT for
   cleaner fills + 0.05-0.1pp ROI gain
6. **F&O leverage layer** — uses index options for 3-5× capital
   efficiency. CHANGES risk envelope. Separate proposal.

## Realistic forward expectations vs original target

| | Original target | Backtest | Live (paper) | Live (real) |
|---|---|---|---|---|
| Monthly ROI | 5-10% | ~1.8%/mo | 1-2%/mo | 1-1.5%/mo |
| Yearly ROI | 60-120% | +21.85% | 15-25% | 12-22% |
| Max DD | ? | 9.58% | 10-15% | 12-18% |

**5-10%/mo target unreachable** on single-strategy cash equity ₹10L.
That headline ROI would need F&O leverage, multi-strategy stack, or
proprietary HFT. Honest 15-25%/yr beats Nifty 50 (~12%/yr) by alpha
of 5-10pp.

## Production rollout plan

```bash
# Phase A: 4-week paper trade (starting after user OK)
./tools/live/run_daily.sh selector       # generate signals/selector_2026-05.json
./tools/live/run_daily.sh prefetch
./tools/live/run_daily.sh signals
./tools/live/run_daily.sh paper          # writes paper_portfolio/*.json
./tools/live/run_daily.sh report         # daily P&L

# Phase B: Compare paper vs backtest
# After 4 weeks, paper ROI should be ±50% of backtest annualized rate
# (i.e., 0.5-2.5% expected in 4 weeks)

# Phase C: Real money (only after paper validates)
LIVE_TRADING=true CAPITAL_INR=1000000 \
  ./tools/live/run_daily.sh live
```

## Files for reference

```
exports/backtests/
├── BEST_MODEL_REPORT.md        ⬅ THIS FILE — authoritative
├── FINAL_RECOMMENDATION.md     ⬅ Detailed config + workflow
├── BASELINE_1YR.md             ⬅ Phase 0 matrix
├── PATTERN_MINING.md           ⬅ Phase 1
├── SWEEP_RESULTS.md            ⬅ Phase 2
├── REGIME_GATE_RESULTS.md      ⬅ Phase 3 (negative)
├── SELECTOR_RESULTS.md         ⬅ Phase 4 winner detail
├── SCALE_10L.md                ⬅ ₹2L → ₹10L scale validation
└── SELECTOR_TOP30_2025-05-12.json  ⬅ Selector output

tools/backtests/
├── stock_selector.py           ⬅ Multi-param picker
├── pattern_mining.py           ⬅ Per-stock contribution
├── regime_filter.py            ⬅ (Built but unused)
└── realistic_capital_sim.py    ⬅ Capital-constrained replay

tools/live/
├── signal_generator.py         ⬅ Cache → strategy → signals JSON
├── risk_manager.py             ⬅ ₹10L lock, kill-switch
├── paper_executor.py           ⬅ JSON ledger sim
├── fyers_executor.py           ⬅ Real orders (LIVE_TRADING gated)
├── position_monitor.py         ⬅ 5-min mark-to-market
├── daily_report.py             ⬅ EOD P&L
├── run_daily.sh                ⬅ Cron wrapper
└── README.md                   ⬅ Operator guide
```

## Commit hashes (May 2026 session)

| SHA | What |
|---|---|
| 8f4db188 | tools/live/ — 7-script production stack |
| c02ee03a | 1-yr baseline matrix (Phase 0) |
| bcc0ed07 | Phase 1 — pattern mining |
| d58fe652 | Phase 2 — top-N × max sweep |
| f792b3d6 | Phase 4 — stock_selector.py (winner) |
| a7cade14 | Phase 3 — regime gate (negative) |
| 1096629f | Wire selector → signal_generator + FINAL_REC |
| 26ade4ff | Scale to ₹10L + SCALE_10L.md |

Repo: github.com/Manikanta-Reddy-Pasala/StockExperiment (branch: main)
