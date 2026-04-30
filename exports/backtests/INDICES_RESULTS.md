# Indian Indices — EMA 200/400 1H Backtest

_Generated: 2026-04-30T23:50 IST_

Source: Yahoo chart API, 1H bars, 720 calendar days (~5015 1H candles each).
Target rule per image spec: **5000 absolute index points**. Stop: 1H close
on the wrong side of EMA400.

## Headline numbers

| Metric | Value |
|--------|-------|
| Indices tested | 5 (NIFTY 50, BANKNIFTY, FINSERVICE, IT, AUTO) |
| Total trades | 37 |
| Trade-level winners | 10 |
| Trade-level win rate | 27.0% |
| Target hits | **0** |
| EMA400 close-exits | 37 |
| Sum P&L per unit | −7393.35 |

## Per-index

| Index | Signals | Closed | Winners | Tgt | EMA | P&L |
|-------|---------|--------|---------|-----|-----|-----|
| NIFTY 50 (^NSEI) | 40 | 10 | 4 | 0 | 10 | +907.05 |
| BANKNIFTY (^NSEBANK) | 56 | 9 | 2 | 0 | 9 | −5727.25 |
| FINSERVICE (^CNXFIN) | 38 | 7 | 2 | 0 | 7 | +778.80 |
| IT (^CNXIT) | 44 | 7 | 1 | 0 | 7 | −2708.30 |
| AUTO (^CNXAUTO) | 23 | 4 | 1 | 0 | 4 | −643.65 |

## Key finding: 5000-pt target unreachable on 1H

Across 720 days × 5 indices, **zero target hits**. NIFTY at 22k–25k means a
5000-pt target = 20–22% absolute move. On 1H timeframe, EMA400 cross-back
always fires before that level is reached.

**Implication for production:**
- Either lengthen target hold duration (multi-week swing)
- Or reduce target for 1H scalping (e.g. 200-500 pts on NIFTY)
- Or use the equity 1:3 RR rule for indices too (configurable in
  `StrategyConfig.target_points` or via `INDEX_SYMBOLS` set)

The image spec says "Target – 5000 points" but assumes higher timeframe / longer
hold. For 1H entry/scalp use, consider a tighter target.

## Per-cycle detail

Open `exports/backtests/indices/<symbol>.md` for full cycle breakdown
(BUY + SELL setups with each stage, prices, EMAs, notes).
