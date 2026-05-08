# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 621.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 3 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 15.02% / 15.00%
- **Sum % (uncompounded):** 90.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 6 | 100.0% | 0 | 3 | 3 | 15.02% | 90.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 0 | 3 | 3 | 15.02% | 90.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 15.02% | 90.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 767.00 | 763.62 | 763.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 771.75 | 763.70 | 763.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 15:15:00 | 762.80 | 764.52 | 764.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 15:15:00 | 762.80 | 764.52 | 764.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 762.80 | 764.52 | 764.07 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 752.15 | 763.64 | 763.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 747.50 | 762.24 | 762.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 761.70 | 758.16 | 760.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 761.70 | 758.16 | 760.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 761.70 | 758.16 | 760.50 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-09 12:15:00 | 750.80 | 760.01 | 761.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 749.40 | 759.90 | 761.13 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 09:15:00 | 744.55 | 759.33 | 760.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 748.50 | 759.23 | 760.72 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 14:15:00 | 748.05 | 758.87 | 760.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 749.40 | 758.78 | 760.45 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:15:00 | 636.99 | 703.22 | 720.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:15:00 | 636.23 | 703.22 | 720.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:15:00 | 636.99 | 703.22 | 720.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 636.45 | 627.54 | 662.32 | SL hit (close>ema200) qty=0.50 sl=627.54 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 636.45 | 627.54 | 662.32 | SL hit (close>ema200) qty=0.50 sl=627.54 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 636.45 | 627.54 | 662.32 | SL hit (close>ema200) qty=0.50 sl=627.54 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-09 13:15:00 | 749.40 | 2026-03-12 10:15:00 | 636.99 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-13 10:15:00 | 748.50 | 2026-03-12 10:15:00 | 636.23 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 749.40 | 2026-03-12 10:15:00 | 636.99 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-09 13:15:00 | 749.40 | 2026-04-15 09:15:00 | 636.45 | STOP_HIT | 0.50 | 15.07% |
| SELL | retest2 | 2026-01-13 10:15:00 | 748.50 | 2026-04-15 09:15:00 | 636.45 | STOP_HIT | 0.50 | 14.97% |
| SELL | retest2 | 2026-01-13 15:15:00 | 749.40 | 2026-04-15 09:15:00 | 636.45 | STOP_HIT | 0.50 | 15.07% |
