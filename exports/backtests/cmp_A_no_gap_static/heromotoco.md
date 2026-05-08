# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 5325.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 6 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** -0.08% / 0.27%
- **Sum % (uncompounded):** -0.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | -0.08% | -0.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 0 | 5 | 2 | -0.08% | -0.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | -0.08% | -0.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 5355.00 | 5706.86 | 5707.38 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 10:15:00 | 5742.00 | 5700.03 | 5699.87 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 5619.50 | 5699.71 | 5699.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 5590.00 | 5696.43 | 5698.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.53 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 5477.50 | 5637.56 | 5660.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 5468.50 | 5635.88 | 5659.10 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-04 14:15:00 | 5499.00 | 5629.74 | 5655.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 5490.00 | 5628.35 | 5654.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-09 09:15:00 | 5427.50 | 5613.66 | 5645.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 10:15:00 | 5397.00 | 5611.50 | 5643.97 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5639.15 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5639.15 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5639.15 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-12 09:15:00 | 5419.50 | 5605.11 | 5637.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 5446.00 | 5603.52 | 5636.50 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 5173.70 | 5587.40 | 5627.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 5335.50 | 5313.87 | 5444.86 | SL hit (close>ema200) qty=0.50 sl=5313.87 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 5461.00 | 5315.47 | 5434.96 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 5343.50 | 5320.07 | 5434.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5269.50 | 5319.56 | 5434.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 13:15:00 | 5006.02 | 5281.01 | 5386.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 5255.50 | 5197.32 | 5314.77 | SL hit (close>ema200) qty=0.50 sl=5197.32 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-07 13:15:00 | 5379.50 | 5201.05 | 5310.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 5335.50 | 5202.39 | 5310.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-03-04 10:15:00 | 5468.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-03-04 15:15:00 | 5490.00 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2026-03-09 10:15:00 | 5397.00 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2026-03-12 10:15:00 | 5446.00 | 2026-03-13 10:15:00 | 5173.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 10:15:00 | 5446.00 | 2026-04-08 09:15:00 | 5335.50 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2026-04-13 10:15:00 | 5269.50 | 2026-04-23 13:15:00 | 5006.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 10:15:00 | 5269.50 | 2026-05-06 09:15:00 | 5255.50 | STOP_HIT | 0.50 | 0.27% |
