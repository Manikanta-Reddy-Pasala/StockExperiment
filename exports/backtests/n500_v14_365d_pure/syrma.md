# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1100.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 0
- **Avg / median % per leg:** -1.84% / -3.13%
- **Sum % (uncompounded):** -12.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 1 | 4 | 0 | -0.36% | -1.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 1 | 20.0% | 1 | 4 | 0 | -0.36% | -1.8% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.53% | -11.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.53% | -11.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 1 | 14.3% | 1 | 6 | 0 | -1.84% | -12.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 538.00 | 478.72 | 478.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 545.05 | 481.17 | 479.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:45:00 | 522.10 | 526.51 | 511.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 524.50 | 526.42 | 512.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 522.75 | 525.97 | 512.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 524.40 | 525.92 | 512.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 512.80 | 525.66 | 513.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 512.80 | 525.66 | 513.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 513.95 | 525.54 | 513.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 511.15 | 525.54 | 513.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 512.45 | 525.41 | 513.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 512.45 | 525.41 | 513.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | SL hit (close<static) qty=1.00 sl=511.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | SL hit (close<static) qty=1.00 sl=511.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | SL hit (close<static) qty=1.00 sl=511.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | SL hit (close<static) qty=1.00 sl=511.25 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 506.35 | 525.24 | 513.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 510.00 | 524.93 | 513.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 509.40 | 524.93 | 513.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 519.50 | 522.44 | 512.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 13:45:00 | 524.00 | 522.37 | 513.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-01 13:15:00 | 576.40 | 531.80 | 519.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 733.60 | 795.88 | 795.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 728.20 | 792.81 | 794.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 11:15:00 | 753.45 | 753.04 | 769.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:00:00 | 753.45 | 753.04 | 769.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 769.95 | 713.24 | 737.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 780.15 | 713.24 | 737.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 776.50 | 719.33 | 739.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 764.00 | 719.84 | 739.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 794.35 | 721.02 | 740.02 | SL hit (close>static) qty=1.00 sl=787.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 12:00:00 | 765.55 | 721.97 | 740.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 819.75 | 724.50 | 741.13 | SL hit (close>static) qty=1.00 sl=787.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 882.70 | 755.38 | 755.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 890.60 | 775.75 | 765.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 09:15:00 | 743.25 | 780.05 | 780.12 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 09:15:00 | 816.00 | 779.89 | 779.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 837.55 | 787.18 | 783.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:45:00 | 522.10 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-06-13 15:15:00 | 524.50 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-06-16 14:45:00 | 522.75 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-06-17 09:15:00 | 524.40 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-24 13:45:00 | 524.00 | 2025-07-01 13:15:00 | 576.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-01 15:15:00 | 764.00 | 2026-02-02 09:15:00 | 794.35 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-02-02 12:00:00 | 765.55 | 2026-02-03 09:15:00 | 819.75 | STOP_HIT | 1.00 | -7.08% |
