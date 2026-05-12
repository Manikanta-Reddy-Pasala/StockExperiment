# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 8851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 0
- **Avg / median % per leg:** -2.41% / -4.95%
- **Sum % (uncompounded):** -33.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -4.48% | -53.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -4.48% | -53.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 2 | 14.3% | 2 | 12 | 0 | -2.41% | -33.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 6575.00 | 6433.30 | 6432.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 6626.50 | 6446.86 | 6439.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 6444.00 | 6459.40 | 6446.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 6444.00 | 6459.40 | 6446.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 6462.00 | 6459.43 | 6446.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 6486.00 | 6458.52 | 6446.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 6472.50 | 6458.79 | 6446.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 09:15:00 | 7134.60 | 6601.30 | 6528.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 7327.50 | 7824.65 | 7825.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 13:15:00 | 7264.00 | 7708.54 | 7763.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 6220.00 | 6140.24 | 6521.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:30:00 | 6176.00 | 6144.37 | 6514.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:15:00 | 6165.00 | 6144.37 | 6514.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 11:15:00 | 6580.50 | 6184.87 | 6503.42 | SL hit (close>static) qty=1.00 sl=6560.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 7815.00 | 6741.32 | 6739.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 7909.50 | 6804.35 | 6771.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 7312.50 | 7367.95 | 7122.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 12:00:00 | 7312.50 | 7367.95 | 7122.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 7009.50 | 7367.46 | 7144.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 6968.00 | 7367.46 | 7144.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 7088.50 | 7364.68 | 7144.43 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 6567.50 | 6990.22 | 6991.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 6482.00 | 6962.72 | 6977.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 7048.00 | 6811.69 | 6892.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 7048.00 | 6811.69 | 6892.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 6966.00 | 6813.23 | 6893.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 6903.00 | 6818.93 | 6894.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 7108.00 | 6822.77 | 6893.32 | SL hit (close>static) qty=1.00 sl=7065.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 7715.00 | 6955.85 | 6954.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 7874.00 | 6964.99 | 6959.11 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 12:45:00 | 6459.00 | 2025-05-20 10:15:00 | 6510.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-16 14:30:00 | 6458.50 | 2025-05-20 10:15:00 | 6510.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-20 11:30:00 | 6445.00 | 2025-05-21 09:15:00 | 6764.00 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2025-05-23 14:00:00 | 6408.50 | 2025-05-27 09:15:00 | 6580.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-06-04 09:30:00 | 6299.50 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2025-06-04 10:15:00 | 6302.50 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2025-06-05 14:15:00 | 6304.50 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-06-06 09:15:00 | 6276.00 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2025-06-09 10:15:00 | 6335.00 | 2025-06-09 11:15:00 | 6660.00 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest2 | 2025-06-20 11:00:00 | 6486.00 | 2025-07-02 09:15:00 | 7134.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 14:15:00 | 6472.50 | 2025-07-02 09:15:00 | 7119.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-03 14:30:00 | 6176.00 | 2026-02-06 11:15:00 | 6580.50 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2026-02-03 15:15:00 | 6165.00 | 2026-02-06 11:15:00 | 6580.50 | STOP_HIT | 1.00 | -6.74% |
| SELL | retest2 | 2026-04-09 09:15:00 | 6903.00 | 2026-04-10 09:15:00 | 7108.00 | STOP_HIT | 1.00 | -2.97% |
