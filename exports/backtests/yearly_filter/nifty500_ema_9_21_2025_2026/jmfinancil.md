# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 145.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 59 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 23 |
| ALERT3 | 123 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 69 |
| PARTIAL | 25 |
| TARGET_HIT | 6 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 57 / 36
- **Target hits / Stop hits / Partials:** 6 / 62 / 25
- **Avg / median % per leg:** 1.89% / 4.09%
- **Sum % (uncompounded):** 175.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 6 | 28.6% | 2 | 19 | 0 | 0.43% | 9.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 6 | 28.6% | 2 | 19 | 0 | 0.43% | 9.0% |
| SELL (all) | 72 | 51 | 70.8% | 4 | 43 | 25 | 2.31% | 166.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 51 | 70.8% | 4 | 43 | 25 | 2.31% | 166.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 93 | 57 | 61.3% | 6 | 62 | 25 | 1.89% | 175.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 104.45 | 100.51 | 100.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 106.10 | 103.92 | 102.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 117.65 | 117.82 | 115.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 117.65 | 117.82 | 115.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 116.00 | 117.29 | 116.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 116.00 | 117.29 | 116.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 116.65 | 117.16 | 116.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 118.30 | 117.40 | 116.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 130.13 | 126.71 | 124.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 146.03 | 148.45 | 148.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 145.99 | 147.67 | 148.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 142.87 | 142.69 | 144.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:45:00 | 142.88 | 142.69 | 144.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 144.59 | 143.18 | 144.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 144.59 | 143.18 | 144.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 143.00 | 143.14 | 143.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 142.42 | 142.95 | 143.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 141.83 | 143.33 | 143.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 142.15 | 143.24 | 143.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 142.05 | 141.30 | 141.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 141.40 | 141.41 | 141.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 141.90 | 141.41 | 141.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 140.17 | 141.16 | 141.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 139.74 | 140.55 | 141.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 139.61 | 140.41 | 140.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 152.88 | 142.58 | 141.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 152.88 | 142.58 | 141.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 156.21 | 150.03 | 146.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 154.59 | 154.61 | 151.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 154.59 | 154.61 | 151.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 158.99 | 160.07 | 158.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 158.68 | 160.07 | 158.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 158.85 | 159.71 | 158.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 158.64 | 159.71 | 158.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 158.96 | 159.77 | 158.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 159.24 | 159.77 | 158.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 159.32 | 159.68 | 158.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 159.32 | 159.68 | 158.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 158.50 | 159.44 | 158.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 159.50 | 159.44 | 158.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 159.71 | 159.25 | 158.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 159.40 | 159.27 | 158.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:30:00 | 161.58 | 159.95 | 159.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 163.22 | 161.02 | 159.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 159.46 | 161.02 | 159.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 165.27 | 164.82 | 162.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 163.40 | 164.82 | 162.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 167.07 | 165.86 | 164.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 173.23 | 166.26 | 165.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 169.36 | 167.56 | 166.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:30:00 | 169.90 | 167.98 | 166.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 169.06 | 167.98 | 166.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 167.40 | 169.30 | 168.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 166.80 | 169.30 | 168.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 166.73 | 168.79 | 168.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 166.73 | 168.79 | 168.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 169.04 | 168.93 | 168.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 168.70 | 168.93 | 168.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 168.10 | 168.79 | 168.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 170.20 | 168.79 | 168.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 167.98 | 168.63 | 168.50 | SL hit (close<static) qty=1.00 sl=168.10 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 166.25 | 168.15 | 168.30 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 171.10 | 168.11 | 168.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 171.65 | 169.35 | 168.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 169.90 | 170.47 | 169.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 169.90 | 170.47 | 169.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 169.90 | 170.47 | 169.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 170.01 | 170.47 | 169.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 171.65 | 170.71 | 169.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 172.21 | 170.71 | 169.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 169.75 | 170.29 | 170.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 169.75 | 170.29 | 170.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 12:15:00 | 169.62 | 170.15 | 170.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 13:15:00 | 169.38 | 168.96 | 169.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 169.38 | 168.96 | 169.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 169.38 | 168.96 | 169.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 169.43 | 168.96 | 169.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 169.23 | 169.02 | 169.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 169.55 | 169.02 | 169.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 169.00 | 169.01 | 169.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 166.50 | 169.01 | 169.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 168.10 | 168.29 | 168.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 170.69 | 168.82 | 168.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 170.69 | 168.82 | 168.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 176.46 | 170.78 | 169.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 176.66 | 176.84 | 174.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:00:00 | 176.66 | 176.84 | 174.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 172.38 | 175.97 | 174.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 172.38 | 175.97 | 174.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 169.89 | 174.75 | 174.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 169.89 | 174.75 | 174.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 169.31 | 173.67 | 173.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 168.78 | 171.32 | 172.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 163.40 | 163.11 | 166.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 163.40 | 163.11 | 166.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 156.48 | 155.10 | 156.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 156.50 | 155.10 | 156.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 156.74 | 155.43 | 156.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 156.74 | 155.43 | 156.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 157.18 | 155.78 | 156.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 157.18 | 155.78 | 156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 157.50 | 156.12 | 156.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 157.94 | 156.12 | 156.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 154.96 | 155.89 | 156.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 154.39 | 155.89 | 156.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 154.25 | 155.24 | 156.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:45:00 | 153.80 | 154.33 | 155.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:30:00 | 154.43 | 154.45 | 154.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 154.06 | 154.26 | 154.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 155.01 | 154.26 | 154.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 154.24 | 154.26 | 154.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 154.24 | 154.26 | 154.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 155.04 | 154.06 | 154.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 155.04 | 154.06 | 154.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 155.20 | 154.29 | 154.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 158.09 | 154.29 | 154.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 157.66 | 154.96 | 154.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 157.66 | 154.96 | 154.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 160.86 | 157.25 | 156.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 159.80 | 160.17 | 158.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 159.80 | 160.17 | 158.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 182.52 | 185.10 | 182.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 182.52 | 185.10 | 182.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 186.50 | 185.38 | 182.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:15:00 | 187.05 | 185.62 | 182.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 187.00 | 186.09 | 183.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 192.97 | 185.30 | 184.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 186.90 | 191.00 | 191.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 186.90 | 191.00 | 191.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 185.55 | 187.68 | 189.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 183.20 | 181.15 | 183.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 183.20 | 181.15 | 183.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 183.20 | 181.15 | 183.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 183.95 | 181.15 | 183.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 182.75 | 181.47 | 183.43 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 188.76 | 184.37 | 184.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 191.80 | 186.48 | 185.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 183.21 | 187.27 | 186.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 183.21 | 187.27 | 186.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 183.21 | 187.27 | 186.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 182.70 | 187.27 | 186.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 181.90 | 186.19 | 185.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 181.90 | 186.19 | 185.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 183.06 | 185.57 | 185.57 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 188.49 | 185.29 | 185.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 189.06 | 186.04 | 185.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 189.94 | 190.12 | 188.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 189.94 | 190.12 | 188.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 190.35 | 190.60 | 189.44 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 184.51 | 188.33 | 188.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 180.35 | 186.73 | 187.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 180.73 | 180.65 | 183.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:45:00 | 180.63 | 180.65 | 183.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 182.45 | 180.98 | 182.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:45:00 | 180.54 | 181.05 | 182.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 180.70 | 180.87 | 182.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 179.92 | 180.39 | 181.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 180.85 | 180.08 | 180.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 180.00 | 180.06 | 180.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 180.60 | 180.06 | 180.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 177.33 | 178.67 | 179.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 176.94 | 178.32 | 179.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 176.91 | 178.32 | 179.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 176.52 | 177.89 | 179.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 176.74 | 176.49 | 177.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 12:15:00 | 171.66 | 174.02 | 175.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 12:15:00 | 171.81 | 174.02 | 175.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 14:15:00 | 171.51 | 173.37 | 175.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 173.21 | 173.12 | 174.64 | SL hit (close>ema200) qty=0.50 sl=173.12 alert=retest2 |

### Cycle 15 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 165.69 | 162.91 | 162.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 168.29 | 164.91 | 163.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 175.72 | 176.65 | 174.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 175.72 | 176.65 | 174.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 175.97 | 176.43 | 175.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 176.69 | 176.43 | 175.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 173.00 | 175.75 | 175.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 173.09 | 175.75 | 175.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 172.55 | 175.11 | 175.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:30:00 | 172.88 | 175.11 | 175.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 172.47 | 174.58 | 174.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 170.81 | 173.45 | 174.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 173.94 | 173.19 | 173.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 173.94 | 173.19 | 173.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 173.94 | 173.19 | 173.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 173.91 | 173.19 | 173.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 174.88 | 173.53 | 173.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 174.60 | 173.53 | 173.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 175.30 | 173.88 | 174.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 175.72 | 173.88 | 174.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 176.50 | 174.41 | 174.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 177.82 | 175.09 | 174.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 174.56 | 175.37 | 174.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 174.56 | 175.37 | 174.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 174.56 | 175.37 | 174.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 174.56 | 175.37 | 174.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 173.60 | 175.02 | 174.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 173.60 | 175.02 | 174.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 172.81 | 174.57 | 174.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 172.60 | 173.77 | 174.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 174.68 | 173.95 | 174.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 174.68 | 173.95 | 174.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 174.68 | 173.95 | 174.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 174.68 | 173.95 | 174.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 174.65 | 174.09 | 174.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 175.03 | 174.09 | 174.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 175.10 | 174.29 | 174.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 175.56 | 174.29 | 174.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 175.70 | 174.57 | 174.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 178.05 | 175.48 | 174.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 176.31 | 176.89 | 175.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 176.31 | 176.89 | 175.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 175.20 | 176.55 | 175.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 175.20 | 176.55 | 175.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 176.15 | 176.47 | 175.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 173.76 | 176.47 | 175.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 175.61 | 176.30 | 175.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 175.75 | 176.30 | 175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 174.75 | 175.99 | 175.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 173.98 | 175.99 | 175.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 171.35 | 175.06 | 175.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 169.70 | 171.20 | 171.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 170.55 | 169.76 | 170.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 170.55 | 169.76 | 170.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 170.55 | 169.76 | 170.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 170.55 | 169.76 | 170.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 171.20 | 170.05 | 170.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 168.54 | 170.05 | 170.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 165.01 | 169.04 | 170.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 164.55 | 169.04 | 170.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 164.45 | 166.69 | 167.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 163.02 | 165.97 | 166.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 164.40 | 164.38 | 165.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 156.32 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 156.23 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 154.87 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 156.18 | 162.63 | 164.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-10 13:15:00 | 148.10 | 152.39 | 157.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 21 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 162.21 | 152.39 | 151.65 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 150.47 | 152.17 | 152.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 149.52 | 151.64 | 152.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 150.10 | 149.69 | 150.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 150.10 | 149.69 | 150.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 150.10 | 149.69 | 150.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 149.86 | 149.69 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 148.49 | 149.45 | 150.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:00:00 | 147.50 | 148.41 | 149.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 147.50 | 148.23 | 148.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 140.12 | 142.32 | 144.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 140.12 | 142.32 | 144.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 139.53 | 139.19 | 140.97 | SL hit (close>ema200) qty=0.50 sl=139.19 alert=retest2 |

### Cycle 23 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 144.13 | 141.82 | 141.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 144.64 | 142.39 | 141.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 145.20 | 145.78 | 144.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 145.21 | 145.78 | 144.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 144.93 | 145.61 | 144.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 144.74 | 145.61 | 144.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 144.94 | 145.39 | 144.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 144.94 | 145.39 | 144.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 144.38 | 145.19 | 144.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 144.38 | 145.19 | 144.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 145.29 | 145.21 | 144.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 145.44 | 145.21 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 151.86 | 152.11 | 150.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:30:00 | 152.49 | 152.29 | 150.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 150.61 | 152.10 | 151.02 | SL hit (close<static) qty=1.00 sl=150.62 alert=retest2 |

### Cycle 24 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 149.20 | 150.60 | 150.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 146.11 | 149.55 | 150.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 139.45 | 138.83 | 140.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 139.45 | 138.83 | 140.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 140.05 | 139.31 | 140.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 144.87 | 139.31 | 140.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 142.48 | 139.95 | 140.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 143.97 | 139.95 | 140.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 142.94 | 140.54 | 140.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 143.58 | 140.54 | 140.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 143.18 | 141.07 | 141.04 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 140.61 | 141.04 | 141.06 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 141.48 | 141.13 | 141.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 142.32 | 141.37 | 141.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 145.19 | 145.25 | 143.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:30:00 | 144.75 | 145.25 | 143.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 141.66 | 144.91 | 144.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 141.66 | 144.91 | 144.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 142.16 | 144.36 | 144.27 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 142.63 | 144.01 | 144.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 141.42 | 142.59 | 143.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 142.86 | 141.78 | 142.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 142.86 | 141.78 | 142.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 142.86 | 141.78 | 142.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:30:00 | 141.85 | 141.78 | 142.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 143.22 | 142.07 | 142.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 143.22 | 142.07 | 142.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 143.22 | 142.30 | 142.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 144.36 | 142.30 | 142.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 142.98 | 142.44 | 142.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 142.73 | 142.44 | 142.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 142.81 | 142.51 | 142.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 141.21 | 142.51 | 142.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 142.54 | 141.94 | 142.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 142.54 | 141.94 | 142.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 142.80 | 142.11 | 142.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 143.00 | 142.11 | 142.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 142.91 | 142.45 | 142.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 146.37 | 143.67 | 143.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 144.74 | 145.19 | 144.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 144.74 | 145.19 | 144.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 144.80 | 145.11 | 144.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 145.38 | 145.08 | 144.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 145.60 | 145.06 | 144.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 146.15 | 145.28 | 144.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 144.39 | 145.07 | 144.83 | SL hit (close<static) qty=1.00 sl=144.41 alert=retest2 |

### Cycle 30 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 144.05 | 144.69 | 144.70 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 146.22 | 144.79 | 144.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 149.10 | 146.17 | 145.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 148.10 | 149.32 | 148.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 148.10 | 149.32 | 148.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 148.10 | 149.32 | 148.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 148.50 | 149.32 | 148.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 148.60 | 149.17 | 148.11 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 146.46 | 147.69 | 147.74 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 148.10 | 147.77 | 147.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 149.53 | 148.12 | 147.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 147.50 | 148.30 | 148.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 147.50 | 148.30 | 148.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 147.50 | 148.30 | 148.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 147.33 | 148.30 | 148.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 146.73 | 147.98 | 147.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 146.37 | 147.61 | 147.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 147.96 | 147.68 | 147.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 147.96 | 147.68 | 147.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 147.96 | 147.68 | 147.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 147.96 | 147.68 | 147.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 147.90 | 147.72 | 147.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 147.25 | 147.72 | 147.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 146.49 | 147.48 | 147.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 146.15 | 147.48 | 147.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 146.15 | 144.91 | 145.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 144.85 | 145.25 | 145.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 138.84 | 142.86 | 143.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 138.84 | 142.86 | 143.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 137.61 | 139.65 | 141.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 137.34 | 137.22 | 139.27 | SL hit (close>ema200) qty=0.50 sl=137.22 alert=retest2 |

### Cycle 35 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 141.94 | 138.33 | 137.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 136.19 | 137.98 | 138.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 134.80 | 137.34 | 137.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 131.13 | 128.92 | 130.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 131.13 | 128.92 | 130.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 131.13 | 128.92 | 130.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 131.75 | 128.92 | 130.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 130.00 | 129.14 | 130.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 129.64 | 129.33 | 130.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:45:00 | 129.60 | 130.32 | 130.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 129.39 | 129.96 | 130.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 129.63 | 128.61 | 128.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 129.63 | 128.61 | 128.55 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 127.34 | 128.30 | 128.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 127.00 | 128.04 | 128.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 14:15:00 | 128.17 | 127.59 | 127.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 128.17 | 127.59 | 127.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 128.17 | 127.59 | 127.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 128.17 | 127.59 | 127.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 128.36 | 127.75 | 128.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 125.94 | 127.75 | 128.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 129.56 | 128.11 | 128.16 | SL hit (close>static) qty=1.00 sl=128.70 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 129.25 | 128.34 | 128.26 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 127.22 | 128.20 | 128.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 125.83 | 127.73 | 128.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 122.31 | 120.77 | 123.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 122.31 | 120.77 | 123.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 122.84 | 121.18 | 123.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 126.08 | 121.18 | 123.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 125.38 | 122.02 | 123.37 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 128.05 | 124.83 | 124.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 131.28 | 127.12 | 125.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 128.06 | 129.58 | 127.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 128.06 | 129.58 | 127.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 128.06 | 129.58 | 127.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 127.89 | 129.58 | 127.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 126.57 | 128.98 | 127.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 126.57 | 128.98 | 127.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 126.26 | 128.43 | 127.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 126.40 | 128.43 | 127.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 125.15 | 127.17 | 127.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 117.75 | 124.66 | 126.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 122.44 | 122.13 | 124.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 122.44 | 122.13 | 124.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 127.40 | 123.26 | 124.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 125.92 | 123.26 | 124.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 129.32 | 124.47 | 124.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 129.32 | 124.47 | 124.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 129.14 | 125.41 | 125.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 130.25 | 127.41 | 126.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 14:15:00 | 138.22 | 138.24 | 135.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:30:00 | 138.00 | 138.24 | 135.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 138.92 | 138.37 | 136.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 140.02 | 138.44 | 136.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 140.74 | 138.44 | 136.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 135.50 | 137.30 | 137.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 135.50 | 137.30 | 137.38 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 138.15 | 136.79 | 136.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 141.56 | 137.74 | 137.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 140.47 | 141.23 | 139.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 140.47 | 141.23 | 139.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 138.80 | 140.58 | 139.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 138.80 | 140.58 | 139.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 139.27 | 140.32 | 139.63 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 136.38 | 138.74 | 139.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 134.89 | 137.61 | 138.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 133.58 | 133.31 | 134.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 133.58 | 133.31 | 134.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 133.58 | 133.31 | 134.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 133.77 | 133.31 | 134.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 133.95 | 133.44 | 134.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 132.74 | 133.35 | 134.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:00:00 | 132.50 | 133.18 | 134.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 132.49 | 133.62 | 133.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 131.70 | 133.43 | 133.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 130.64 | 132.87 | 133.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 129.88 | 132.87 | 133.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 126.10 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 125.88 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 125.87 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 125.11 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 123.39 | 129.08 | 130.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 126.59 | 126.32 | 128.46 | SL hit (close>ema200) qty=0.50 sl=126.32 alert=retest2 |

### Cycle 47 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 126.43 | 120.22 | 119.81 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 119.80 | 121.02 | 121.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 118.99 | 120.62 | 120.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 118.34 | 117.93 | 119.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 118.34 | 117.93 | 119.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 118.34 | 117.93 | 119.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 118.81 | 117.93 | 119.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 118.80 | 118.10 | 119.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 118.05 | 118.10 | 119.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 124.06 | 120.14 | 119.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 124.06 | 120.14 | 119.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 126.29 | 121.37 | 120.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 124.48 | 125.21 | 123.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 124.48 | 125.21 | 123.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 124.48 | 125.21 | 123.08 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 117.09 | 122.02 | 122.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 116.51 | 119.73 | 121.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 119.17 | 118.92 | 120.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 119.17 | 118.92 | 120.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 119.17 | 118.92 | 120.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 118.77 | 118.96 | 120.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 118.85 | 118.96 | 120.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:00:00 | 118.65 | 118.90 | 120.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 125.03 | 120.95 | 120.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 125.03 | 120.95 | 120.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 125.64 | 121.89 | 121.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 120.72 | 123.19 | 122.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 120.72 | 123.19 | 122.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 120.72 | 123.19 | 122.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 120.72 | 123.19 | 122.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 120.01 | 122.56 | 122.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 120.10 | 122.56 | 122.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 119.75 | 121.57 | 121.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 117.05 | 120.14 | 120.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 119.56 | 116.83 | 118.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 119.56 | 116.83 | 118.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 119.56 | 116.83 | 118.46 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 120.03 | 119.22 | 119.18 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 115.97 | 118.69 | 118.96 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 120.00 | 119.02 | 118.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 120.80 | 119.63 | 119.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 120.70 | 120.97 | 120.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 120.70 | 120.97 | 120.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 120.85 | 120.99 | 120.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 120.57 | 120.99 | 120.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 120.67 | 120.93 | 120.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 124.41 | 120.93 | 120.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 136.85 | 134.64 | 133.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 132.80 | 134.96 | 135.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 132.68 | 133.66 | 134.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 134.70 | 133.79 | 134.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 134.70 | 133.79 | 134.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 134.70 | 133.79 | 134.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:45:00 | 135.07 | 133.79 | 134.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 133.66 | 133.76 | 134.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 133.25 | 133.79 | 134.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 133.12 | 133.66 | 134.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 135.70 | 133.94 | 134.10 | SL hit (close>static) qty=1.00 sl=134.77 alert=retest2 |

### Cycle 57 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 136.76 | 134.50 | 134.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 137.38 | 135.08 | 134.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 13:15:00 | 137.50 | 137.82 | 136.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 137.50 | 137.82 | 136.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 138.78 | 139.78 | 138.89 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 137.73 | 138.54 | 138.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 136.85 | 138.20 | 138.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 136.65 | 136.58 | 137.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:00:00 | 136.65 | 136.58 | 137.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 137.62 | 136.78 | 137.44 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 141.50 | 138.22 | 137.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 143.40 | 142.41 | 141.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 142.86 | 142.88 | 142.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 142.86 | 142.88 | 142.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 143.78 | 143.07 | 142.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:00:00 | 145.20 | 143.49 | 142.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:30:00 | 118.30 | 2025-05-29 09:15:00 | 130.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 142.42 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2025-06-18 14:15:00 | 141.83 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.79% |
| SELL | retest2 | 2025-06-18 15:15:00 | 142.15 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.55% |
| SELL | retest2 | 2025-06-20 12:15:00 | 142.05 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -7.62% |
| SELL | retest2 | 2025-06-23 13:00:00 | 139.74 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -9.40% |
| SELL | retest2 | 2025-06-23 14:15:00 | 139.61 | 2025-06-24 09:15:00 | 152.88 | STOP_HIT | 1.00 | -9.51% |
| BUY | retest2 | 2025-07-02 09:15:00 | 159.50 | 2025-07-11 09:15:00 | 167.98 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2025-07-02 12:15:00 | 159.71 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | 4.09% |
| BUY | retest2 | 2025-07-02 13:15:00 | 159.40 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-07-02 14:30:00 | 161.58 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-07-08 09:15:00 | 173.23 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2025-07-08 11:30:00 | 169.36 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-08 13:30:00 | 169.90 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-07-08 14:15:00 | 169.06 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-11 09:15:00 | 170.20 | 2025-07-11 10:15:00 | 166.25 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-07-15 11:15:00 | 172.21 | 2025-07-17 11:15:00 | 169.75 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-21 09:15:00 | 166.50 | 2025-07-22 13:15:00 | 170.69 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-07-22 09:45:00 | 168.10 | 2025-07-22 13:15:00 | 170.69 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-05 10:15:00 | 154.39 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-05 11:30:00 | 154.25 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-08-06 09:45:00 | 153.80 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-08-06 14:30:00 | 154.43 | 2025-08-08 09:15:00 | 157.66 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-19 12:15:00 | 187.05 | 2025-08-26 10:15:00 | 186.90 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-08-19 14:15:00 | 187.00 | 2025-08-26 10:15:00 | 186.90 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-08-21 10:15:00 | 192.97 | 2025-08-26 10:15:00 | 186.90 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-09-12 12:45:00 | 180.54 | 2025-09-19 12:15:00 | 171.66 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-09-12 14:45:00 | 180.70 | 2025-09-19 12:15:00 | 171.81 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-09-15 09:45:00 | 179.92 | 2025-09-19 14:15:00 | 171.51 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2025-09-12 12:45:00 | 180.54 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-09-12 14:45:00 | 180.70 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-09-16 09:30:00 | 180.85 | 2025-09-22 09:15:00 | 170.92 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2025-09-15 09:45:00 | 179.92 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-09-16 09:30:00 | 180.85 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-09-17 10:30:00 | 176.94 | 2025-09-22 09:15:00 | 168.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 176.94 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-09-17 11:00:00 | 176.91 | 2025-09-22 09:15:00 | 168.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:00:00 | 176.91 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2025-09-17 11:45:00 | 176.52 | 2025-09-22 09:15:00 | 167.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:45:00 | 176.52 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-09-18 11:00:00 | 176.74 | 2025-09-22 09:15:00 | 167.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 176.74 | 2025-09-22 09:15:00 | 173.21 | STOP_HIT | 0.50 | 2.00% |
| SELL | retest2 | 2025-09-23 11:00:00 | 171.66 | 2025-09-26 14:15:00 | 163.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 171.26 | 2025-09-26 14:15:00 | 162.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:15:00 | 171.87 | 2025-09-26 14:15:00 | 163.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:00:00 | 171.66 | 2025-10-01 11:15:00 | 160.63 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2025-09-24 09:15:00 | 171.26 | 2025-10-01 11:15:00 | 160.63 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2025-09-24 14:15:00 | 171.87 | 2025-10-01 11:15:00 | 160.63 | STOP_HIT | 0.50 | 6.54% |
| SELL | retest2 | 2025-10-29 10:15:00 | 164.55 | 2025-11-07 09:15:00 | 156.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 15:00:00 | 164.45 | 2025-11-07 09:15:00 | 156.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:15:00 | 163.02 | 2025-11-07 09:15:00 | 154.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 14:30:00 | 164.40 | 2025-11-07 09:15:00 | 156.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 10:15:00 | 164.55 | 2025-11-10 13:15:00 | 148.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 15:00:00 | 164.45 | 2025-11-10 13:15:00 | 148.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 09:15:00 | 163.02 | 2025-11-10 13:15:00 | 146.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 14:30:00 | 164.40 | 2025-11-10 13:15:00 | 147.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 14:00:00 | 147.50 | 2025-11-21 14:15:00 | 140.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 147.50 | 2025-11-21 14:15:00 | 140.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 14:00:00 | 147.50 | 2025-11-25 11:15:00 | 139.53 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-11-19 15:15:00 | 147.50 | 2025-11-25 11:15:00 | 139.53 | STOP_HIT | 0.50 | 5.40% |
| BUY | retest2 | 2025-12-02 14:30:00 | 152.49 | 2025-12-03 09:15:00 | 150.61 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 09:45:00 | 145.38 | 2025-12-26 14:15:00 | 144.39 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-26 12:15:00 | 145.60 | 2025-12-26 14:15:00 | 144.39 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-26 13:00:00 | 146.15 | 2025-12-26 14:15:00 | 144.39 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-06 10:15:00 | 146.15 | 2026-01-09 09:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:45:00 | 146.15 | 2026-01-09 09:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 144.85 | 2026-01-12 09:15:00 | 137.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:15:00 | 146.15 | 2026-01-12 15:15:00 | 137.34 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-01-07 14:45:00 | 146.15 | 2026-01-12 15:15:00 | 137.34 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-01-08 09:15:00 | 144.85 | 2026-01-12 15:15:00 | 137.34 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-22 11:30:00 | 129.64 | 2026-01-28 15:15:00 | 129.63 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-01-23 09:45:00 | 129.60 | 2026-01-28 15:15:00 | 129.63 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-01-23 10:30:00 | 129.39 | 2026-01-28 15:15:00 | 129.63 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-01-30 09:15:00 | 125.94 | 2026-01-30 09:15:00 | 129.56 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-02-12 12:30:00 | 140.02 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-02-12 13:15:00 | 140.74 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-02-25 12:45:00 | 132.74 | 2026-03-02 09:15:00 | 126.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 132.50 | 2026-03-02 09:15:00 | 125.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 14:15:00 | 132.49 | 2026-03-02 09:15:00 | 125.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 131.70 | 2026-03-02 09:15:00 | 125.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:15:00 | 129.88 | 2026-03-02 09:15:00 | 123.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 132.74 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2026-02-25 14:00:00 | 132.50 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2026-02-26 14:15:00 | 132.49 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-27 09:15:00 | 131.70 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-02-27 10:15:00 | 129.88 | 2026-03-02 15:15:00 | 126.59 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2026-03-17 09:15:00 | 118.05 | 2026-03-18 09:15:00 | 124.06 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2026-03-24 10:30:00 | 118.77 | 2026-03-25 09:15:00 | 125.03 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest2 | 2026-03-24 11:15:00 | 118.85 | 2026-03-25 09:15:00 | 125.03 | STOP_HIT | 1.00 | -5.20% |
| SELL | retest2 | 2026-03-24 12:00:00 | 118.65 | 2026-03-25 09:15:00 | 125.03 | STOP_HIT | 1.00 | -5.38% |
| BUY | retest2 | 2026-04-08 09:15:00 | 124.41 | 2026-04-16 09:15:00 | 136.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 133.25 | 2026-04-23 11:15:00 | 135.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-23 10:00:00 | 133.12 | 2026-04-23 11:15:00 | 135.70 | STOP_HIT | 1.00 | -1.94% |
