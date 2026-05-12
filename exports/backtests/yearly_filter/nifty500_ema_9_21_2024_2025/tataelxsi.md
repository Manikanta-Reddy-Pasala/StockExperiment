# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 4313.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 134 |
| ALERT1 | 89 |
| ALERT2 | 87 |
| ALERT2_SKIP | 40 |
| ALERT3 | 244 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 133 |
| PARTIAL | 36 |
| TARGET_HIT | 21 |
| STOP_HIT | 112 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 169 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 90 / 79
- **Target hits / Stop hits / Partials:** 21 / 112 / 36
- **Avg / median % per leg:** 2.35% / 0.62%
- **Sum % (uncompounded):** 396.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 16 | 29.6% | 8 | 46 | 0 | 1.12% | 60.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 54 | 16 | 29.6% | 8 | 46 | 0 | 1.12% | 60.3% |
| SELL (all) | 115 | 74 | 64.3% | 13 | 66 | 36 | 2.92% | 336.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 115 | 74 | 64.3% | 13 | 66 | 36 | 2.92% | 336.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 169 | 90 | 53.3% | 21 | 112 | 36 | 2.35% | 396.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 7035.00 | 7094.91 | 7098.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 10:15:00 | 7021.00 | 7080.12 | 7091.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 14:15:00 | 7058.90 | 7058.77 | 7076.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 14:45:00 | 7067.15 | 7058.77 | 7076.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 7100.00 | 7067.02 | 7078.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 7122.00 | 7067.02 | 7078.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 7079.85 | 7069.59 | 7078.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 11:30:00 | 7069.80 | 7070.15 | 7077.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 09:15:00 | 7088.00 | 7081.70 | 7080.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 7088.00 | 7081.70 | 7080.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 7141.00 | 7099.38 | 7089.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 09:15:00 | 7321.55 | 7323.75 | 7272.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 09:45:00 | 7321.55 | 7323.75 | 7272.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 7277.95 | 7312.61 | 7280.27 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 7225.00 | 7259.42 | 7263.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 7199.45 | 7247.42 | 7258.03 | Break + close below crossover candle low |

### Cycle 4 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 7425.75 | 7261.88 | 7254.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 7319.45 | 7341.73 | 7342.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 7275.00 | 7316.77 | 7327.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 14:15:00 | 7291.20 | 7289.12 | 7307.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 15:00:00 | 7291.20 | 7289.12 | 7307.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 7198.10 | 7271.05 | 7296.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:45:00 | 7176.10 | 7241.59 | 7277.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 12:15:00 | 7180.00 | 7241.59 | 7277.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:15:00 | 7182.50 | 7231.27 | 7269.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:30:00 | 7171.40 | 7214.86 | 7255.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 7162.45 | 7203.68 | 7242.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:30:00 | 7140.30 | 7191.77 | 7233.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 6817.30 | 6964.29 | 7045.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 6821.00 | 6964.29 | 7045.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 6823.38 | 6964.29 | 7045.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 6812.83 | 6964.29 | 7045.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 6783.28 | 6927.50 | 7021.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 6458.49 | 6849.86 | 6967.48 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 7140.60 | 6958.73 | 6934.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 7242.75 | 7144.66 | 7117.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 13:15:00 | 7239.95 | 7241.50 | 7199.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 14:00:00 | 7239.95 | 7241.50 | 7199.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 7221.15 | 7273.30 | 7249.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 7232.65 | 7273.30 | 7249.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 7221.95 | 7263.03 | 7247.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 7223.75 | 7263.03 | 7247.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 7246.50 | 7259.24 | 7248.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 7246.50 | 7259.24 | 7248.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 7248.80 | 7257.15 | 7248.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:30:00 | 7241.60 | 7257.15 | 7248.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 7246.70 | 7255.06 | 7248.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 14:30:00 | 7245.35 | 7255.06 | 7248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 7248.00 | 7253.65 | 7248.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 7235.00 | 7253.65 | 7248.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 7207.80 | 7244.48 | 7244.49 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 7304.35 | 7244.49 | 7242.12 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 7084.35 | 7215.66 | 7230.90 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 7322.80 | 7238.78 | 7234.48 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 09:15:00 | 7126.15 | 7225.18 | 7231.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 7113.00 | 7173.53 | 7203.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 7027.90 | 7010.49 | 7037.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 7027.90 | 7010.49 | 7037.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 7027.90 | 7010.49 | 7037.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:15:00 | 7059.80 | 7010.49 | 7037.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 7055.60 | 7019.51 | 7038.93 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 7075.00 | 7050.24 | 7048.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 7095.80 | 7059.35 | 7052.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 7044.00 | 7064.50 | 7057.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 7044.00 | 7064.50 | 7057.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 7044.00 | 7064.50 | 7057.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 7044.00 | 7064.50 | 7057.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 7066.00 | 7064.80 | 7058.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 7067.00 | 7064.80 | 7058.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 7076.15 | 7063.25 | 7058.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 13:15:00 | 7071.15 | 7072.84 | 7065.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 14:30:00 | 7080.00 | 7075.17 | 7067.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 7095.25 | 7082.82 | 7072.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 7107.40 | 7082.82 | 7072.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 7072.00 | 7084.02 | 7077.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 7072.00 | 7084.02 | 7077.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 7071.00 | 7081.42 | 7077.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 7095.00 | 7081.42 | 7077.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 7044.00 | 7071.95 | 7073.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 7044.00 | 7071.95 | 7073.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 11:15:00 | 7034.00 | 7064.36 | 7069.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 7058.00 | 7054.73 | 7063.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 14:15:00 | 7058.00 | 7054.73 | 7063.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 7058.00 | 7054.73 | 7063.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 7058.00 | 7054.73 | 7063.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 7024.90 | 7048.49 | 7058.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 6988.40 | 7032.22 | 7044.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:00:00 | 7011.95 | 7023.01 | 7038.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 12:00:00 | 7015.00 | 7021.41 | 7035.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:30:00 | 7011.90 | 7017.99 | 7030.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 7080.00 | 7030.55 | 7034.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 7090.00 | 7042.44 | 7039.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 10:15:00 | 7090.00 | 7042.44 | 7039.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 7118.65 | 7075.04 | 7057.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 6981.00 | 7061.02 | 7054.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 6981.00 | 7061.02 | 7054.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 6981.00 | 7061.02 | 7054.33 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 10:15:00 | 6981.80 | 7045.17 | 7047.73 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 7058.05 | 7027.03 | 7026.77 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 12:15:00 | 7017.00 | 7029.18 | 7030.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 09:15:00 | 6995.90 | 7018.28 | 7024.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 15:15:00 | 7009.00 | 7002.11 | 7011.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 09:15:00 | 6972.70 | 7002.11 | 7011.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 6980.65 | 6997.82 | 7008.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 6938.40 | 6972.42 | 6984.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 6935.00 | 6966.93 | 6980.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 6940.00 | 6951.87 | 6963.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:45:00 | 6924.00 | 6948.04 | 6961.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 6965.00 | 6952.07 | 6960.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:00:00 | 6965.00 | 6952.07 | 6960.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 6961.40 | 6953.93 | 6960.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 7018.60 | 6966.87 | 6965.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 7018.60 | 6966.87 | 6965.98 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 11:15:00 | 6940.00 | 6962.12 | 6964.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 6875.00 | 6932.10 | 6947.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 6932.55 | 6893.80 | 6914.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 6932.55 | 6893.80 | 6914.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 6932.55 | 6893.80 | 6914.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 6932.55 | 6893.80 | 6914.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 6930.00 | 6901.04 | 6916.10 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 6952.40 | 6926.61 | 6924.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 6974.00 | 6936.09 | 6929.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 6923.00 | 6941.16 | 6935.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 13:15:00 | 6923.00 | 6941.16 | 6935.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 6923.00 | 6941.16 | 6935.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:00:00 | 6923.00 | 6941.16 | 6935.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 6932.35 | 6939.40 | 6935.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 15:15:00 | 6935.00 | 6939.40 | 6935.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 6935.00 | 6938.52 | 6935.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 6940.10 | 6938.52 | 6935.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 6939.60 | 6935.21 | 6933.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:30:00 | 6935.30 | 6934.19 | 6933.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:45:00 | 6936.00 | 6934.95 | 6933.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 6930.00 | 6933.96 | 6933.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 6930.00 | 6933.96 | 6933.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 6928.00 | 6932.77 | 6933.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 6928.00 | 6932.77 | 6933.12 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 6971.80 | 6939.60 | 6936.10 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 6924.75 | 6946.84 | 6949.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 6827.00 | 6910.19 | 6929.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 6825.00 | 6813.02 | 6857.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 6825.00 | 6813.02 | 6857.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 6825.00 | 6813.02 | 6857.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 6751.65 | 6782.00 | 6826.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 6767.00 | 6761.10 | 6804.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 6822.65 | 6815.45 | 6815.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 6822.65 | 6815.45 | 6815.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 14:15:00 | 6863.00 | 6838.31 | 6828.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 6816.45 | 6841.73 | 6831.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 6816.45 | 6841.73 | 6831.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 6816.45 | 6841.73 | 6831.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 6884.00 | 6844.35 | 6834.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 6778.50 | 6821.71 | 6827.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 6778.50 | 6821.71 | 6827.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 6745.00 | 6798.10 | 6815.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 15:15:00 | 6798.00 | 6797.09 | 6811.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 15:15:00 | 6798.00 | 6797.09 | 6811.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 6798.00 | 6797.09 | 6811.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 6764.75 | 6797.09 | 6811.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 14:15:00 | 6828.10 | 6803.83 | 6807.27 | SL hit (close>static) qty=1.00 sl=6820.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 6834.00 | 6809.86 | 6809.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 6844.10 | 6821.96 | 6816.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 14:15:00 | 6820.00 | 6821.57 | 6816.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 6820.00 | 6821.57 | 6816.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 6820.00 | 6821.57 | 6816.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 6820.00 | 6821.57 | 6816.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 15:15:00 | 6756.00 | 6808.45 | 6810.92 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 6852.00 | 6817.16 | 6814.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 6936.95 | 6845.63 | 6829.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 6934.80 | 6934.95 | 6904.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:00:00 | 6934.80 | 6934.95 | 6904.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 6929.00 | 6929.93 | 6907.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 6902.25 | 6929.93 | 6907.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 6934.70 | 6944.46 | 6926.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 6920.60 | 6944.46 | 6926.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 6966.30 | 6948.83 | 6930.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 6993.70 | 6948.96 | 6931.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:45:00 | 6981.70 | 6954.94 | 6938.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 14:15:00 | 7693.07 | 7323.66 | 7162.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 7940.00 | 8028.80 | 8037.44 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 8230.15 | 8069.07 | 8054.96 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 7964.95 | 8058.78 | 8062.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 7865.00 | 8020.02 | 8044.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 7917.10 | 7865.05 | 7933.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 7917.10 | 7865.05 | 7933.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 7917.10 | 7865.05 | 7933.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 7917.10 | 7865.05 | 7933.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 7815.00 | 7782.54 | 7828.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:15:00 | 7764.50 | 7786.61 | 7816.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:45:00 | 7765.85 | 7754.62 | 7793.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 15:15:00 | 7823.95 | 7704.78 | 7704.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 7823.95 | 7704.78 | 7704.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 7969.25 | 7757.67 | 7728.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 14:15:00 | 7851.25 | 7859.93 | 7800.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 15:00:00 | 7851.25 | 7859.93 | 7800.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 7785.40 | 7842.32 | 7802.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:30:00 | 7801.30 | 7842.32 | 7802.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 7802.60 | 7834.37 | 7802.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:30:00 | 7778.35 | 7834.37 | 7802.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 7786.95 | 7824.89 | 7800.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:00:00 | 7786.95 | 7824.89 | 7800.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 7783.25 | 7816.56 | 7799.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:30:00 | 7783.50 | 7816.56 | 7799.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 7715.00 | 7785.77 | 7787.66 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 7790.00 | 7784.70 | 7784.54 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 14:15:00 | 7775.95 | 7782.95 | 7783.76 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 7800.05 | 7787.18 | 7785.56 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 7701.10 | 7770.61 | 7778.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 12:15:00 | 7681.30 | 7728.05 | 7752.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 09:15:00 | 7718.85 | 7712.33 | 7736.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 7718.85 | 7712.33 | 7736.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 7718.85 | 7712.33 | 7736.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 7718.85 | 7712.33 | 7736.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 7687.00 | 7700.38 | 7726.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:30:00 | 7724.15 | 7700.38 | 7726.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 7695.35 | 7699.37 | 7723.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 09:30:00 | 7675.50 | 7701.03 | 7716.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:00:00 | 7657.00 | 7689.34 | 7707.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:45:00 | 7641.00 | 7646.14 | 7679.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:45:00 | 7674.05 | 7596.81 | 7623.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 7790.00 | 7635.45 | 7638.63 | SL hit (close>static) qty=1.00 sl=7738.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 7715.45 | 7651.45 | 7645.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 7835.55 | 7702.44 | 7670.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 7795.70 | 7815.33 | 7768.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 7795.70 | 7815.33 | 7768.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 7830.00 | 7867.16 | 7830.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 7835.00 | 7867.16 | 7830.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 7830.70 | 7859.87 | 7830.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 7834.00 | 7859.87 | 7830.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 7817.35 | 7851.37 | 7829.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 7819.00 | 7851.37 | 7829.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 7825.00 | 7846.09 | 7828.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:30:00 | 7881.50 | 7846.23 | 7832.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:30:00 | 7846.85 | 7846.21 | 7835.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 7863.65 | 7846.21 | 7835.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 8040.00 | 7833.86 | 7832.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 8026.70 | 7872.43 | 7850.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-27 15:15:00 | 7782.00 | 7878.89 | 7869.88 | SL hit (close<static) qty=1.00 sl=7810.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 7700.00 | 7843.11 | 7854.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 7683.75 | 7753.92 | 7779.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 7471.00 | 7458.39 | 7528.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 7475.00 | 7458.39 | 7528.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 7545.00 | 7475.71 | 7529.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 7545.00 | 7475.71 | 7529.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 7584.95 | 7497.56 | 7534.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 7578.05 | 7497.56 | 7534.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 7542.35 | 7514.44 | 7536.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:30:00 | 7548.85 | 7514.44 | 7536.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 7546.75 | 7520.91 | 7537.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:45:00 | 7548.65 | 7520.91 | 7537.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 7586.00 | 7533.92 | 7541.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 7601.60 | 7533.92 | 7541.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 7605.40 | 7548.22 | 7547.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 7713.95 | 7602.50 | 7574.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 7603.00 | 7613.28 | 7584.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 7603.00 | 7613.28 | 7584.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 7620.05 | 7614.64 | 7588.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 7766.85 | 7614.64 | 7588.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 7600.00 | 7675.65 | 7685.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 7600.00 | 7675.65 | 7685.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 7574.20 | 7644.07 | 7668.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 09:15:00 | 7464.95 | 7437.19 | 7488.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 7464.95 | 7437.19 | 7488.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 7464.95 | 7437.19 | 7488.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:30:00 | 7485.00 | 7437.19 | 7488.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 7460.00 | 7441.75 | 7485.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:45:00 | 7480.50 | 7441.75 | 7485.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 7441.85 | 7361.80 | 7392.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 7441.85 | 7361.80 | 7392.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 7365.70 | 7362.58 | 7389.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 7352.05 | 7362.58 | 7389.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 7354.05 | 7374.61 | 7384.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 12:15:00 | 6984.45 | 7124.38 | 7194.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 12:15:00 | 6986.35 | 7124.38 | 7194.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 6978.85 | 6978.79 | 7056.64 | SL hit (close>ema200) qty=0.50 sl=6978.79 alert=retest2 |

### Cycle 42 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 7089.30 | 7056.31 | 7052.04 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 14:15:00 | 7030.00 | 7052.90 | 7055.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 15:15:00 | 7015.70 | 7045.46 | 7052.32 | Break + close below crossover candle low |

### Cycle 44 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 7121.80 | 7060.73 | 7058.64 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 6995.90 | 7055.31 | 7057.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 6983.00 | 7040.85 | 7050.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 7036.10 | 7029.78 | 7038.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 7036.10 | 7029.78 | 7038.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 7036.10 | 7029.78 | 7038.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:45:00 | 7000.05 | 7023.74 | 7034.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 7168.30 | 7056.78 | 7045.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 7168.30 | 7056.78 | 7045.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 7217.15 | 7138.46 | 7094.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 7164.05 | 7193.76 | 7152.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 7164.05 | 7193.76 | 7152.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 7145.00 | 7184.01 | 7152.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 7154.95 | 7184.01 | 7152.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 7203.95 | 7187.99 | 7156.78 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 6999.85 | 7126.18 | 7138.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 6900.00 | 7080.95 | 7116.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 14:15:00 | 6631.50 | 6628.54 | 6760.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 15:00:00 | 6631.50 | 6628.54 | 6760.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 6475.95 | 6430.77 | 6485.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 6466.00 | 6430.77 | 6485.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 6534.60 | 6451.54 | 6490.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:45:00 | 6544.35 | 6451.54 | 6490.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 6449.65 | 6451.16 | 6486.60 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 6567.30 | 6504.43 | 6502.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 6575.00 | 6518.55 | 6508.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 6476.55 | 6523.49 | 6516.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 6476.55 | 6523.49 | 6516.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 6476.55 | 6523.49 | 6516.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 6476.55 | 6523.49 | 6516.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 6483.60 | 6515.51 | 6513.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:00:00 | 6501.25 | 6512.66 | 6512.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 6491.35 | 6510.43 | 6511.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 6491.35 | 6510.43 | 6511.44 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 6558.50 | 6516.78 | 6513.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 10:15:00 | 6584.55 | 6530.33 | 6520.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 6701.25 | 6739.73 | 6684.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 6701.25 | 6739.73 | 6684.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 6800.25 | 6743.88 | 6695.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 6711.10 | 6743.88 | 6695.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 6728.40 | 6755.06 | 6724.65 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 12:15:00 | 6681.65 | 6723.54 | 6728.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 14:15:00 | 6670.00 | 6706.76 | 6719.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 6698.75 | 6697.36 | 6712.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 6698.75 | 6697.36 | 6712.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 6698.75 | 6697.36 | 6712.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 6692.75 | 6697.36 | 6712.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 6735.00 | 6704.89 | 6714.81 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 7101.00 | 6789.76 | 6750.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 11:15:00 | 7187.60 | 7084.15 | 6966.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 7346.60 | 7370.84 | 7291.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 10:00:00 | 7346.60 | 7370.84 | 7291.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 7348.05 | 7361.00 | 7335.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 7343.00 | 7361.00 | 7335.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 7351.45 | 7369.87 | 7347.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:45:00 | 7344.45 | 7369.87 | 7347.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 7353.10 | 7366.51 | 7347.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 7400.40 | 7357.90 | 7349.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 7335.15 | 7357.21 | 7351.91 | SL hit (close<static) qty=1.00 sl=7342.60 alert=retest2 |

### Cycle 53 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 7319.55 | 7344.12 | 7346.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 7266.10 | 7324.02 | 7336.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 7326.00 | 7320.16 | 7332.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 7326.00 | 7320.16 | 7332.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 7326.00 | 7320.16 | 7332.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 7326.00 | 7320.16 | 7332.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 7353.75 | 7326.88 | 7334.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 7353.75 | 7326.88 | 7334.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 7370.85 | 7335.67 | 7337.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 7385.35 | 7335.67 | 7337.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 7366.00 | 7341.74 | 7340.47 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 7324.20 | 7339.87 | 7341.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 7300.15 | 7331.92 | 7337.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 10:15:00 | 7290.00 | 7286.45 | 7307.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 10:15:00 | 7290.00 | 7286.45 | 7307.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 7290.00 | 7286.45 | 7307.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:45:00 | 7314.65 | 7286.45 | 7307.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 7206.75 | 7249.13 | 7278.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:45:00 | 7237.80 | 7249.13 | 7278.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 7240.00 | 7247.24 | 7268.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:15:00 | 7222.35 | 7247.24 | 7268.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 7166.85 | 7243.56 | 7262.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 12:15:00 | 6861.23 | 6949.40 | 7055.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 6995.00 | 6933.70 | 7011.07 | SL hit (close>ema200) qty=0.50 sl=6933.70 alert=retest2 |

### Cycle 56 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 6206.85 | 6121.78 | 6110.29 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 6125.00 | 6167.17 | 6172.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 6100.30 | 6149.49 | 6162.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 6168.00 | 6139.25 | 6153.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 6168.00 | 6139.25 | 6153.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 6168.00 | 6139.25 | 6153.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:45:00 | 6168.95 | 6139.25 | 6153.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 6170.00 | 6145.40 | 6155.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 6196.15 | 6145.40 | 6155.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 6319.20 | 6180.16 | 6169.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 6337.50 | 6211.63 | 6185.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 6308.05 | 6311.80 | 6257.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:00:00 | 6308.05 | 6311.80 | 6257.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 6204.80 | 6324.06 | 6296.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:45:00 | 6202.00 | 6324.06 | 6296.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 6205.00 | 6300.25 | 6288.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 11:45:00 | 6211.35 | 6282.73 | 6281.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 12:15:00 | 6212.95 | 6268.77 | 6275.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 12:15:00 | 6212.95 | 6268.77 | 6275.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 6144.50 | 6243.92 | 6263.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 6186.65 | 6180.83 | 6216.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:45:00 | 6182.40 | 6180.83 | 6216.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 6188.40 | 6162.35 | 6194.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 6210.80 | 6162.35 | 6194.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 6189.25 | 6170.60 | 6192.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 6191.15 | 6170.60 | 6192.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 6151.25 | 6166.73 | 6189.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 6151.10 | 6166.73 | 6189.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 6174.55 | 6166.24 | 6184.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 6174.55 | 6166.24 | 6184.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 6182.75 | 6167.74 | 6182.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 6131.75 | 6161.99 | 6178.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 6130.60 | 6154.59 | 6171.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 6284.25 | 6183.87 | 6177.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 6284.25 | 6183.87 | 6177.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 6321.65 | 6211.42 | 6190.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 6310.00 | 6335.82 | 6294.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 6310.00 | 6335.82 | 6294.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 6310.00 | 6335.82 | 6294.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 14:30:00 | 6352.00 | 6343.90 | 6312.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 13:45:00 | 6355.00 | 6355.95 | 6335.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:30:00 | 6361.65 | 6360.90 | 6340.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 6376.90 | 6396.45 | 6387.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 6396.95 | 6393.47 | 6387.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:45:00 | 6414.20 | 6404.91 | 6393.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 6356.60 | 6396.86 | 6391.97 | SL hit (close<static) qty=1.00 sl=6372.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 6340.00 | 6386.58 | 6388.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 6281.55 | 6355.31 | 6371.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 6164.50 | 6159.19 | 6213.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 6093.40 | 6159.19 | 6213.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 6109.95 | 6139.22 | 6173.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 6104.15 | 6139.22 | 6173.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 6102.00 | 6133.94 | 6153.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:00:00 | 6076.35 | 6122.42 | 6146.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 6103.30 | 6123.80 | 6135.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 6096.25 | 6095.70 | 6113.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 6125.60 | 6095.70 | 6113.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 6060.00 | 6088.56 | 6108.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 12:45:00 | 6023.00 | 6067.14 | 6094.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 13:15:00 | 6020.00 | 6049.03 | 6068.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 15:00:00 | 6024.25 | 6045.05 | 6063.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 09:30:00 | 6025.00 | 6028.25 | 6052.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 5798.94 | 5954.16 | 6001.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 5796.90 | 5954.16 | 6001.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 5798.14 | 5954.16 | 6001.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 5772.53 | 5847.37 | 5910.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 13:15:00 | 5721.85 | 5789.87 | 5865.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 13:15:00 | 5719.00 | 5789.87 | 5865.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 13:15:00 | 5723.04 | 5789.87 | 5865.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 13:15:00 | 5723.75 | 5789.87 | 5865.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 09:15:00 | 5493.73 | 5573.19 | 5687.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 62 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 5534.00 | 5483.50 | 5478.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 5554.75 | 5506.57 | 5490.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 5626.40 | 5632.68 | 5589.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 5626.40 | 5632.68 | 5589.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 5611.65 | 5623.04 | 5600.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 5615.00 | 5623.04 | 5600.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 5605.35 | 5619.50 | 5600.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 5605.35 | 5619.50 | 5600.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 5577.15 | 5611.03 | 5598.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 5577.15 | 5611.03 | 5598.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 5570.70 | 5602.97 | 5595.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 5574.00 | 5602.97 | 5595.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 5505.25 | 5575.85 | 5584.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 5479.35 | 5556.55 | 5574.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 5284.70 | 5223.10 | 5267.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 5284.70 | 5223.10 | 5267.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 5284.70 | 5223.10 | 5267.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 5287.45 | 5223.10 | 5267.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 5269.00 | 5232.28 | 5267.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 5251.20 | 5246.56 | 5268.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 13:45:00 | 5268.00 | 5251.48 | 5268.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 15:00:00 | 5268.10 | 5254.80 | 5268.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 5259.55 | 5259.94 | 5269.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 5316.00 | 5271.15 | 5273.85 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 5316.00 | 5271.15 | 5273.85 | SL hit (close>static) qty=1.00 sl=5310.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 5303.90 | 5277.70 | 5276.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 12:15:00 | 5341.00 | 5295.17 | 5285.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 5379.95 | 5383.95 | 5349.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:45:00 | 5376.95 | 5383.95 | 5349.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 5423.20 | 5389.57 | 5358.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 5457.85 | 5405.59 | 5370.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:15:00 | 5459.00 | 5426.26 | 5387.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:45:00 | 5458.75 | 5446.51 | 5407.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 5425.00 | 5530.77 | 5534.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 5425.00 | 5530.77 | 5534.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 09:15:00 | 5350.00 | 5477.37 | 5508.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 5120.05 | 5114.99 | 5215.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 5120.05 | 5114.99 | 5215.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 5201.50 | 5132.29 | 5214.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 5201.50 | 5132.29 | 5214.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 5211.70 | 5148.17 | 5213.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 5211.70 | 5148.17 | 5213.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 5203.60 | 5159.26 | 5212.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 5210.85 | 5159.26 | 5212.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 5226.05 | 5172.62 | 5214.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 5226.05 | 5172.62 | 5214.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 5224.95 | 5183.08 | 5215.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 5194.45 | 5183.08 | 5215.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 10:15:00 | 5206.05 | 5190.98 | 5215.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:00:00 | 5215.80 | 5195.94 | 5215.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 5212.90 | 5194.54 | 5213.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 5216.70 | 5196.57 | 5210.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:30:00 | 5217.75 | 5196.57 | 5210.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 5222.50 | 5201.75 | 5211.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:30:00 | 5224.35 | 5201.75 | 5211.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 4934.73 | 5011.27 | 5095.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 4945.75 | 5011.27 | 5095.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 4955.01 | 5011.27 | 5095.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 4952.25 | 5011.27 | 5095.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4816.55 | 4833.56 | 4942.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 4778.00 | 4828.84 | 4930.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 11:15:00 | 4867.95 | 4836.67 | 4924.93 | SL hit (close>ema200) qty=0.50 sl=4836.67 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 4902.50 | 4832.12 | 4828.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 4931.50 | 4872.61 | 4849.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 4882.50 | 4883.85 | 4861.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 11:15:00 | 4882.50 | 4883.85 | 4861.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 4882.50 | 4883.85 | 4861.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 12:00:00 | 4882.50 | 4883.85 | 4861.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 4914.50 | 4889.69 | 4867.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 4879.00 | 4889.69 | 4867.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 4856.00 | 4895.20 | 4876.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 4919.50 | 4889.23 | 4877.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 4921.50 | 4894.58 | 4880.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:45:00 | 4930.00 | 4893.47 | 4881.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 5036.50 | 4895.86 | 4884.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 5142.50 | 4945.19 | 4908.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 5170.00 | 4945.19 | 4908.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:00:00 | 5187.00 | 4993.55 | 4933.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 09:15:00 | 5411.45 | 5258.41 | 5114.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 5754.50 | 5784.83 | 5785.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 5706.50 | 5769.17 | 5778.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 5728.50 | 5720.45 | 5747.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 5728.50 | 5720.45 | 5747.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 5716.50 | 5719.66 | 5744.24 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 5838.00 | 5766.33 | 5758.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 5882.50 | 5789.57 | 5769.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 5734.00 | 5782.20 | 5769.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 5734.00 | 5782.20 | 5769.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 5734.00 | 5782.20 | 5769.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 5734.00 | 5782.20 | 5769.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 5711.50 | 5768.06 | 5764.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 5711.50 | 5768.06 | 5764.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 5660.00 | 5746.45 | 5755.06 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 5922.00 | 5772.62 | 5760.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 5963.50 | 5835.81 | 5792.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 6016.00 | 6017.96 | 5934.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 6016.00 | 6017.96 | 5934.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 6221.00 | 6232.05 | 6204.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 6196.00 | 6232.05 | 6204.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 6218.00 | 6229.24 | 6205.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:30:00 | 6218.00 | 6229.24 | 6205.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 6203.00 | 6223.99 | 6205.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 6203.00 | 6223.99 | 6205.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 6200.00 | 6219.20 | 6204.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 6151.00 | 6219.20 | 6204.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 6190.00 | 6213.36 | 6203.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 6156.50 | 6213.36 | 6203.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 6197.50 | 6210.19 | 6202.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 6197.50 | 6210.19 | 6202.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 6139.00 | 6195.95 | 6197.07 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 6283.00 | 6202.87 | 6194.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 6358.50 | 6282.67 | 6244.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 6383.00 | 6404.81 | 6352.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 12:45:00 | 6387.00 | 6404.81 | 6352.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 6382.00 | 6399.06 | 6366.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 6383.50 | 6399.06 | 6366.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 6395.00 | 6398.25 | 6368.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 6401.50 | 6398.25 | 6368.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 6398.00 | 6440.10 | 6434.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 6394.00 | 6430.88 | 6431.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 6394.00 | 6430.88 | 6431.14 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 6447.00 | 6413.62 | 6411.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 6474.00 | 6435.69 | 6422.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 6462.00 | 6473.77 | 6454.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:00:00 | 6462.00 | 6473.77 | 6454.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 6459.50 | 6470.92 | 6454.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:30:00 | 6467.00 | 6467.13 | 6454.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:15:00 | 6471.00 | 6467.13 | 6454.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:00:00 | 6484.00 | 6470.51 | 6457.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 6544.00 | 6598.74 | 6602.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 6544.00 | 6598.74 | 6602.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 6460.00 | 6558.36 | 6582.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 6405.00 | 6389.53 | 6444.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 6405.00 | 6389.53 | 6444.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 6410.00 | 6390.53 | 6427.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 6414.00 | 6390.53 | 6427.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 6446.00 | 6401.62 | 6429.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 6446.00 | 6401.62 | 6429.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 6423.00 | 6405.90 | 6428.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 6387.00 | 6411.39 | 6423.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 14:15:00 | 6460.00 | 6416.01 | 6420.51 | SL hit (close>static) qty=1.00 sl=6446.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 6369.00 | 6284.78 | 6275.67 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 6226.00 | 6290.83 | 6298.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 6162.00 | 6265.07 | 6285.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 13:15:00 | 6189.00 | 6176.64 | 6217.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 14:00:00 | 6189.00 | 6176.64 | 6217.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 6209.00 | 6183.79 | 6210.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 6207.50 | 6183.79 | 6210.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 6229.00 | 6192.83 | 6212.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 6229.00 | 6192.83 | 6212.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 6214.00 | 6197.07 | 6212.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 6215.00 | 6197.07 | 6212.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 6254.00 | 6208.45 | 6216.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 6254.00 | 6208.45 | 6216.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 6228.00 | 6212.36 | 6217.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 6243.00 | 6212.36 | 6217.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 6217.50 | 6214.05 | 6217.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 6184.00 | 6214.05 | 6217.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 6190.50 | 6209.34 | 6214.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 6133.00 | 6169.45 | 6187.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 13:30:00 | 6150.50 | 6148.06 | 6169.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 6149.00 | 6148.06 | 6169.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 6140.00 | 6153.77 | 6167.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 6151.50 | 6144.13 | 6156.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:15:00 | 6126.00 | 6144.13 | 6156.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 6126.00 | 6140.50 | 6153.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:00:00 | 6120.50 | 6136.50 | 6150.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:45:00 | 6104.50 | 6120.75 | 6138.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 5902.50 | 6124.69 | 6135.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 5826.35 | 6087.75 | 6118.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 5842.97 | 6087.75 | 6118.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 5841.55 | 6087.75 | 6118.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 5833.00 | 6087.75 | 6118.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 5814.47 | 6087.75 | 6118.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 5799.27 | 6087.75 | 6118.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 13:15:00 | 6039.00 | 6033.61 | 6078.26 | SL hit (close>ema200) qty=0.50 sl=6033.61 alert=retest2 |

### Cycle 78 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 6156.00 | 6099.94 | 6096.10 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 6049.00 | 6089.75 | 6091.82 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 6162.00 | 6104.20 | 6098.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 6181.50 | 6119.66 | 6105.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 6331.00 | 6333.54 | 6276.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:00:00 | 6331.00 | 6333.54 | 6276.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 6306.50 | 6322.25 | 6285.34 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 6199.50 | 6259.84 | 6266.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 6192.00 | 6246.27 | 6259.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 13:15:00 | 6198.50 | 6194.95 | 6224.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:00:00 | 6198.50 | 6194.95 | 6224.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 6187.50 | 6187.07 | 6210.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 6204.50 | 6187.07 | 6210.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 6209.00 | 6193.07 | 6204.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 6209.00 | 6193.07 | 6204.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 6191.00 | 6192.66 | 6203.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 6178.00 | 6190.42 | 6200.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 6227.00 | 6207.58 | 6205.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 6227.00 | 6207.58 | 6205.31 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 6189.50 | 6206.65 | 6208.39 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 6225.50 | 6210.42 | 6209.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 6245.00 | 6217.34 | 6213.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 6205.00 | 6217.04 | 6214.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 6205.00 | 6217.04 | 6214.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6205.00 | 6217.04 | 6214.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 6186.50 | 6217.04 | 6214.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 6201.00 | 6213.83 | 6213.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 6186.00 | 6213.83 | 6213.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 6193.00 | 6209.67 | 6211.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 6161.00 | 6199.93 | 6206.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 6056.50 | 6046.39 | 6088.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 6056.50 | 6046.39 | 6088.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 6092.50 | 6055.31 | 6085.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 6092.50 | 6055.31 | 6085.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 6086.00 | 6061.45 | 6085.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 6093.50 | 6079.36 | 6091.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 6092.50 | 6081.99 | 6091.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 6044.00 | 6091.59 | 6093.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:30:00 | 6062.50 | 6079.26 | 6082.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 10:15:00 | 6059.50 | 6079.26 | 6082.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 6048.00 | 6076.69 | 6079.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 6030.50 | 6029.41 | 6047.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 6058.00 | 6029.41 | 6047.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 5986.50 | 6020.45 | 6040.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:30:00 | 5971.50 | 6000.10 | 6023.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 5964.00 | 6000.10 | 6023.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 5759.38 | 5835.27 | 5904.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:15:00 | 5741.80 | 5821.91 | 5892.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:15:00 | 5756.52 | 5821.91 | 5892.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:15:00 | 5745.60 | 5821.91 | 5892.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 5863.00 | 5821.90 | 5873.88 | SL hit (close>ema200) qty=0.50 sl=5821.90 alert=retest2 |

### Cycle 86 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 5786.00 | 5704.24 | 5698.87 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 15:15:00 | 5687.00 | 5698.11 | 5698.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 5661.50 | 5690.79 | 5695.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 5712.50 | 5694.28 | 5696.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 11:15:00 | 5712.50 | 5694.28 | 5696.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 5712.50 | 5694.28 | 5696.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 5712.50 | 5694.28 | 5696.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 5699.00 | 5695.23 | 5696.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 5692.50 | 5695.23 | 5696.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 5686.00 | 5693.68 | 5695.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 14:15:00 | 5727.50 | 5700.44 | 5698.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 5727.50 | 5700.44 | 5698.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 5743.00 | 5708.96 | 5702.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 5741.00 | 5743.38 | 5727.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 5722.00 | 5743.38 | 5727.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 5714.00 | 5737.50 | 5725.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 5710.50 | 5737.50 | 5725.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 5705.00 | 5731.00 | 5723.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 5708.00 | 5731.00 | 5723.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 5687.50 | 5717.10 | 5718.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 5660.00 | 5698.87 | 5709.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 5313.50 | 5289.81 | 5360.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 5313.50 | 5289.81 | 5360.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 5313.50 | 5289.81 | 5360.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 5347.50 | 5289.81 | 5360.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 5357.00 | 5326.17 | 5353.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 5357.00 | 5326.17 | 5353.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 5358.00 | 5332.54 | 5353.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 5319.50 | 5332.54 | 5353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 5378.00 | 5341.63 | 5355.99 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 5431.50 | 5369.58 | 5366.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 5465.00 | 5410.36 | 5389.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 5412.00 | 5412.55 | 5394.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 5412.00 | 5412.55 | 5394.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 5429.50 | 5416.09 | 5399.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 5404.00 | 5416.09 | 5399.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 5398.50 | 5414.45 | 5404.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 5398.50 | 5414.45 | 5404.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 5385.00 | 5408.56 | 5402.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 5385.00 | 5408.56 | 5402.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 5393.50 | 5415.21 | 5409.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 5393.50 | 5415.21 | 5409.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 5420.00 | 5416.17 | 5410.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 5472.50 | 5418.83 | 5412.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 5687.00 | 5697.97 | 5699.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 5687.00 | 5697.97 | 5699.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 5665.50 | 5691.47 | 5696.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 5693.00 | 5684.31 | 5690.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 5693.00 | 5684.31 | 5690.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 5693.00 | 5684.31 | 5690.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 5693.00 | 5684.31 | 5690.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 5690.00 | 5685.45 | 5690.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 5698.00 | 5685.45 | 5690.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 5659.00 | 5680.16 | 5687.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 5655.00 | 5680.16 | 5687.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 5708.00 | 5682.26 | 5687.14 | SL hit (close>static) qty=1.00 sl=5696.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 5719.00 | 5695.17 | 5692.51 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 5684.50 | 5691.36 | 5692.21 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 5746.00 | 5702.29 | 5697.10 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 5643.50 | 5702.84 | 5706.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 5546.50 | 5617.52 | 5654.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 5560.00 | 5559.82 | 5603.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 5496.50 | 5559.82 | 5603.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 5244.00 | 5228.51 | 5261.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 5250.00 | 5228.51 | 5261.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 5308.50 | 5248.84 | 5262.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 5308.50 | 5248.84 | 5262.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 5307.00 | 5260.47 | 5266.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 5314.50 | 5260.47 | 5266.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 5363.50 | 5281.08 | 5275.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 5370.00 | 5311.65 | 5291.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 5353.00 | 5362.27 | 5333.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:30:00 | 5374.50 | 5362.27 | 5333.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 5330.50 | 5355.92 | 5332.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 5330.50 | 5355.92 | 5332.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 5327.00 | 5350.13 | 5332.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 5320.00 | 5350.13 | 5332.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 5370.00 | 5354.11 | 5335.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 5377.50 | 5354.11 | 5335.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 5408.00 | 5452.77 | 5453.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 5408.00 | 5452.77 | 5453.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 15:15:00 | 5400.00 | 5442.21 | 5448.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 5367.00 | 5355.76 | 5381.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 5367.00 | 5355.76 | 5381.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 5367.00 | 5355.76 | 5381.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 5367.00 | 5355.76 | 5381.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 5358.00 | 5347.78 | 5368.93 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 5400.00 | 5371.50 | 5370.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 5410.00 | 5389.19 | 5380.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 5380.00 | 5388.28 | 5381.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 5380.00 | 5388.28 | 5381.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 5380.00 | 5388.28 | 5381.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 5379.50 | 5388.28 | 5381.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 5369.00 | 5384.42 | 5380.34 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 5365.00 | 5376.55 | 5377.22 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 5407.00 | 5381.14 | 5378.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 5429.50 | 5395.11 | 5388.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 5574.50 | 5578.91 | 5536.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 5562.00 | 5578.91 | 5536.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 5538.00 | 5570.73 | 5536.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 5538.00 | 5570.73 | 5536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5529.50 | 5562.48 | 5535.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 5529.50 | 5562.48 | 5535.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5530.50 | 5556.09 | 5535.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 5530.50 | 5556.09 | 5535.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 5539.50 | 5552.77 | 5535.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 5539.50 | 5552.77 | 5535.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 5534.00 | 5549.02 | 5535.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 5532.50 | 5549.02 | 5535.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 5546.50 | 5548.51 | 5536.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 5530.50 | 5548.51 | 5536.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 5552.00 | 5549.21 | 5537.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 5549.00 | 5549.21 | 5537.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 5492.50 | 5537.87 | 5533.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 5492.50 | 5537.87 | 5533.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 5523.00 | 5534.89 | 5532.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:00:00 | 5542.00 | 5536.32 | 5533.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 5542.00 | 5536.15 | 5533.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 5544.50 | 5537.82 | 5534.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 5560.00 | 5536.13 | 5534.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 5531.50 | 5535.20 | 5534.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 5531.50 | 5535.20 | 5534.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 5522.00 | 5532.56 | 5533.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 5522.00 | 5532.56 | 5533.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 5509.00 | 5523.97 | 5528.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 5450.00 | 5438.40 | 5466.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 5429.00 | 5438.40 | 5466.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 5412.00 | 5433.12 | 5461.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 5405.00 | 5424.89 | 5455.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 5134.75 | 5251.19 | 5321.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 5213.00 | 5195.82 | 5246.45 | SL hit (close>ema200) qty=0.50 sl=5195.82 alert=retest2 |

### Cycle 102 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 5281.00 | 5248.39 | 5244.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 5296.00 | 5257.91 | 5249.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 5370.00 | 5374.96 | 5331.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 5333.50 | 5360.11 | 5338.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 5333.50 | 5360.11 | 5338.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 5333.50 | 5360.11 | 5338.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 5309.00 | 5349.89 | 5335.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 5309.00 | 5349.89 | 5335.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 5306.00 | 5341.11 | 5332.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 5247.00 | 5341.11 | 5332.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 5269.00 | 5326.69 | 5326.95 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 5316.00 | 5292.36 | 5291.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 5362.50 | 5306.39 | 5297.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 5314.50 | 5354.81 | 5341.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 5314.50 | 5354.81 | 5341.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 5314.50 | 5354.81 | 5341.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 5398.00 | 5337.01 | 5336.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 5367.50 | 5348.16 | 5343.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 5264.00 | 5327.54 | 5334.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 5264.00 | 5327.54 | 5334.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 5162.00 | 5294.43 | 5318.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 5213.00 | 5201.30 | 5245.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 5213.00 | 5201.30 | 5245.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 5237.00 | 5215.13 | 5238.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 5217.00 | 5222.24 | 5235.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 5217.50 | 5160.96 | 5156.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 5217.50 | 5160.96 | 5156.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 5239.50 | 5185.24 | 5170.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 14:15:00 | 5226.50 | 5231.26 | 5213.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 5226.50 | 5231.26 | 5213.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 5226.50 | 5231.26 | 5213.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 5212.00 | 5231.26 | 5213.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 5217.00 | 5228.41 | 5213.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 5229.00 | 5228.41 | 5213.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 5164.00 | 5215.52 | 5208.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 5164.00 | 5215.52 | 5208.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 5166.50 | 5205.72 | 5205.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:15:00 | 5153.50 | 5205.72 | 5205.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 5136.00 | 5191.78 | 5198.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 5116.50 | 5176.72 | 5191.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 4913.00 | 4912.56 | 4980.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 4913.00 | 4912.56 | 4980.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 4964.00 | 4930.36 | 4967.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 4964.00 | 4930.36 | 4967.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 5018.50 | 4947.99 | 4972.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 5018.50 | 4947.99 | 4972.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 5010.00 | 4960.39 | 4975.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 5028.00 | 4960.39 | 4975.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 5008.00 | 4987.23 | 4985.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 5024.00 | 4994.58 | 4988.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 4995.00 | 5032.73 | 5020.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 4995.00 | 5032.73 | 5020.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 4995.00 | 5032.73 | 5020.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 5005.00 | 5032.73 | 5020.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 5004.00 | 5026.99 | 5019.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 5009.50 | 5026.99 | 5019.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 5021.00 | 5025.79 | 5019.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 4994.00 | 5013.36 | 5015.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 4994.00 | 5013.36 | 5015.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 4968.50 | 4994.45 | 5004.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 4996.50 | 4980.92 | 4992.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 4996.50 | 4980.92 | 4992.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 4996.50 | 4980.92 | 4992.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 4996.50 | 4980.92 | 4992.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 5002.00 | 4985.14 | 4993.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 5002.00 | 4985.14 | 4993.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4975.50 | 4983.21 | 4991.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 4960.50 | 4983.21 | 4991.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 5046.00 | 4997.98 | 4997.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 5046.00 | 4997.98 | 4997.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 5176.00 | 5033.58 | 5013.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 5434.00 | 5437.68 | 5355.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 5434.00 | 5437.68 | 5355.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 5382.50 | 5406.09 | 5378.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:45:00 | 5380.00 | 5406.09 | 5378.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 5382.00 | 5401.27 | 5378.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 5387.50 | 5401.27 | 5378.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 5390.50 | 5399.12 | 5379.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 5379.00 | 5399.12 | 5379.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 5395.00 | 5397.32 | 5382.13 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 5352.50 | 5371.99 | 5374.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 5328.00 | 5358.08 | 5367.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 5320.00 | 5319.85 | 5339.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 5298.50 | 5319.85 | 5339.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 5287.00 | 5313.28 | 5334.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 5277.50 | 5313.28 | 5334.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 5303.50 | 5249.29 | 5245.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 5303.50 | 5249.29 | 5245.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5316.00 | 5271.55 | 5257.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 5332.50 | 5348.25 | 5322.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 5351.00 | 5347.46 | 5328.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 5351.00 | 5347.46 | 5328.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 5527.50 | 5346.57 | 5330.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 5562.00 | 5679.41 | 5692.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 5562.00 | 5679.41 | 5692.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 5520.00 | 5607.04 | 5652.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 11:15:00 | 5670.50 | 5588.78 | 5625.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 11:15:00 | 5670.50 | 5588.78 | 5625.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 5670.50 | 5588.78 | 5625.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 5670.50 | 5588.78 | 5625.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 5637.50 | 5598.52 | 5626.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 5611.00 | 5608.53 | 5626.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 5330.45 | 5428.63 | 5494.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 5464.00 | 5384.66 | 5431.54 | SL hit (close>ema200) qty=0.50 sl=5384.66 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 5428.00 | 5359.00 | 5351.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 5500.00 | 5426.53 | 5397.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 12:15:00 | 5458.00 | 5459.36 | 5422.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 13:00:00 | 5458.00 | 5459.36 | 5422.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 5367.00 | 5453.97 | 5432.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:15:00 | 5425.00 | 5432.50 | 5425.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 5440.00 | 5467.32 | 5450.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 5411.50 | 5439.97 | 5440.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 5411.50 | 5439.97 | 5440.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 5390.00 | 5426.14 | 5433.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 13:15:00 | 5233.50 | 5229.78 | 5281.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 13:45:00 | 5241.50 | 5229.78 | 5281.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 5338.00 | 5250.05 | 5277.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:45:00 | 5343.00 | 5250.05 | 5277.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 5358.00 | 5271.64 | 5284.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 5362.50 | 5271.64 | 5284.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 5371.00 | 5304.37 | 5298.18 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 5268.00 | 5306.20 | 5306.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 14:15:00 | 5253.00 | 5291.85 | 5299.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 4824.50 | 4816.36 | 4908.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 4824.50 | 4816.36 | 4908.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 4939.00 | 4840.03 | 4902.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 4939.00 | 4840.03 | 4902.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 4930.00 | 4858.02 | 4905.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 4913.00 | 4858.02 | 4905.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 4904.00 | 4911.40 | 4919.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 4906.00 | 4911.40 | 4919.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 4833.00 | 4895.72 | 4911.55 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 4953.00 | 4905.98 | 4904.55 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 4838.00 | 4894.58 | 4900.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 4828.00 | 4881.26 | 4893.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 4881.50 | 4875.99 | 4888.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 4881.50 | 4875.99 | 4888.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 4881.50 | 4875.99 | 4888.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 4898.00 | 4875.99 | 4888.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 4868.00 | 4874.39 | 4886.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 4828.00 | 4849.75 | 4869.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 4586.60 | 4717.19 | 4783.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 4602.00 | 4568.30 | 4656.38 | SL hit (close>ema200) qty=0.50 sl=4568.30 alert=retest2 |

### Cycle 120 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 4357.00 | 4331.13 | 4329.35 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 4259.80 | 4333.71 | 4335.96 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 4357.50 | 4338.58 | 4337.79 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 4255.40 | 4323.31 | 4331.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 4251.00 | 4297.47 | 4316.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 4212.40 | 4198.48 | 4245.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 4215.70 | 4198.48 | 4245.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 4244.30 | 4152.53 | 4185.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:15:00 | 4247.50 | 4152.53 | 4185.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 4284.00 | 4178.83 | 4194.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 4284.00 | 4178.83 | 4194.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 4253.70 | 4205.48 | 4204.84 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 4104.30 | 4187.35 | 4197.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 4079.00 | 4165.68 | 4186.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 4165.00 | 4108.25 | 4141.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 4165.00 | 4108.25 | 4141.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 4165.00 | 4108.25 | 4141.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 4178.00 | 4108.25 | 4141.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 4241.00 | 4134.80 | 4150.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 4241.00 | 4134.80 | 4150.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 4250.00 | 4172.87 | 4166.06 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 4118.90 | 4165.61 | 4170.08 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 4189.90 | 4165.86 | 4164.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4248.10 | 4184.74 | 4174.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 4216.00 | 4224.82 | 4201.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 4216.00 | 4224.82 | 4201.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 4204.50 | 4220.76 | 4201.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 4252.00 | 4220.76 | 4201.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4207.90 | 4218.19 | 4202.48 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 4162.00 | 4195.99 | 4196.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 4144.00 | 4180.71 | 4188.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4191.50 | 4079.33 | 4117.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4191.50 | 4079.33 | 4117.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4191.50 | 4079.33 | 4117.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 4191.50 | 4079.33 | 4117.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4172.30 | 4097.92 | 4122.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 4208.70 | 4097.92 | 4122.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 4128.70 | 4124.70 | 4129.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 4128.70 | 4124.70 | 4129.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 4131.00 | 4125.96 | 4129.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 4064.10 | 4125.96 | 4129.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:00:00 | 4112.00 | 4108.43 | 4119.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 4169.00 | 4120.55 | 4123.75 | SL hit (close>static) qty=1.00 sl=4137.80 alert=retest2 |

### Cycle 130 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 4230.00 | 4142.44 | 4133.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 4252.00 | 4164.35 | 4144.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 4378.10 | 4402.03 | 4353.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 4378.10 | 4402.03 | 4353.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4401.30 | 4431.65 | 4396.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 4422.80 | 4431.65 | 4396.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 4417.00 | 4428.72 | 4398.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 4443.10 | 4430.57 | 4404.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 14:30:00 | 4439.90 | 4434.13 | 4410.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 4383.70 | 4426.26 | 4411.52 | SL hit (close<static) qty=1.00 sl=4391.10 alert=retest2 |

### Cycle 131 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 4511.80 | 4542.08 | 4543.98 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 4598.00 | 4545.20 | 4544.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 4617.20 | 4559.60 | 4550.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 4440.00 | 4581.64 | 4572.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 4440.00 | 4581.64 | 4572.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4440.00 | 4581.64 | 4572.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 4440.00 | 4581.64 | 4572.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 4428.90 | 4551.09 | 4559.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 4390.10 | 4518.89 | 4544.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 4233.20 | 4219.36 | 4284.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 4267.00 | 4219.36 | 4284.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 4270.00 | 4229.48 | 4283.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 4270.00 | 4229.48 | 4283.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 4178.00 | 4159.91 | 4198.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 4170.00 | 4159.91 | 4198.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 4144.00 | 4139.45 | 4163.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 4151.70 | 4139.45 | 4163.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4155.80 | 4142.72 | 4162.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 4155.80 | 4142.72 | 4162.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 4136.90 | 4141.55 | 4160.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 4132.40 | 4141.55 | 4160.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 4177.10 | 4150.78 | 4160.04 | SL hit (close>static) qty=1.00 sl=4171.40 alert=retest2 |

### Cycle 134 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 4186.70 | 4165.22 | 4163.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 4196.00 | 4171.38 | 4166.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 4313.00 | 4314.97 | 4291.21 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 11:30:00 | 7069.80 | 2024-05-15 09:15:00 | 7088.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-05-30 11:45:00 | 7176.10 | 2024-06-04 09:15:00 | 6817.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 12:15:00 | 7180.00 | 2024-06-04 09:15:00 | 6821.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 13:15:00 | 7182.50 | 2024-06-04 09:15:00 | 6823.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 14:30:00 | 7171.40 | 2024-06-04 09:15:00 | 6812.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 10:30:00 | 7140.30 | 2024-06-04 10:15:00 | 6783.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 11:45:00 | 7176.10 | 2024-06-04 12:15:00 | 6458.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-30 12:15:00 | 7180.00 | 2024-06-04 12:15:00 | 6462.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-30 13:15:00 | 7182.50 | 2024-06-04 12:15:00 | 6464.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-30 14:30:00 | 7171.40 | 2024-06-04 12:15:00 | 6454.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-31 10:30:00 | 7140.30 | 2024-06-04 12:15:00 | 6426.27 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-02 14:15:00 | 7067.00 | 2024-07-05 10:15:00 | 7044.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-07-03 09:15:00 | 7076.15 | 2024-07-05 10:15:00 | 7044.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-07-03 13:15:00 | 7071.15 | 2024-07-05 10:15:00 | 7044.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-03 14:30:00 | 7080.00 | 2024-07-05 10:15:00 | 7044.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-07-09 09:15:00 | 6988.40 | 2024-07-10 10:15:00 | 7090.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-07-09 11:00:00 | 7011.95 | 2024-07-10 10:15:00 | 7090.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-07-09 12:00:00 | 7015.00 | 2024-07-10 10:15:00 | 7090.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-07-09 14:30:00 | 7011.90 | 2024-07-10 10:15:00 | 7090.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-07-22 09:15:00 | 6938.40 | 2024-07-23 14:15:00 | 7018.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-22 10:15:00 | 6935.00 | 2024-07-23 14:15:00 | 7018.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-07-23 09:30:00 | 6940.00 | 2024-07-23 14:15:00 | 7018.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-07-23 10:45:00 | 6924.00 | 2024-07-23 14:15:00 | 7018.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-07-30 09:15:00 | 6940.10 | 2024-07-30 14:15:00 | 6928.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-30 10:15:00 | 6939.60 | 2024-07-30 14:15:00 | 6928.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-30 11:30:00 | 6935.30 | 2024-07-30 14:15:00 | 6928.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-07-30 12:45:00 | 6936.00 | 2024-07-30 14:15:00 | 6928.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-08-06 14:30:00 | 6751.65 | 2024-08-08 12:15:00 | 6822.65 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-07 11:00:00 | 6767.00 | 2024-08-08 12:15:00 | 6822.65 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-08-12 11:15:00 | 6884.00 | 2024-08-13 11:15:00 | 6778.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-14 09:15:00 | 6764.75 | 2024-08-14 14:15:00 | 6828.10 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-08-23 09:15:00 | 6993.70 | 2024-08-26 14:15:00 | 7693.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 11:45:00 | 6981.70 | 2024-08-26 14:15:00 | 7679.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-05 14:15:00 | 7764.50 | 2024-09-09 15:15:00 | 7823.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-09-06 09:45:00 | 7765.85 | 2024-09-09 15:15:00 | 7823.95 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-09-18 09:30:00 | 7675.50 | 2024-09-20 11:15:00 | 7790.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-09-18 13:00:00 | 7657.00 | 2024-09-20 11:15:00 | 7790.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-09-19 09:45:00 | 7641.00 | 2024-09-20 11:15:00 | 7790.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-09-20 10:45:00 | 7674.05 | 2024-09-20 11:15:00 | 7790.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-26 09:30:00 | 7881.50 | 2024-09-27 15:15:00 | 7782.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-26 11:30:00 | 7846.85 | 2024-09-27 15:15:00 | 7782.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-09-26 12:15:00 | 7863.65 | 2024-09-27 15:15:00 | 7782.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-09-27 09:15:00 | 8040.00 | 2024-09-27 15:15:00 | 7782.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-10-10 09:15:00 | 7766.85 | 2024-10-14 10:15:00 | 7600.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-21 11:15:00 | 7352.05 | 2024-10-24 12:15:00 | 6984.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 7354.05 | 2024-10-24 12:15:00 | 6986.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:15:00 | 7352.05 | 2024-10-25 14:15:00 | 6978.85 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2024-10-22 09:15:00 | 7354.05 | 2024-10-25 14:15:00 | 6978.85 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2024-11-05 11:45:00 | 7000.05 | 2024-11-06 09:15:00 | 7168.30 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-11-21 12:00:00 | 6501.25 | 2024-11-21 14:15:00 | 6491.35 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-12-12 09:15:00 | 7400.40 | 2024-12-12 11:15:00 | 7335.15 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-19 14:15:00 | 7222.35 | 2024-12-23 12:15:00 | 6861.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 14:15:00 | 7222.35 | 2024-12-24 09:15:00 | 6995.00 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2024-12-20 09:15:00 | 7166.85 | 2024-12-26 09:15:00 | 6808.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 09:15:00 | 7166.85 | 2024-12-26 12:15:00 | 6887.85 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest2 | 2025-01-27 11:45:00 | 6211.35 | 2025-01-27 12:15:00 | 6212.95 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-01-30 10:45:00 | 6131.75 | 2025-01-31 10:15:00 | 6284.25 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-01-30 13:15:00 | 6130.60 | 2025-01-31 10:15:00 | 6284.25 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-02-03 14:30:00 | 6352.00 | 2025-02-10 09:15:00 | 6356.60 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-02-04 13:45:00 | 6355.00 | 2025-02-10 11:15:00 | 6340.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-02-04 14:30:00 | 6361.65 | 2025-02-10 11:15:00 | 6340.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-02-07 10:30:00 | 6376.90 | 2025-02-10 11:15:00 | 6340.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-02-07 14:45:00 | 6414.20 | 2025-02-10 11:15:00 | 6340.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-02-14 10:15:00 | 6104.15 | 2025-02-24 09:15:00 | 5798.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 10:00:00 | 6102.00 | 2025-02-24 09:15:00 | 5796.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 11:00:00 | 6076.35 | 2025-02-24 09:15:00 | 5798.14 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2025-02-18 11:00:00 | 6103.30 | 2025-02-25 10:15:00 | 5772.53 | PARTIAL | 0.50 | 5.42% |
| SELL | retest2 | 2025-02-19 12:45:00 | 6023.00 | 2025-02-25 13:15:00 | 5721.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 13:15:00 | 6020.00 | 2025-02-25 13:15:00 | 5719.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 15:00:00 | 6024.25 | 2025-02-25 13:15:00 | 5723.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 09:30:00 | 6025.00 | 2025-02-25 13:15:00 | 5723.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 6104.15 | 2025-02-28 09:15:00 | 5493.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-17 10:00:00 | 6102.00 | 2025-02-28 09:15:00 | 5491.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-17 11:00:00 | 6076.35 | 2025-02-28 09:15:00 | 5468.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-18 11:00:00 | 6103.30 | 2025-02-28 09:15:00 | 5492.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-19 12:45:00 | 6023.00 | 2025-02-28 09:15:00 | 5420.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-20 13:15:00 | 6020.00 | 2025-02-28 09:15:00 | 5418.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-20 15:00:00 | 6024.25 | 2025-02-28 09:15:00 | 5421.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-21 09:30:00 | 6025.00 | 2025-02-28 09:15:00 | 5422.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-18 12:30:00 | 5251.20 | 2025-03-19 09:15:00 | 5316.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-18 13:45:00 | 5268.00 | 2025-03-19 09:15:00 | 5316.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-03-18 15:00:00 | 5268.10 | 2025-03-19 09:15:00 | 5316.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-03-19 09:15:00 | 5259.55 | 2025-03-19 09:15:00 | 5316.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-03-21 12:00:00 | 5457.85 | 2025-03-27 14:15:00 | 5425.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-03-21 14:15:00 | 5459.00 | 2025-03-27 14:15:00 | 5425.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-03-24 09:45:00 | 5458.75 | 2025-03-27 14:15:00 | 5425.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-04-03 09:15:00 | 5194.45 | 2025-04-07 09:15:00 | 4934.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 10:15:00 | 5206.05 | 2025-04-07 09:15:00 | 4945.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 11:00:00 | 5215.80 | 2025-04-07 09:15:00 | 4955.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 11:30:00 | 5212.90 | 2025-04-07 09:15:00 | 4952.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 5194.45 | 2025-04-08 11:15:00 | 4867.95 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2025-04-03 10:15:00 | 5206.05 | 2025-04-08 11:15:00 | 4867.95 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2025-04-03 11:00:00 | 5215.80 | 2025-04-08 11:15:00 | 4867.95 | STOP_HIT | 0.50 | 6.67% |
| SELL | retest2 | 2025-04-03 11:30:00 | 5212.90 | 2025-04-08 11:15:00 | 4867.95 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2025-04-08 10:30:00 | 4778.00 | 2025-04-15 12:15:00 | 4902.50 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-04-09 09:45:00 | 4755.60 | 2025-04-15 12:15:00 | 4902.50 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-04-09 14:00:00 | 4784.70 | 2025-04-15 12:15:00 | 4902.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-04-11 13:00:00 | 4786.00 | 2025-04-15 12:15:00 | 4902.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-04-17 12:15:00 | 4919.50 | 2025-04-22 09:15:00 | 5411.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:45:00 | 4921.50 | 2025-04-22 09:15:00 | 5413.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 13:45:00 | 4930.00 | 2025-04-22 09:15:00 | 5423.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 5036.50 | 2025-04-23 09:15:00 | 5540.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 10:15:00 | 5170.00 | 2025-04-24 09:15:00 | 5687.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 11:00:00 | 5187.00 | 2025-04-24 09:15:00 | 5705.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 11:15:00 | 6401.50 | 2025-06-02 10:15:00 | 6394.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-06-02 09:30:00 | 6398.00 | 2025-06-02 10:15:00 | 6394.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-06-06 13:30:00 | 6467.00 | 2025-06-12 10:15:00 | 6544.00 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2025-06-06 14:15:00 | 6471.00 | 2025-06-12 10:15:00 | 6544.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2025-06-06 15:00:00 | 6484.00 | 2025-06-12 10:15:00 | 6544.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-06-18 10:45:00 | 6387.00 | 2025-06-18 14:15:00 | 6460.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-19 09:15:00 | 6391.00 | 2025-06-27 09:15:00 | 6369.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-06-19 09:45:00 | 6384.50 | 2025-06-27 09:15:00 | 6369.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-08 09:30:00 | 6133.00 | 2025-07-11 09:15:00 | 5826.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 13:30:00 | 6150.50 | 2025-07-11 09:15:00 | 5842.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 14:00:00 | 6149.00 | 2025-07-11 09:15:00 | 5841.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 09:45:00 | 6140.00 | 2025-07-11 09:15:00 | 5833.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 10:00:00 | 6120.50 | 2025-07-11 09:15:00 | 5814.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 12:45:00 | 6104.50 | 2025-07-11 09:15:00 | 5799.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:30:00 | 6133.00 | 2025-07-11 13:15:00 | 6039.00 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-07-08 13:30:00 | 6150.50 | 2025-07-11 13:15:00 | 6039.00 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2025-07-08 14:00:00 | 6149.00 | 2025-07-11 13:15:00 | 6039.00 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2025-07-09 09:45:00 | 6140.00 | 2025-07-11 13:15:00 | 6039.00 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2025-07-10 10:00:00 | 6120.50 | 2025-07-11 13:15:00 | 6039.00 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2025-07-10 12:45:00 | 6104.50 | 2025-07-11 13:15:00 | 6039.00 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-07-11 09:15:00 | 5902.50 | 2025-07-14 10:15:00 | 6172.50 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-07-22 12:00:00 | 6178.00 | 2025-07-23 10:15:00 | 6227.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-31 09:15:00 | 6044.00 | 2025-08-07 10:15:00 | 5759.38 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-08-01 09:30:00 | 6062.50 | 2025-08-07 11:15:00 | 5741.80 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2025-08-01 10:15:00 | 6059.50 | 2025-08-07 11:15:00 | 5756.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 14:15:00 | 6048.00 | 2025-08-07 11:15:00 | 5745.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 6044.00 | 2025-08-07 14:15:00 | 5863.00 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-08-01 09:30:00 | 6062.50 | 2025-08-07 14:15:00 | 5863.00 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-08-01 10:15:00 | 6059.50 | 2025-08-07 14:15:00 | 5863.00 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2025-08-01 14:15:00 | 6048.00 | 2025-08-07 14:15:00 | 5863.00 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-08-05 13:30:00 | 5971.50 | 2025-08-11 11:15:00 | 5672.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 14:00:00 | 5964.00 | 2025-08-11 15:15:00 | 5665.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 13:30:00 | 5971.50 | 2025-08-12 09:15:00 | 5712.00 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2025-08-05 14:00:00 | 5964.00 | 2025-08-12 09:15:00 | 5712.00 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-08-19 13:15:00 | 5692.50 | 2025-08-19 14:15:00 | 5727.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-19 13:45:00 | 5686.00 | 2025-08-19 14:15:00 | 5727.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-05 15:15:00 | 5472.50 | 2025-09-15 12:15:00 | 5687.00 | STOP_HIT | 1.00 | 3.92% |
| SELL | retest2 | 2025-09-16 13:15:00 | 5655.00 | 2025-09-16 14:15:00 | 5708.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-07 13:15:00 | 5377.50 | 2025-10-10 14:15:00 | 5408.00 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-10-29 12:00:00 | 5542.00 | 2025-10-30 10:15:00 | 5522.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-29 13:15:00 | 5542.00 | 2025-10-30 10:15:00 | 5522.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-29 14:00:00 | 5544.50 | 2025-10-30 10:15:00 | 5522.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-10-30 09:15:00 | 5560.00 | 2025-10-30 10:15:00 | 5522.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-04 10:30:00 | 5405.00 | 2025-11-07 09:15:00 | 5134.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:30:00 | 5405.00 | 2025-11-10 10:15:00 | 5213.00 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2025-11-24 09:15:00 | 5398.00 | 2025-11-24 14:15:00 | 5264.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-11-24 13:15:00 | 5367.50 | 2025-11-24 14:15:00 | 5264.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-27 11:00:00 | 5217.00 | 2025-12-03 11:15:00 | 5217.50 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-12-16 11:15:00 | 5009.50 | 2025-12-16 15:15:00 | 4994.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-12-16 12:00:00 | 5021.00 | 2025-12-16 15:15:00 | 4994.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-18 14:15:00 | 4960.50 | 2025-12-18 15:15:00 | 5046.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-12-30 10:15:00 | 5277.50 | 2026-01-02 12:15:00 | 5303.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2026-01-07 09:15:00 | 5527.50 | 2026-01-14 11:15:00 | 5562.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2026-01-16 15:00:00 | 5611.00 | 2026-01-21 09:15:00 | 5330.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 15:00:00 | 5611.00 | 2026-01-22 09:15:00 | 5464.00 | STOP_HIT | 0.50 | 2.62% |
| BUY | retest2 | 2026-02-04 12:15:00 | 5425.00 | 2026-02-05 13:15:00 | 5411.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-05 11:00:00 | 5440.00 | 2026-02-05 13:15:00 | 5411.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-02-23 09:30:00 | 4828.00 | 2026-02-24 09:15:00 | 4586.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 4828.00 | 2026-02-25 09:15:00 | 4602.00 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2026-04-02 09:15:00 | 4064.10 | 2026-04-02 12:15:00 | 4169.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-04-02 12:00:00 | 4112.00 | 2026-04-02 12:15:00 | 4169.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-10 12:30:00 | 4443.10 | 2026-04-13 09:15:00 | 4383.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-04-10 14:30:00 | 4439.90 | 2026-04-13 09:15:00 | 4383.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-15 09:15:00 | 4515.00 | 2026-04-20 13:15:00 | 4511.80 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-04-30 15:15:00 | 4132.40 | 2026-05-04 10:15:00 | 4177.10 | STOP_HIT | 1.00 | -1.08% |
