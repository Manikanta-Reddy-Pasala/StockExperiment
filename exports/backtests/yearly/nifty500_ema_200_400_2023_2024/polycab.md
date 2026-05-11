# Polycab India Ltd. (POLYCAB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 9080.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 23 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 20
- **Target hits / Stop hits / Partials:** 4 / 22 / 2
- **Avg / median % per leg:** 0.03% / -1.32%
- **Sum % (uncompounded):** 0.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 4 | 18.2% | 4 | 18 | 0 | -0.33% | -7.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.95% | -17.8% |
| BUY @ 3rd Alert (retest2) | 19 | 4 | 21.1% | 4 | 15 | 0 | 0.56% | 10.7% |
| SELL (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.36% | 8.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.36% | 8.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.95% | -17.8% |
| retest2 (combined) | 25 | 8 | 32.0% | 4 | 19 | 2 | 0.75% | 18.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 11:15:00 | 3974.95 | 5223.30 | 5226.28 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 15:15:00 | 5019.40 | 4784.66 | 4784.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 5104.00 | 4787.83 | 4785.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 09:15:00 | 6699.85 | 6809.94 | 6373.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:15:00 | 6802.80 | 6809.44 | 6375.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 12:45:00 | 6774.50 | 6808.60 | 6378.95 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:45:00 | 6779.00 | 6801.68 | 6398.38 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 6447.00 | 6750.14 | 6437.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 14:30:00 | 6479.00 | 6738.73 | 6437.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 6381.70 | 6732.38 | 6437.35 | SL hit (close<ema400) qty=1.00 sl=6437.35 alert=retest1 |

### Cycle 3 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 6437.80 | 6774.98 | 6775.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 6399.00 | 6767.89 | 6771.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 6824.10 | 6759.91 | 6767.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 6824.10 | 6759.91 | 6767.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 6824.10 | 6759.91 | 6767.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:45:00 | 6832.70 | 6759.91 | 6767.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 6833.85 | 6760.65 | 6767.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 6833.85 | 6760.65 | 6767.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 6784.00 | 6772.32 | 6773.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 6762.30 | 6772.32 | 6773.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 13:30:00 | 6753.45 | 6771.70 | 6772.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 6424.18 | 6751.58 | 6762.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 6415.78 | 6751.58 | 6762.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 6733.00 | 6653.75 | 6705.44 | SL hit (close>ema200) qty=0.50 sl=6653.75 alert=retest2 |

### Cycle 4 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 7288.05 | 6750.51 | 6748.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 7360.10 | 6767.50 | 6757.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 7156.05 | 7189.47 | 7023.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:30:00 | 7203.05 | 7189.47 | 7023.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 7073.40 | 7185.62 | 7029.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:30:00 | 7055.00 | 7185.62 | 7029.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 7045.90 | 7204.78 | 7077.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 7045.90 | 7204.78 | 7077.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 7060.00 | 7203.33 | 7077.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 7026.00 | 7203.33 | 7077.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 7007.70 | 7201.39 | 7077.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:00:00 | 7007.70 | 7201.39 | 7077.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 7116.95 | 7192.08 | 7075.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 12:30:00 | 7127.05 | 7190.34 | 7075.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 6993.85 | 7187.15 | 7076.52 | SL hit (close<static) qty=1.00 sl=7062.10 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 6481.10 | 6986.65 | 6987.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 6437.40 | 6905.72 | 6944.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 5379.85 | 5353.17 | 5764.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 11:00:00 | 5379.85 | 5353.17 | 5764.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 5529.00 | 5207.14 | 5455.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:30:00 | 5544.00 | 5207.14 | 5455.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 5539.50 | 5210.45 | 5456.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:45:00 | 5544.00 | 5210.45 | 5456.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 5463.00 | 5272.48 | 5464.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:45:00 | 5486.50 | 5272.48 | 5464.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 5457.00 | 5274.31 | 5464.26 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 5946.00 | 5571.86 | 5571.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 5979.50 | 5575.91 | 5573.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 5891.00 | 5953.68 | 5839.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 5891.00 | 5953.68 | 5839.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 5879.50 | 5952.95 | 5839.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 5841.00 | 5952.95 | 5839.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5810.00 | 5950.64 | 5839.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5810.00 | 5950.64 | 5839.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5855.00 | 5949.69 | 5839.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:30:00 | 5871.00 | 5945.25 | 5839.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 5948.00 | 5944.96 | 5840.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 09:15:00 | 6458.10 | 5994.80 | 5875.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 7305.50 | 7393.33 | 7393.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 7270.00 | 7392.10 | 7392.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 7368.50 | 7349.88 | 7370.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 7437.50 | 7350.75 | 7370.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 7437.50 | 7350.75 | 7370.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 7435.00 | 7351.59 | 7371.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 7595.00 | 7351.59 | 7371.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 7653.00 | 7390.11 | 7389.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 7805.00 | 7459.35 | 7427.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 7536.50 | 7571.57 | 7498.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:30:00 | 7535.00 | 7571.57 | 7498.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 7486.50 | 7569.94 | 7498.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 7486.50 | 7569.94 | 7498.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 7424.50 | 7568.49 | 7498.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 7424.50 | 7568.49 | 7498.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 7517.50 | 7567.82 | 7498.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 7397.00 | 7567.82 | 7498.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 7395.00 | 7566.10 | 7498.21 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 7038.50 | 7442.15 | 7443.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 14:15:00 | 6991.50 | 7433.49 | 7439.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 7812.50 | 7390.98 | 7390.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 7827.50 | 7472.70 | 7434.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 7959.50 | 7964.14 | 7741.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 12:00:00 | 7959.50 | 7964.14 | 7741.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 7774.00 | 7960.13 | 7742.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 7774.00 | 7960.13 | 7742.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 7722.00 | 7957.77 | 7741.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 7752.00 | 7957.77 | 7741.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 7725.00 | 7955.45 | 7741.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 7619.50 | 7955.45 | 7741.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 6800.00 | 7592.16 | 7592.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 6769.50 | 7416.95 | 7497.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 8253.00 | 7531.77 | 7529.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 8271.00 | 7733.81 | 7642.27 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-28 11:15:00 | 6802.80 | 2024-07-10 09:15:00 | 6381.70 | STOP_HIT | 1.00 | -6.19% |
| BUY | retest1 | 2024-06-28 12:45:00 | 6774.50 | 2024-07-10 09:15:00 | 6381.70 | STOP_HIT | 1.00 | -5.80% |
| BUY | retest1 | 2024-07-02 09:45:00 | 6779.00 | 2024-07-10 09:15:00 | 6381.70 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest2 | 2024-07-09 14:30:00 | 6479.00 | 2024-07-10 09:15:00 | 6381.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-11 09:45:00 | 6505.90 | 2024-07-19 09:15:00 | 6380.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-07-26 11:30:00 | 6477.00 | 2024-08-06 14:15:00 | 6396.75 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-08-05 11:15:00 | 6482.10 | 2024-08-06 14:15:00 | 6396.75 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-08-06 09:15:00 | 6555.70 | 2024-08-06 14:15:00 | 6396.75 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-08-06 10:45:00 | 6560.00 | 2024-08-06 14:15:00 | 6396.75 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-08-06 13:00:00 | 6534.85 | 2024-08-06 14:15:00 | 6396.75 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-08-07 09:15:00 | 6540.00 | 2024-08-13 14:15:00 | 6425.45 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-16 12:00:00 | 6650.50 | 2024-09-19 09:15:00 | 6573.65 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-08-19 09:30:00 | 6641.95 | 2024-09-19 11:15:00 | 6488.05 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-19 15:15:00 | 6677.30 | 2024-09-19 11:15:00 | 6488.05 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-09-05 15:00:00 | 6638.00 | 2024-09-19 11:15:00 | 6488.05 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-09-09 15:00:00 | 6678.95 | 2024-09-19 11:15:00 | 6488.05 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-09-24 15:00:00 | 6700.00 | 2024-10-01 12:15:00 | 7315.22 | TARGET_HIT | 1.00 | 9.18% |
| BUY | retest2 | 2024-09-25 15:00:00 | 6650.20 | 2024-10-03 09:15:00 | 7370.00 | TARGET_HIT | 1.00 | 10.82% |
| BUY | retest2 | 2024-10-22 15:15:00 | 6655.90 | 2024-10-24 10:15:00 | 6570.15 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-11-08 11:15:00 | 6762.30 | 2024-11-13 09:15:00 | 6424.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 13:30:00 | 6753.45 | 2024-11-13 09:15:00 | 6415.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 6762.30 | 2024-11-25 09:15:00 | 6733.00 | STOP_HIT | 0.50 | 0.43% |
| SELL | retest2 | 2024-11-08 13:30:00 | 6753.45 | 2024-11-25 09:15:00 | 6733.00 | STOP_HIT | 0.50 | 0.30% |
| SELL | retest2 | 2024-11-25 15:00:00 | 6768.70 | 2024-11-26 13:15:00 | 6853.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-11-26 09:30:00 | 6763.50 | 2024-11-26 13:15:00 | 6853.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-01-07 12:30:00 | 7127.05 | 2025-01-08 09:15:00 | 6993.85 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-20 10:30:00 | 5871.00 | 2025-06-25 09:15:00 | 6458.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 11:30:00 | 5948.00 | 2025-06-27 09:15:00 | 6542.80 | TARGET_HIT | 1.00 | 10.00% |
