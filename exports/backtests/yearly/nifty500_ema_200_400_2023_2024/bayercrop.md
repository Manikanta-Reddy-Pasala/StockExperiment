# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 4600.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 37
- **Target hits / Stop hits / Partials:** 7 / 38 / 7
- **Avg / median % per leg:** 0.43% / -1.25%
- **Sum % (uncompounded):** 22.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 2 | 8.7% | 2 | 21 | 0 | -1.19% | -27.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 2 | 21 | 0 | -1.19% | -27.3% |
| SELL (all) | 29 | 13 | 44.8% | 5 | 17 | 7 | 1.71% | 49.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 13 | 44.8% | 5 | 17 | 7 | 1.71% | 49.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 15 | 28.8% | 7 | 38 | 7 | 0.43% | 22.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 10:15:00 | 4392.50 | 4210.83 | 4210.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 10:15:00 | 4403.95 | 4273.07 | 4247.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 12:15:00 | 4488.65 | 4518.57 | 4414.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 12:45:00 | 4481.30 | 4518.57 | 4414.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 4395.40 | 4512.39 | 4416.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 15:00:00 | 4395.40 | 4512.39 | 4416.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 15:15:00 | 4406.90 | 4511.34 | 4416.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 09:15:00 | 4430.05 | 4511.34 | 4416.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 12:00:00 | 4419.15 | 4509.03 | 4416.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 09:30:00 | 4434.10 | 4504.10 | 4416.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 15:15:00 | 4381.55 | 4498.12 | 4415.84 | SL hit (close<static) qty=1.00 sl=4385.10 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 09:15:00 | 5112.40 | 5663.48 | 5665.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 14:15:00 | 4956.85 | 5600.08 | 5633.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 5398.60 | 5390.41 | 5503.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 09:45:00 | 5404.10 | 5390.41 | 5503.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 5542.00 | 5394.37 | 5502.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 15:00:00 | 5542.00 | 5394.37 | 5502.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 5560.00 | 5396.02 | 5502.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:15:00 | 5520.80 | 5396.02 | 5502.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 5584.80 | 5399.12 | 5503.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:00:00 | 5584.80 | 5399.12 | 5503.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 5497.10 | 5411.68 | 5487.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:00:00 | 5497.10 | 5411.68 | 5487.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 5470.00 | 5412.26 | 5487.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 5349.90 | 5413.49 | 5487.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 5577.20 | 5415.12 | 5487.65 | SL hit (close>static) qty=1.00 sl=5508.85 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 6079.60 | 5455.67 | 5453.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 6117.35 | 5480.38 | 5466.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 6605.30 | 6650.11 | 6366.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 6605.30 | 6650.11 | 6366.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 6605.30 | 6650.11 | 6366.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 6756.95 | 6389.20 | 6365.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 13:45:00 | 6675.05 | 6406.40 | 6375.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 6756.65 | 6410.84 | 6378.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 10:00:00 | 6705.60 | 6413.77 | 6379.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 6390.00 | 6537.38 | 6466.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 6390.00 | 6537.38 | 6466.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 6380.00 | 6535.81 | 6466.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 6370.05 | 6535.81 | 6466.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 6300.00 | 6533.46 | 6465.70 | SL hit (close<static) qty=1.00 sl=6350.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 5650.00 | 6444.00 | 6447.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 12:15:00 | 5638.40 | 6435.98 | 6443.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 6119.00 | 6110.68 | 6250.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:30:00 | 6124.55 | 6110.68 | 6250.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 6291.00 | 6107.09 | 6223.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:30:00 | 6154.90 | 6127.89 | 6225.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 12:15:00 | 6179.05 | 6128.78 | 6225.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 14:00:00 | 6172.65 | 6129.86 | 6224.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 6164.10 | 6138.05 | 6224.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 6215.00 | 6138.82 | 6224.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:15:00 | 6191.35 | 6138.82 | 6224.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 6169.85 | 6139.13 | 6224.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:45:00 | 6140.00 | 6138.99 | 6224.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 5870.10 | 6132.66 | 6218.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 5864.02 | 6132.66 | 6218.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:15:00 | 5847.15 | 6129.77 | 6216.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:15:00 | 5855.90 | 6129.77 | 6216.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:15:00 | 5833.00 | 6129.77 | 6216.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-19 09:15:00 | 5561.15 | 6108.08 | 6203.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 5716.00 | 4939.39 | 4936.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 13:15:00 | 5745.50 | 4947.41 | 4940.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 5994.50 | 6198.11 | 5943.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:30:00 | 6219.50 | 6198.11 | 5943.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 5946.50 | 6195.61 | 5943.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 5936.00 | 6195.61 | 5943.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5695.00 | 6190.63 | 5942.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 5695.00 | 6190.63 | 5942.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 5637.00 | 6185.12 | 5940.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 5637.00 | 6185.12 | 5940.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 5273.00 | 5793.40 | 5795.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 5266.00 | 5788.16 | 5792.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 15:15:00 | 5110.00 | 5108.62 | 5281.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 5091.50 | 5108.62 | 5281.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4519.60 | 4441.85 | 4524.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 4524.00 | 4441.85 | 4524.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 4561.80 | 4443.05 | 4524.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 4555.10 | 4443.05 | 4524.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 4538.80 | 4444.00 | 4524.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 4574.70 | 4444.00 | 4524.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 15:15:00 | 4779.80 | 4584.44 | 4583.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 12:15:00 | 4800.10 | 4591.68 | 4587.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4623.50 | 4633.62 | 4611.71 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 4463.00 | 4594.67 | 4594.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 4453.80 | 4593.27 | 4594.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 4584.80 | 4559.80 | 4575.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 4584.80 | 4559.80 | 4575.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 4686.00 | 4561.05 | 4576.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 4686.00 | 4561.05 | 4576.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4535.00 | 4559.14 | 4574.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 4571.70 | 4559.14 | 4574.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 4594.00 | 4558.61 | 4574.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 4594.00 | 4558.61 | 4574.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 4568.00 | 4558.71 | 4573.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 4541.60 | 4558.71 | 4573.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 4608.00 | 4559.03 | 4574.00 | SL hit (close>static) qty=1.00 sl=4598.80 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 4777.30 | 4584.57 | 4583.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 4818.00 | 4590.77 | 4586.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 4698.80 | 4727.93 | 4670.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 4698.80 | 4727.93 | 4670.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 4665.70 | 4726.42 | 4670.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 4665.70 | 4726.42 | 4670.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 4670.10 | 4725.86 | 4670.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 4699.40 | 4725.38 | 4670.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 4688.00 | 4724.10 | 4670.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 4685.00 | 4723.27 | 4670.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 12:30:00 | 4704.60 | 4722.00 | 4671.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 4614.90 | 4720.94 | 4671.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 4614.90 | 4720.94 | 4671.20 | SL hit (close<static) qty=1.00 sl=4659.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-07-27 09:15:00 | 4430.05 | 2023-07-28 15:15:00 | 4381.55 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-07-27 12:00:00 | 4419.15 | 2023-07-28 15:15:00 | 4381.55 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-07-28 09:30:00 | 4434.10 | 2023-07-28 15:15:00 | 4381.55 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2023-07-31 09:30:00 | 4424.00 | 2023-08-08 10:15:00 | 4866.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-23 11:15:00 | 5068.25 | 2023-10-23 15:15:00 | 5006.45 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-11-03 12:45:00 | 5208.15 | 2023-12-26 10:15:00 | 5728.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-13 15:15:00 | 5135.00 | 2024-03-14 09:15:00 | 5112.40 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-04-15 09:15:00 | 5349.90 | 2024-04-15 09:15:00 | 5577.20 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-04-16 09:30:00 | 5449.15 | 2024-04-18 12:15:00 | 5518.95 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-04-16 10:30:00 | 5454.00 | 2024-04-18 12:15:00 | 5518.95 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-04-16 11:45:00 | 5450.00 | 2024-04-18 12:15:00 | 5518.95 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-04-29 13:15:00 | 5475.15 | 2024-04-30 11:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-04-29 15:15:00 | 5411.90 | 2024-04-30 11:15:00 | 5520.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-05-02 10:45:00 | 5468.25 | 2024-05-03 09:15:00 | 5580.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-05-03 10:30:00 | 5484.80 | 2024-05-13 09:15:00 | 5210.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 10:30:00 | 5484.80 | 2024-05-15 10:15:00 | 5427.00 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2024-05-06 15:15:00 | 5403.00 | 2024-05-17 11:15:00 | 5575.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-05-08 13:30:00 | 5423.95 | 2024-05-17 11:15:00 | 5575.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-05-09 09:45:00 | 5401.45 | 2024-05-17 11:15:00 | 5575.00 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-05-15 14:15:00 | 5403.30 | 2024-05-17 11:15:00 | 5575.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-05-16 13:45:00 | 5408.55 | 2024-05-21 10:15:00 | 5658.75 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2024-05-24 09:15:00 | 5315.00 | 2024-05-29 14:15:00 | 5049.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 09:15:00 | 5315.00 | 2024-06-06 09:15:00 | 5372.80 | STOP_HIT | 0.50 | -1.09% |
| BUY | retest2 | 2024-10-03 09:15:00 | 6756.95 | 2024-10-23 09:15:00 | 6300.00 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2024-10-04 13:45:00 | 6675.05 | 2024-10-23 09:15:00 | 6300.00 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest2 | 2024-10-07 09:15:00 | 6756.65 | 2024-10-23 09:15:00 | 6300.00 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2024-10-07 10:00:00 | 6705.60 | 2024-10-23 09:15:00 | 6300.00 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest2 | 2024-10-29 09:15:00 | 6486.30 | 2024-10-29 09:15:00 | 6430.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-10-29 11:00:00 | 6500.00 | 2024-10-30 11:15:00 | 6424.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-10-30 11:30:00 | 6496.70 | 2024-10-30 12:15:00 | 6397.60 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-10-31 09:45:00 | 6535.80 | 2024-11-11 12:15:00 | 6433.50 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-12-13 10:30:00 | 6154.90 | 2024-12-18 09:15:00 | 5870.10 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2024-12-13 12:15:00 | 6179.05 | 2024-12-18 09:15:00 | 5864.02 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-12-13 14:00:00 | 6172.65 | 2024-12-18 10:15:00 | 5847.15 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2024-12-17 09:15:00 | 6164.10 | 2024-12-18 10:15:00 | 5855.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 6140.00 | 2024-12-18 10:15:00 | 5833.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 10:30:00 | 6154.90 | 2024-12-19 09:15:00 | 5561.15 | TARGET_HIT | 0.50 | 9.65% |
| SELL | retest2 | 2024-12-13 12:15:00 | 6179.05 | 2024-12-30 13:15:00 | 5555.39 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2024-12-13 14:00:00 | 6172.65 | 2024-12-30 14:15:00 | 5539.41 | TARGET_HIT | 0.50 | 10.26% |
| SELL | retest2 | 2024-12-17 09:15:00 | 6164.10 | 2024-12-30 14:15:00 | 5547.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 6140.00 | 2024-12-30 14:15:00 | 5526.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 15:15:00 | 4541.60 | 2026-03-25 09:15:00 | 4608.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-25 12:30:00 | 4540.00 | 2026-03-30 14:15:00 | 4641.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-03-30 14:15:00 | 4557.90 | 2026-03-30 14:15:00 | 4641.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-04-27 09:15:00 | 4699.40 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-04-27 12:30:00 | 4688.00 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-04-27 15:15:00 | 4685.00 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-28 12:30:00 | 4704.60 | 2026-04-28 13:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-05-06 09:15:00 | 4685.80 | 2026-05-06 13:15:00 | 4627.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-05-07 09:15:00 | 4670.00 | 2026-05-08 10:15:00 | 4603.70 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-05-07 09:45:00 | 4668.90 | 2026-05-08 10:15:00 | 4603.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-05-07 13:45:00 | 4669.10 | 2026-05-08 10:15:00 | 4603.70 | STOP_HIT | 1.00 | -1.40% |
