# Dixon Technologies (India) Ltd. (DIXON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 11220.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -2107.05
- **Avg P&L per closed trade:** -351.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 14761.15 | 16196.04 | 16199.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 14392.10 | 16123.31 | 16162.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 14281.60 | 14085.06 | 14652.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 09:15:00 | 13989.95 | 14114.01 | 14648.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 13989.95 | 14114.01 | 14648.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 13892.00 | 14111.80 | 14644.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 14170.00 | 13620.07 | 14186.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-11 11:15:00 | 14273.00 | 13626.57 | 14186.93 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 16335.00 | 14594.43 | 14587.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 12:15:00 | 16501.00 | 14613.40 | 14596.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 15169.00 | 15369.64 | 15040.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 16136.00 | 15365.48 | 15049.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 15690.00 | 15767.67 | 15340.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-22 09:15:00 | 15211.00 | 15753.93 | 15348.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 14490.00 | 15124.03 | 15125.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 14326.00 | 15116.09 | 15121.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 14823.00 | 14705.96 | 14879.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 12:15:00 | 14397.00 | 14694.43 | 14859.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 14796.00 | 14693.73 | 14855.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-02 11:15:00 | 14915.00 | 14696.56 | 14855.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 15716.00 | 14981.43 | 14977.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 15900.00 | 15007.08 | 14990.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 16130.00 | 16182.44 | 15764.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-18 09:15:00 | 16799.00 | 16144.30 | 15803.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 09:15:00 | 16890.00 | 17642.55 | 17052.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 15517.00 | 16809.61 | 16813.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 15456.00 | 16770.56 | 16793.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 11446.00 | 11356.58 | 12477.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-23 09:15:00 | 10854.00 | 11432.96 | 12119.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 10814.50 | 10433.76 | 10893.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 11:15:00 | 10945.00 | 10438.84 | 10894.24 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-25 09:15:00 | 13989.95 | 2025-04-11 11:15:00 | 14273.00 | EXIT_EMA400 | -283.05 |
| SELL | 2025-03-25 10:15:00 | 13892.00 | 2025-04-11 11:15:00 | 14273.00 | EXIT_EMA400 | -381.00 |
| BUY | 2025-05-12 09:15:00 | 16136.00 | 2025-05-22 09:15:00 | 15211.00 | EXIT_EMA400 | -925.00 |
| SELL | 2025-07-01 12:15:00 | 14397.00 | 2025-07-02 11:15:00 | 14915.00 | EXIT_EMA400 | -518.00 |
| BUY | 2025-08-18 09:15:00 | 16799.00 | 2025-09-29 09:15:00 | 16890.00 | EXIT_EMA400 | 91.00 |
| SELL | 2026-02-23 09:15:00 | 10854.00 | 2026-04-15 11:15:00 | 10945.00 | EXIT_EMA400 | -91.00 |
