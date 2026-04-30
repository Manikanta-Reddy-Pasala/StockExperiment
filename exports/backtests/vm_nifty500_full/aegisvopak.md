# Aegis Vopak Terminals Ltd. (AEGISVOPAK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-06-02 09:15:00 → 2026-04-30 15:30:00 (1574 bars)
- **Last close:** 189.62
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -34.81
- **Avg P&L per closed trade:** -11.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 237.05 | 247.62 | 247.67 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 248.00 | 247.65 | 247.65 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 246.77 | 247.65 | 247.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 244.15 | 247.61 | 247.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 14:15:00 | 249.27 | 246.51 | 247.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 09:15:00 | 243.15 | 246.42 | 246.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-29 09:15:00 | 246.52 | 245.24 | 246.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 266.15 | 247.34 | 247.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 283.55 | 248.17 | 247.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 260.95 | 270.37 | 262.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 14:15:00 | 272.30 | 267.97 | 262.54 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 11:15:00 | 263.40 | 268.15 | 263.46 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 242.60 | 261.49 | 261.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 240.45 | 254.48 | 257.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 229.00 | 228.22 | 239.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 208.20 | 224.43 | 234.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 230.08 | 221.23 | 230.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-23 13:15:00 | 230.74 | 221.59 | 230.56 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-09-24 09:15:00 | 243.15 | 2025-09-29 09:15:00 | 246.52 | EXIT_EMA400 | -3.37 |
| BUY | 2025-11-11 14:15:00 | 272.30 | 2025-11-18 11:15:00 | 263.40 | EXIT_EMA400 | -8.90 |
| SELL | 2026-02-13 09:15:00 | 208.20 | 2026-02-23 13:15:00 | 230.74 | EXIT_EMA400 | -22.54 |
