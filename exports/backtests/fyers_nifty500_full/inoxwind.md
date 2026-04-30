# Inox Wind Ltd. (INOXWIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 100.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 25.36
- **Avg P&L per closed trade:** 12.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 185.47 | 208.89 | 208.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 179.90 | 207.30 | 208.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 204.32 | 198.84 | 203.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 192.71 | 200.53 | 202.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-01 09:15:00 | 179.09 | 163.46 | 175.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 173.10 | 164.10 | 164.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 174.70 | 164.21 | 164.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 178.96 | 178.98 | 174.04 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 162.60 | 172.19 | 172.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 162.24 | 172.09 | 172.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 147.54 | 147.12 | 153.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 13:15:00 | 145.25 | 148.73 | 152.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 150.67 | 144.87 | 149.02 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 09:15:00 | 192.71 | 2025-01-09 12:15:00 | 161.93 | TARGET | 30.78 |
| SELL | 2025-09-24 13:15:00 | 145.25 | 2025-10-10 09:15:00 | 150.67 | EXIT_EMA400 | -5.42 |
