# Tata Consultancy Services Ltd. (TCS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2474.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -26.24
- **Avg P&L per closed trade:** -4.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 4092.20 | 4276.31 | 4277.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 4088.45 | 4247.99 | 4262.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 4139.00 | 4133.48 | 4189.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 4101.35 | 4133.32 | 4188.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4101.35 | 4133.32 | 4188.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-11 09:15:00 | 4199.75 | 4134.59 | 4185.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 4352.55 | 4203.89 | 4203.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.03 | 4204.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4308.99 | 4265.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 11:15:00 | 4329.85 | 4309.20 | 4265.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4284.60 | 4310.45 | 4267.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 10:15:00 | 4266.50 | 4308.87 | 4268.16 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 4133.40 | 4237.89 | 4238.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 4096.85 | 4222.73 | 4230.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-17 09:15:00 | 4141.15 | 4203.44 | 4215.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-12 12:15:00 | 3602.10 | 3452.61 | 3563.02 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3097.87 | 3090.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3168.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-01 10:15:00 | 3221.00 | 3212.38 | 3168.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3192.10 | 3219.83 | 3182.70 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-12 12:15:00 | 3213.20 | 3219.57 | 3182.94 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 3191.00 | 3221.24 | 3186.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-16 09:15:00 | 3208.50 | 3220.88 | 3186.81 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.90 | 3167.66 | 3168.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2933.90 | 3160.19 | 3164.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.20 | 2525.99 | 2688.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-10 09:15:00 | 2511.00 | 2531.89 | 2677.14 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 09:15:00 | 4101.35 | 2024-11-11 09:15:00 | 4199.75 | EXIT_EMA400 | -98.40 |
| BUY | 2024-12-18 11:15:00 | 4329.85 | 2024-12-20 10:15:00 | 4266.50 | EXIT_EMA400 | -63.35 |
| SELL | 2025-01-17 09:15:00 | 4141.15 | 2025-02-13 14:15:00 | 3917.94 | TARGET | 223.21 |
| BUY | 2026-01-01 10:15:00 | 3221.00 | 2026-01-19 09:15:00 | 3185.00 | EXIT_EMA400 | -36.00 |
| BUY | 2026-01-12 12:15:00 | 3213.20 | 2026-01-19 09:15:00 | 3185.00 | EXIT_EMA400 | -28.20 |
| BUY | 2026-01-16 09:15:00 | 3208.50 | 2026-01-19 09:15:00 | 3185.00 | EXIT_EMA400 | -23.50 |
