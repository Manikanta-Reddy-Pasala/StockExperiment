# n20_daily_large_only — SUMMARY

**Top-20 ADV + uptrend + Nifty 100. Daily rotation top-1 by 30d ret. No price filter — honest baseline.**

## When it BUYS (entry rules)

Single position (`max_concurrent=1`). **Every weekday** (no monthly gate):
1. Universe = top-**20** by 20-day ADV from N500, rebuilt daily (`UNIV_SIZE=20`).
2. **Uptrend gate** — close > 200-day SMA (`SMA_LONG=200`).
3. **Large-cap gate** — stock must be in NSE Nifty 100.
4. Rank survivors by **30-day return** (`LOOKBACK_RET=30`), buy **rank-1** (`--top-n 1`).
- Code: `build_pit_universe_and_rank()` + `emit_signals()` in `live_signal.py:99-191`.

## When it SELLS (exit rules)

Daily rotation, single position. **Sells only on rank rotation — NO price stop or target:**
- **Every weekday**: SELLS the held stock the moment it is **no longer rank-1** (top-1 by 30-day return within the top-20 ADV ∩ Nifty-100 uptrend pool), and buys the new rank-1. If the held stock is still rank-1, keeps it.
- SELL labelled `TARGET_HIT`/`STOP_HIT` by exit-vs-entry price only — the **trigger is losing the #1 rank, not a price level.** Highest churn of the four models.
- Live and backtest exit logic **agree** here (both top-1) — no live-vs-backtest mismatch.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-13 |
| Total trades | 138 |
| Trades per year | ~46.0 |
| Rebalance | Daily |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: top-20 by 20-day ADV from N500 (rebuilt daily)
2. Uptrend filter: close > 200-day SMA
3. Large-cap filter: stock must be in NSE Nifty 100
4. Rank by 30-day return, pick top-1
5. Rebalance: every trading day

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.13,655,640** |
| Total return | **+1265.56%** |
| 2.99-yr CAGR | **+139.55%** |
| Max DD | **25.66%** |
| Calmar (CAGR / Max DD) | **5.44** |
| Trades closed | 138 |
| Wins / Losses | 60 / 76 |
| Win rate | 44.1% |
| Live deployment | YES (live since 2026-05) |
| Open position | **ADANIPOWER** qty 65,141 entry Rs.181.35 (2026-04-13) last Rs.209.63 unrealized +1,842,187 |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 138 | 60 | 76 | 44% | +10,813,445 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MAZDOCK      | 2024-05-29 → 2024-08-02 | 1,678.68 | +51.77% | +3,345,243 |
| ETERNAL      | 2025-07-21 → 2025-08-26 | 271.70 | +17.00% | +1,664,540 |
| HAL          | 2024-05-10 → 2024-05-29 | 3,872.90 | +30.44% | +1,507,749 |
| BEL          | 2025-05-30 → 2025-07-02 | 384.60 | +11.01% | +1,286,466 |
| BAJFINANCE   | 2025-01-28 → 2025-03-19 | 760.66 | +14.79% | +1,260,000 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2025-09-22 → 2025-09-24 | 170.25 | -15.12% | -1,902,075 |
| TRENT        | 2025-07-02 → 2025-07-04 | 6,222.50 | -12.32% | -1,597,386 |
| ETERNAL      | 2024-10-17 → 2024-10-18 | 270.55 | -4.84% | -480,063 |
| MAZDOCK      | 2025-05-05 → 2025-05-06 | 3,099.50 | -4.09% | -439,201 |
| BEL          | 2026-01-30 → 2026-02-05 | 449.00 | -3.59% | -435,746 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
