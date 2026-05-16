# momentum_n500_top1

V1 momentum-rotation strategy applied to **full Nifty 500** universe (no ADV slicing, no real-index filter). **NOT recommended for production** — wider universe gets whipsawed by small-cap speculation pumps.

## Strategy

| Knob | Value |
|---|---|
| Universe | **Full Nifty 500** (`src/data/symbols/nifty500.csv`) |
| Signal | Rank by 30-day return |
| Position | Hold top-1 (`max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when stock drops out of rank-1 |

No ADV filter, no market-cap filter. Strategy can pick ANY of 500 NSE stocks each month.

## Backtest result (2023-05-15 → 2026-05-12, ₹10L start)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹9,55,718 | **-4.43%** |
| Y2 (2024-05 → 2025-05) | ₹13,08,629 | **+36.93%** |
| Y3 (2025-05 → 2026-05) | ₹10,66,606 | **-18.49%** |
| **3-yr CAGR** | | **+2.17%** |
| Total return | | **+6.66%** |

32 round-trips · 62.5% WR · Max DD 41.07%

Year 1 and 3 negative. Y2 carry from FORCEMOT/COCHINSHIP/PCBL streak briefly recovered, then Y3 wipeout.

## Why it fails

Wider universe = more small-cap **speculation pumps** that mean-revert hard. Top-1 by 30d return systematically picks the **most parabolic** name — exactly the names about to break.

### Major losses (all small/mid-cap, would not be in real N100):

| Symbol | Entry → Exit | PnL | Ret |
|---|---|---:|---:|
| ANGELONE | 2025-01-01 → 2025-03-03 | -₹5,34,252 | -34.45% |
| RPOWER | 2025-07-01 → 2025-08-01 | -₹4,73,142 | -28.06% |
| IFCI | 2024-02-01 → 2024-03-01 | -₹2,49,465 | -26.73% |
| SAMMAANCAP | 2025-11-03 → 2025-12-01 | -₹2,44,820 | -19.69% |
| OLAELEC | 2025-09-01 → 2025-10-01 | -₹1,57,821 | -10.61% |
| BSE | 2023-12-01 → 2024-01-01 | -₹1,37,252 | -11.83% |
| JWL | 2023-09-01 → 2023-10-03 | -₹1,25,727 | -12.93% |

### Winners that worked:

| Symbol | Entry → Exit | PnL | Ret |
|---|---|---:|---:|
| FORCEMOT | 2024-03-01 → 2024-05-02 | +₹2,71,905 | +40.03% |
| GABRIEL | 2025-08-01 → 2025-09-01 | +₹2,74,443 | +22.65% |
| BSE | 2025-05-02 → 2025-06-02 | +₹3,67,683 | +28.12% |
| ANANDRATHI | 2023-11-01 → 2023-12-01 | +₹2,55,684 | +28.29% |
| GODFRYPHLP | 2025-04-01 → 2025-05-02 | +₹1,99,653 | +18.01% |
| GABRIEL | various | various | wins |

Winners barely offset losers. Asymmetric tail risk in small-caps.

## Comparison vs other variants

| Universe | CAGR | Max DD | WR | Trades |
|---|---:|---:|---:|---:|
| Pseudo-N100 v1 (top-100 by ADV) | +136.39% | 16.15% | 86.7% | 30 |
| Real Nifty 100 (NSE CSV, LIVE) | +80.38% | 29.71% | 74.2% | 31 |
| **Full Nifty 500 (this)** | **+2.17%** | 41.07% | 62.5% | 32 |

**Insight**: Pseudo-N100's ADV filter was load-bearing. Cutting it (full N500) drops CAGR to ~0%. The ADV filter implicitly removed illiquid speculation pumps. Pseudo-N100 ≠ Nifty 100 but ≠ random N500 slice either — it was a "liquid mid-caps + large-caps" universe that happened to include 2023-2026 winners.

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone reproducer (full N500, lb=30, mc=1, monthly) |
| `trade_ledger.json` | 32 trades + open position |

## Reproduce

```bash
docker exec trading_system_app python tools/models/momentum_n500_top1/backtest.py
```

## Verdict

**Hypothesis falsified**: "use full N500 to catch more winners" doesn't work. Strategy needs filtering — either ADV (top-100 liquid, +136% lookahead) or real index membership (real N100, +80% honest). Without either, +2% CAGR is barely above cash.

Use `momentum_n100_top5_max1` (real N100) for production.
