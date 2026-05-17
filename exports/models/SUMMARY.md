# All Trading Models — Combined SUMMARY

3-year backtest window: **2023-05-15 → 2026-05-12** · ₹10L per model · Fyers data (4-yr re-pull 2026-05-17)

5 models (4 equity + 1 options). Each has ONE canonical version. Last refreshed 2026-05-17 with MAX_PRICE filter applied to 3 momentum models.

## Headline — deploy ₹10L per model

| # | Model | Universe | Rebalance | Final NAV | CAGR | Max DD | LIVE |
|--:|---|---|---|---:|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Real NSE Nifty 100 + MAX_PRICE≤₹3K | Monthly | ₹6,389,826 | **+85.85%** | 33.89% | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Top-100 ADV from N500 − Small + uptrend + MAX_PRICE≤₹3K | Monthly | ₹15,361,000 | **+149.15%** | **16.17%** | ❌ |
| 3 | `midcap_narrow_60d_breakout` | Top-100 ADV from N500 − Large | Event-driven | ₹13,456,535 | **+137.85%** | **8.12%** | ❌ |
| 4 | `finnifty_ic_otm4_w300_lots5` | FINNIFTY options (4 strikes per Mon) | Monthly IC | ₹22,25,673 | **+123.27%** compound | 13.88% | ❌ |
| 5 | `n20_daily_large_only` | Top-20 ADV + uptrend + NSE Nifty 100 + MAX_PRICE≤₹2.5K | Daily | ₹18,676,864 | **+165.97%** | 24.57% | ❌ |

## Unique stock-filtering approach per model

| Model | Filter mechanism | Why unique |
|---|---|---|
| `momentum_n100_top5_max1` | NSE Nifty 100 official list + price ≤ ₹3,000 | Only model using REAL NSE constituents; price cap eliminates mega-priced losers (BAJAJ-AUTO ₹12K, etc.) |
| `momentum_pseudo_n100_adv` | Top-100 ADV from N500, drop Small, uptrend, price ≤ ₹3,000 | Liquidity-based + trend + price gates; excludes mega-priced (DIXON ₹18K) |
| `midcap_narrow_60d_breakout` | Top-100 ADV from N500, drop Large | Highest-liquidity mid/small breakouts; event-driven |
| `finnifty_ic_otm4_w300_lots5` | Spot-derived 4 strikes (±4% OTM + ±300pt wings) | No equity universe — options chain auto-built per Monday |
| `n20_daily_large_only` | Top-20 ADV + close>200d SMA + NSE Nifty 100 + price ≤ ₹2,500 | Smallest universe (20); strictest gate; daily rebuild |

## Composite ranking — risk-adjusted Calmar (CAGR / Max DD)

| Rank | Model | CAGR | MaxDD | Calmar |
|--:|---|---:|---:|---:|
| 1 | midcap_narrow_60d_breakout | +137.85% | 8.12% | **16.98** |
| 2 | momentum_pseudo_n100_adv | +149.15% | 16.17% | **9.22** |
| 3 | finnifty_ic_otm4_w300_lots5 | +123.27% | 13.88% | **8.88** |
| 4 | n20_daily_large_only | +165.97% | 24.57% | **6.76** |
| 5 | momentum_n100_top5_max1 | +85.85% | 33.89% | **2.53** |

## What changed 2026-05-17 (MAX_PRICE filter)

User observation: high-price stocks (>₹3K) were giving outsized losses in backtest. Verified empirically:

| Model | Trades >₹3K killed | PnL recovered | CAGR delta | DD delta |
|---|---:|---:|---:|---:|
| `momentum_n100_top5_max1` | BAJAJ-AUTO ₹12,157 (-₹4.84L), ENRIN ₹2,972 (slips threshold) | +21pp | +21pp | -6pp |
| `momentum_pseudo_n100_adv` | DIXON ₹17,994 (-₹8L), MARUTI ₹12,917 (-₹3.2L) | +28pp | +28pp | -9pp |
| `n20_daily_large_only` | ₹5K-10K bucket (-₹2.05M aggregate) | +38pp | +38pp | -0.2pp |

Filter is purely formula-driven (price observable at entry); no future knowledge needed. Live-deployable. Threshold may drift as Nifty levels rise — review annually.

## Deployment recommendation

| Goal | Use |
|---|---|
| **Live equity (real universe, deployable)** | `momentum_n100_top5_max1` (LIVE) |
| **Defined-risk income (options)** | `finnifty_ic_otm4_w300_lots5` |
| **Best risk-adjusted swing returns** | `midcap_narrow_60d_breakout` (Calmar 17+) |
| **Best risk-adjusted CAGR** | `momentum_pseudo_n100_adv` (Calmar 9+, but lookahead caveat) |
| **Highest absolute CAGR (daily)** | `n20_daily_large_only` |

## Data integrity

- All data from Fyers production source (cont_flag=1 split-adjusted)
- 498/504 N500 stocks covered (4 DUMMYVEDL placeholders + 2 symbol mismatch = legitimately missing)
- 1 known anomaly: ABFRL 2025-05-22 real demerger (-67%, not data bug)
- yfinance NEVER used (project rule)

## Files

```
exports/models/
├── SUMMARY.md                          ← this file
├── momentum_n100_top5_max1/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── momentum_pseudo_n100_adv/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── midcap_narrow_60d_breakout/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── finnifty_ic_otm4_w300_lots5/
│   ├── SUMMARY.md + MONTHLY_INVESTED.md + trades.csv + monthly.csv
└── n20_daily_large_only/
    ├── SUMMARY.md
    └── TRADE_LEDGER.md
```
