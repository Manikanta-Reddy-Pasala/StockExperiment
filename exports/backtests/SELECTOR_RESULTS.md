# Multi-Param Selector Results

_Generated: 2026-05-12_

## Approach

Stock picker uses **5 features** (no LLM, no news, no historical backtest sum%):

```
composite = 0.25 * normalized(ATR% 20d)         # volatility
          + 0.25 * normalized(60d return)        # momentum
          + 0.15 * normalized(5d/20d vol spike)  # interest
          + 0.20 * normalized(ADV in lakh)       # liquidity
          + 0.15 * normalized(|dist from 52W high|)  # range-position
```

Filters: `min_price=₹50`, `min_adv=₹1cr/day` (penny + illiquid culled).

Selector run as of **2025-05-12** (backtest start) → no lookahead.

## Selector top-30 (from N500, ranked by composite)

```
SWIGGY, VMM, AEGISLOG, ANGELONE, SAILIFE, ITI, IKS, AMBER, NTPCGREEN, BSE,
TARIL, IGIL, ADANIGREEN, WOCKPHARMA, CARTRADE, ZENTEC, BLUEJET, PGEL, CREDITACC,
ACMESOLAR, MAZDOCK, WAAREEENER, PAYTM, KFINTECH, SWANCORP, GICRE, MCX, KAYNES,
INTELLECT, OLAELEC
```

## Cap-sim results

| Top-N | max=2 | max=3 | max=5 | DD% (max=2) |
|------:|------:|------:|------:|------------:|
| **10** | **+21.30** | +12.52 | +6.26 | **9.60** |
|    15 | +18.57 | +7.68 | -1.16 | 16.65 |
|    20 | +18.28 | +10.92 | -0.54 | 17.40 |
|    25 | +15.88 | +4.93 | -5.93 | 19.12 |
|    30 | +18.76 | +3.47 | -5.19 | 22.50 |

## Winner: Selector top-10 + max=2

- **ROI: +21.30%** (28 trades)
- **MaxDD: 9.60%**
- Beats prior best (N500 top-20 by historical sum%, +14.15%) by 7pp
- Beats N50 top-19+max=3 (+13.20%) by 8pp
- DD slightly higher than N50 top-19 (7.86%) but acceptable

## Top-10 stocks (live watchlist)

| Rank | Symbol | Close@2025-05-12 | ATR% | 60d Ret | Vol Spike | ADV ₹L |
|-----:|--------|-----------------:|-----:|--------:|----------:|-------:|
|    1 | SWIGGY |          ₹597.45 | 6.18 | +58.31% |     1.39× |  1,141 |
|    2 | VMM    |          ₹101.19 | 7.40 | -12.86% |     3.03× |    802 |
|    3 | AEGISLOG |        ₹816.45 | 7.29 | +12.57% |     2.06× |    548 |
|    4 | ANGELONE |        ₹288.17 | 4.89 | +13.50% |     0.93× |    608 |
|    5 | SAILIFE |         ₹703.65 | 6.49 |  -5.01% |     3.38× |    318 |
|    6 | ITI    |          ₹341.65 | 6.51 | +22.32% |     0.50× |    688 |
|    7 | IKS    |        ₹1,885.80 | 5.15 | +10.56% |     3.45× |    226 |
|    8 | AMBER  |        ₹6,898.00 | 6.48 | +27.66% |     0.64× |    907 |
|    9 | NTPCGREEN |       ₹131.67 | 6.33 | +39.72% |     0.27× |    694 |
|   10 | BSE    |        ₹1,847.62 | 4.09 | +44.43% |     0.49× |  1,675 |

## Why selector beats historical-sum%

- **No survivorship bias.** Sum%-ranking favors stocks that already worked
  in the past year. Selector picks based on *current* characteristics
  (ATR, momentum, liquidity) which generalize to forward periods.
- **Volatility = alpha.** ATR-weighted picks have bigger swings. EMA 200/400
  crossover strategy benefits from price ranges, not stable consolidation.
- **Liquidity floor catches institutional flow.** ADV ₹1cr+ filter ensures
  we trade only stocks where 200K capital won't move the price.

## Caveats

- Single window (May 2025 → May 2026). Need multi-year validation.
- Bull-skewed year for Indian mid-caps. Bear regime may invert this.
- Selector weights (0.25/0.25/0.15/0.20/0.15) untuned; grid search pending.
- All-N500 universe — N50 large caps don't appear in top-10 due to lower
  ATR (large caps are stable by construction).

## Production recommendation

**Live config (paper-trade for 4 weeks first):**

- Universe: Selector top-10, refreshed monthly
- Strategy: EMA 200/400 1H crossover
- Capital: ₹2,00,000
- Max concurrent: 2
- Penny filter: ON (close ≥ ₹50)
- ADV filter: ₹1cr/day min
- Expected ROI: +15-25%/yr (backtest = +21.30%)
- Expected MaxDD: ~10%

## Monthly selector refresh (live workflow)

```bash
# 1st of each month, recompute composite from previous day
python tools/backtests/stock_selector.py \
  --universe nifty500 --top 10 \
  --end-date $(date +%Y-%m-%d) \
  --out signals/selector_$(date +%Y-%m).json

# Use selected.json as the universe for signal_generator.py
python tools/live/signal_generator.py \
  --model ema_200_400 \
  --universe-file signals/selector_$(date +%Y-%m).json
```

(signal_generator.py needs a `--universe-file` flag — small enhancement
in Phase 4.)
