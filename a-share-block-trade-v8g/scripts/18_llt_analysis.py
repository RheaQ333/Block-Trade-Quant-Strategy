"""
LLT状态 × 溢价大宗交易 联合分析
LLT = (2-α) × LLT_prev - (1-α) × LLT_prev2 + α × (price - price_prev)
α=0.15
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, LLT_ALPHA


def compute_llt(series, alpha=LLT_ALPHA):
    n = len(series)
    llt = np.zeros(n)
    if n < 3:
        return llt
    llt[0] = series[0]
    llt[1] = series[1]
    for i in range(2, n):
        llt[i] = (2 - alpha) * llt[i-1] - (1 - alpha) * llt[i-2] + alpha * (series[i] - series[i-1])
    return llt


def classify_llt(price, llt_val, slope):
    if price > llt_val and slope > 0:
        return 'BULL'
    elif price <= llt_val and slope <= 0:
        return 'BEAR'
    else:
        return 'NEUTRAL'


# 1. 加载数据
print("1. 加载数据...")
bt = pd.read_csv(os.path.join(DATA_DIR, "block_trades.csv"))
bt['trade_dt'] = pd.to_datetime(bt['trade_dt'], format='%Y%m%d')
bt = bt[bt['trade_dt'].dt.year >= 2025]

dp = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                 usecols=['windcode', 'trade_dt', 'adj_close'])
dp['trade_dt'] = pd.to_datetime(dp['trade_dt'], format='%Y%m%d')
dp = dp.sort_values(['windcode', 'trade_dt']).reset_index(drop=True)

# 2. 计算LLT
print("2. 计算LLT...")
needed = set(bt['windcode'].unique())
llt_data = {}

for wc in needed:
    s = dp[dp['windcode'] == wc].copy().reset_index(drop=True)
    if len(s) < 30:
        continue
    prices = s['adj_close'].values
    llt_vals = compute_llt(prices, LLT_ALPHA)
    s['llt'] = llt_vals
    s['llt_slope'] = np.gradient(llt_vals)
    llt_data[wc] = s

print(f"  LLT完成: {len(llt_data)} 只")

# 3. 合并LLT状态
print("3. 合并LLT状态...")
bt['date_str'] = bt['trade_dt'].dt.strftime('%Y-%m-%d')
records = []
for _, r in bt.iterrows():
    wc = r['windcode']
    if wc not in llt_data:
        continue
    s = llt_data[wc]
    m = s[s['trade_dt'] == r['trade_dt']]
    if len(m) == 0:
        continue
    i = m.index[0]
    if i < 2:
        continue
    px = m.iloc[0]['adj_close']
    llv = m.iloc[0]['llt']
    slp = m.iloc[0]['llt_slope']
    state = classify_llt(px, llv, slp)
    records.append({
        'windcode': wc, 'trade_dt': r['trade_dt'], 'premium': r['block_price'],
        'block_amount': r['block_amount'], 'llt_state': state,
        'price': px, 'llt': llv, 'slope': slp,
    })

df = pd.DataFrame(records)
print(f"  合并完成: {len(df)} 条")

# 4. 统计
print(f"\n{'=' * 60}")
print("LLT状态分布")
print("=" * 60)
for state in ['BULL', 'NEUTRAL', 'BEAR']:
    g = df[df['llt_state'] == state]
    print(f"  {state:>8s}: {len(g):>5d} 笔 ({len(g) / len(df) * 100:.0f}%)")

print(f"\n{'=' * 60}")
print("各状态下的溢价率分布")
print("=" * 60)
for state in ['BULL', 'NEUTRAL', 'BEAR']:
    g = df[df['llt_state'] == state]
    if len(g) == 0:
        continue
    print(f"  {state:>8s}: 均值{g['premium'].mean():.1%}  "
          f"中位数{g['premium'].median():.1%}  "
          f"25%分位{g['premium'].quantile(0.25):.1%}  "
          f"75%分位{g['premium'].quantile(0.75):.1%}")