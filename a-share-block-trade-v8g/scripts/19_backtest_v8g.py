"""
v8g — 大宗交易溢价因子策略最终版
===========================================
选股: 溢价>=5% + 市值<100亿
仓位: LLT动态仓位 (BULL=5%, NEUTRAL=25%, BEAR=35%)
持有: 30天, 开盘买/开盘卖
过滤: BULL状态下价格高于MA5/MA10则跳过

回测结果 (2025~2026):
  - 总收益: +226.86%
  - 最大回撤: -11.96%
  - 胜率: 69.6%
  - 交易数: 56笔
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    DATA_DIR, OUTPUT_DIR, INITIAL_CAPITAL,
    PREMIUM_THRESHOLD, MAX_MARKET_CAP,
    COMMISSION_RATE, STAMP_TAX_RATE, MIN_COMMISSION,
    POSITION_MAP, BULL_MA_FILTER, LLT_ALPHA, HOLD_DAYS
)

# ===== LLT计算 =====
def compute_llt(series, alpha=LLT_ALPHA):
    """低延迟趋势线 (Low-Lag Trendline)"""
    n = len(series)
    llt = np.zeros(n)
    if n < 3:
        return llt
    llt[0] = series[0]
    llt[1] = series[1]
    for i in range(2, n):
        llt[i] = (2 - alpha) * llt[i-1] - (1 - alpha) * llt[i-2] + alpha * (series[i] - series[i-1])
    return llt


def llt_state(price, llt_val, slope):
    """判断LLT状态: BULL / BEAR / NEUTRAL"""
    if price > llt_val and slope > 0:
        return 'BULL'
    elif price <= llt_val and slope <= 0:
        return 'BEAR'
    else:
        return 'NEUTRAL'


# ===== 1. 加载数据 =====
print("回测 v8g (LLT动态仓位)")
print("BULL=5% NEUTRAL=25% BEAR=35% | BULL>MA不买")

bt = pd.read_csv(os.path.join(DATA_DIR, "block_trades.csv"))
bt['trade_dt'] = pd.to_datetime(bt['trade_dt'], format='%Y%m%d')
bt = bt[bt['trade_dt'].dt.year >= 2025]

dp = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                 usecols=['windcode', 'trade_dt', 'close', 'open', 'adj_close'])
dp['trade_dt'] = pd.to_datetime(dp['trade_dt'], format='%Y%m%d')
dp = dp.sort_values(['windcode', 'trade_dt']).reset_index(drop=True)

indi = pd.read_csv(os.path.join(DATA_DIR, "derivative_indicators.csv"),
                   usecols=['windcode', 'trade_dt', 'mkt_cap'])
indi['trade_dt'] = pd.to_datetime(indi['trade_dt'], format='%Y%m%d')

# ===== 2. 信号生成 =====
bt2 = bt.merge(dp[['windcode', 'trade_dt', 'close']], on=['windcode', 'trade_dt'], how='left')
bt2['premium'] = (bt2['block_price'] - bt2['close']) / bt2['close']
bt2 = bt2.dropna(subset=['premium'])
bt2 = bt2.sort_values(['windcode', 'trade_dt', 'premium']).drop_duplicates(
    subset=['windcode', 'trade_dt'], keep='last')
bt2 = bt2.merge(indi[['windcode', 'trade_dt', 'mkt_cap']], on=['windcode', 'trade_dt'], how='left')
bt2['mkt_cap_亿'] = bt2['mkt_cap'] / 10000
for wc in bt2[bt2['mkt_cap_亿'].isna()]['windcode'].unique():
    vals = indi[indi['windcode'] == wc]['mkt_cap'].dropna()
    if len(vals) > 0:
        bt2.loc[bt2['windcode'] == wc, 'mkt_cap_亿'] = vals.mean() / 10000

# 筛选: 溢价>=5% + 市值<100亿
sig = bt2[(bt2['premium'] > PREMIUM_THRESHOLD) & (bt2['mkt_cap_亿'] < MAX_MARKET_CAP)].copy()
sig = sig.sort_values(['trade_dt', 'windcode']).reset_index(drop=True)
sig['ds'] = sig['trade_dt'].dt.strftime('%Y-%m-%d')
print(f"信号: {len(sig)} 条")

# 信号按日期索引
sbd = {}
for _, r in sig.iterrows():
    d = r['ds']
    if d not in sbd:
        sbd[d] = []
    sbd[d].append(r)

# ===== 3. LLT计算 =====
print("计算LLT...")
nw = set(sig['windcode'].unique())
llt_cache = {}

for wc in nw:
    s = dp[dp['windcode'] == wc].copy().reset_index(drop=True)
    if len(s) < 30:
        continue
    prices = s['adj_close'].values
    llt_v = compute_llt(prices)
    s['llt'] = llt_v
    s['llt_slope'] = np.gradient(llt_v)
    states = []
    for i in range(len(s)):
        if i < 2:
            states.append('NEUTRAL')
        else:
            states.append(llt_state(prices[i], llt_v[i], s['llt_slope'].iloc[i]))
    s['llt_state'] = states
    s['ma5'] = s['adj_close'].rolling(5).mean()
    s['ma10'] = s['adj_close'].rolling(10).mean()
    s['ds'] = s['trade_dt'].dt.strftime('%Y-%m-%d')
    llt_cache[wc] = s

print(f"  LLT完成: {len(llt_cache)} 只")

# ===== 4. 辅助函数 =====
dpi = {}
for wc in nw:
    s = dp[dp['windcode'] == wc].copy().reset_index(drop=True)
    s['ds'] = s['trade_dt'].dt.strftime('%Y-%m-%d')
    if len(s) > 0:
        dpi[wc] = s

ads = sorted(dp[dp['trade_dt'].dt.year >= 2025]['trade_dt'].dt.strftime('%Y-%m-%d').unique())


def gop(wc, ds):
    """获取开盘价"""
    if wc not in dpi:
        return None
    m = dpi[wc][dpi[wc]['ds'] == ds]
    return float(m.iloc[0]['open']) if len(m) > 0 else None


def nxt(wc, ds, n=1):
    """获取未来第n天的日期和开盘价"""
    if wc not in dpi:
        return None, None
    p = dpi[wc]
    m = p[p['ds'] == ds]
    si = m.index[0] if len(m) > 0 else (
        p[p['trade_dt'] > pd.to_datetime(ds)].index[0] if len(p[p['trade_dt'] > pd.to_datetime(ds)]) > 0 else -1)
    if si < 0:
        return None, None
    ti = si + n
    if ti >= len(p):
        return None, None
    return p.iloc[ti]['ds'], float(p.iloc[ti]['open'])


def get_llt_state(wc, ds):
    """获取LLT状态"""
    if wc not in llt_cache:
        return 'NEUTRAL'
    m = llt_cache[wc][llt_cache[wc]['ds'] == ds]
    return m.iloc[0]['llt_state'] if len(m) > 0 else 'NEUTRAL'


# ===== 5. 回测 =====
cash = INITIAL_CAPITAL
pos = {}
trades = []
daily = []

for cd in ads:
    # 到期卖出
    rem = []
    for wc, p in list(pos.items()):
        if p['xd'] == cd:
            ep = gop(wc, cd)
            if ep and ep > 0:
                amt = p['sh'] * ep
                fee = max(amt * COMMISSION_RATE, MIN_COMMISSION) + amt * STAMP_TAX_RATE
                net = amt - fee
                cash += net
                trades.append({
                    'code': wc, 'buy_dt': p['ed'], 'sell_dt': cd,
                    'buy': p['epx'], 'sell': ep, 'shares': p['sh'],
                    'invested': p['cost'], 'ror': ep / p['epx'] - 1,
                    'profit': net - p['cost'], 'llt_state': p.get('state', ''),
                })
            rem.append(wc)
    for c in rem:
        if c in pos:
            del pos[c]

    # 新信号
    if cd in sbd:
        ts = []
        for s in sbd[cd]:
            wc = s['windcode']
            if wc in pos:
                continue
            ed, opx = nxt(wc, cd)
            if ed is None:
                continue
            state = get_llt_state(wc, cd)

            # BULL过滤: 价格>MA5/MA10时跳过
            if BULL_MA_FILTER and state == 'BULL':
                if wc in llt_cache:
                    m = llt_cache[wc][llt_cache[wc]['ds'] == cd]
                    if len(m) > 0:
                        price = m.iloc[0]['adj_close']
                        ma5 = m.iloc[0]['ma5']
                        ma10 = m.iloc[0]['ma10']
                        if not pd.isna(ma5) and not pd.isna(ma10):
                            if price > ma5 or price > ma10:
                                continue

            mp = POSITION_MAP.get(state, 0.25)
            ts.append({'wc': wc, 'ed': ed, 'opx': opx, 'prem': s['premium'],
                       'state': state, 'mp': mp})

        ts.sort(key=lambda x: x['prem'], reverse=True)

        for s in ts:
            wc = s['wc']
            state = s['state']
            mp = s['mp']
            if wc in pos:
                continue

            tv = cash
            for pw, pp in pos.items():
                px = gop(pw, cd)
                tv += pp['sh'] * px if px else pp['cost']

            target = tv * mp
            epx = s['opx']
            sh = int(target / (epx * 100)) * 100
            if sh <= 0:
                continue

            inv = sh * epx
            fee = max(inv * COMMISSION_RATE, MIN_COMMISSION)
            total = inv + fee

            if total > cash:
                sh2 = int((cash - fee) / (epx * 100)) * 100
                if sh2 <= 0:
                    continue
                sh = sh2
                inv = sh * epx
                fee = max(inv * COMMISSION_RATE, MIN_COMMISSION)
                total = inv + fee
                if total > cash:
                    continue

            xd, _ = nxt(wc, s['ed'], HOLD_DAYS)
            if xd is None:
                continue

            cash -= total
            pos[wc] = {'ed': s['ed'], 'epx': epx, 'sh': sh, 'xd': xd,
                       'cost': total, 'state': state}

    # 日净值
    tv = cash
    for wc, p in pos.items():
        px = gop(wc, cd)
        tv += p['sh'] * px if px else p['cost']
    daily.append({'date': cd, 'cash': cash, 'npos': len(pos), 'value': tv})

# 清算
ld = ads[-1]
for wc, p in list(pos.items()):
    ep = gop(wc, ld)
    if ep and ep > 0:
        amt = p['sh'] * ep
        fee = max(amt * COMMISSION_RATE, MIN_COMMISSION) + amt * STAMP_TAX_RATE
        net = amt - fee
        cash += net
        trades.append({
            'code': wc, 'buy_dt': p['ed'], 'sell_dt': ld,
            'buy': p['epx'], 'sell': ep, 'shares': p['sh'],
            'invested': p['cost'], 'ror': ep / p['epx'] - 1,
            'profit': net - p['cost'], 'llt_state': p.get('state', ''),
        })

# ===== 6. 结果 =====
df_t = pd.DataFrame(trades).sort_values('buy_dt').reset_index(drop=True)
df_d = pd.DataFrame(daily)
fv = df_d.iloc[-1]['value']
df_d['pk'] = df_d['value'].cummax()
df_d['dd'] = df_d['value'] / df_d['pk'] - 1

print(f"\n{'=' * 60}")
print(f"结果 (v8g — LLT动态仓位)")
print(f"{'=' * 60}")
print(f"  BULL→5% | NEUTRAL→25% | BEAR→35%")
print(f"  初始:     {INITIAL_CAPITAL:>10,}")
print(f"  最终:     {fv:>10,.0f}")
print(f"  总收益:   {fv / INITIAL_CAPITAL - 1:>+10.2%}")
print(f"  年化:     {(fv / INITIAL_CAPITAL) ** (252 / len(df_d)) - 1:>+10.2%}")
print(f"  最大回撤: {df_d['dd'].min():>10.2%}")
print(f"  交易数:   {len(df_t):>10d}")
print(f"  胜率:     {(df_t['ror'] > 0).mean():>10.1%}")
print(f"  总利润:   {df_t['profit'].sum():>+10,.0f}")

print(f"\n  按LLT状态统计:")
for state in ['BULL', 'NEUTRAL', 'BEAR']:
    g = df_t[df_t['llt_state'] == state]
    if len(g) == 0:
        continue
    print(f"    {state:>8s}: {len(g):>3d}笔  "
          f"平均{g['ror'].mean():+.2%}  "
          f"胜率{(g['ror'] > 0).mean():.0%}  "
          f"总利润{g['profit'].sum():>+,.0f}")

# 保存
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_t.to_csv(os.path.join(OUTPUT_DIR, "v8g_trades.csv"), index=False)
df_d.to_csv(os.path.join(OUTPUT_DIR, "v8g_daily_nav.csv"), index=False)
print(f"\n结果已保存到: {OUTPUT_DIR}")
print("完成!")