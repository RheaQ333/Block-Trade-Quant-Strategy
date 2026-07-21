"""
v8g 实时信号扫描 — 从Wind数据库查询近期大宗交易并筛选
依赖: Wind MySQL只读库 (通过 wind_query.py 工具查询)

用法:
  python scripts/22_v8g_live_signals.py

注意: 需要配置 Wind MySQL 连接 (见 wind-data skill)
"""
import json
import subprocess
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, LLT_ALPHA, BULL_MA_FILTER, PREMIUM_THRESHOLD, MAX_MARKET_CAP

# Wind查询脚本路径 (需根据实际环境配置)
WIND_QUERY = os.path.expanduser("~/.workbuddy/skills/wind-data/scripts/wind_query.py")
EXCLUDED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "stock_list", "excluded_stocks.csv")

# ===== 1. 加载排除名单 =====
excluded = pd.read_csv(EXCLUDED_PATH)
ex_codes = set(excluded['code'].astype(str).str.zfill(6).tolist())

# ===== 2. LLT计算 =====
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

# ===== 3. 加载本地历史数据 =====
print("加载本地历史数据...")
dp = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                 usecols=['windcode', 'trade_dt', 'adj_close', 'close', 'open'])
dp['trade_dt'] = pd.to_datetime(dp['trade_dt'], format='%Y%m%d')
dp = dp.sort_values(['windcode', 'trade_dt']).reset_index(drop=True)

indi = pd.read_csv(os.path.join(DATA_DIR, "derivative_indicators.csv"),
                   usecols=['windcode', 'trade_dt', 'mkt_cap', 'turnover'])
indi['trade_dt'] = pd.to_datetime(indi['trade_dt'], format='%Y%m%d')

# ===== 4. 从Wind查询近期大宗交易 =====
print("加载Wind近期大宗交易...")
BT_SQL = ("SELECT S_INFO_WINDCODE, TRADE_DT, S_BLOCK_PRICE, S_BLOCK_AMOUNT "
          "FROM ASHAREBLOCKTRADE "
          "WHERE TRADE_DT >= '20260701' ORDER BY TRADE_DT DESC, S_INFO_WINDCODE LIMIT 500")

result = subprocess.run(
    [sys.executable, WIND_QUERY, "sql", "--sql", BT_SQL, "--format", "json"],
    capture_output=True, text=True, timeout=30
)
bt_list = json.loads(result.stdout)
bt = pd.DataFrame([{
    'windcode': d['S_INFO_WINDCODE'],
    'trade_dt': pd.to_datetime(d['TRADE_DT'], format='%Y%m%d'),
    'block_price': float(d['S_BLOCK_PRICE']),
    'block_amount': float(d['S_BLOCK_AMOUNT']),
} for d in bt_list if isinstance(d, dict)])

bt['code_6d'] = bt['windcode'].str.extract(r'^(\d{6})')
bt = bt[~bt['code_6d'].isin(ex_codes)]
print(f"  排除后: {len(bt)} 条, {bt['windcode'].nunique()} 只")

# ===== 5. 合并收盘价 =====
bt['date_str'] = bt['trade_dt'].dt.strftime('%Y-%m-%d')
bt['trade_dt_str'] = bt['trade_dt'].dt.strftime('%Y%m%d')

close_prices = {}
dates_needed = sorted(bt['trade_dt_str'].unique())
for td in dates_needed:
    codes_for_date = bt[bt['trade_dt_str'] == td]['windcode'].tolist()
    if not codes_for_date:
        continue
    batch_size = max(1, len(codes_for_date) // 5)
    for batch_start in range(0, len(codes_for_date), batch_size):
        batch = codes_for_date[batch_start:batch_start + batch_size]
        result = subprocess.run(
            [sys.executable, WIND_QUERY, "daily", "--table", "ASHAREEODPRICES",
             "--trade-dt", td, "--fields", "S_INFO_WINDCODE,S_DQ_CLOSE",
             "--code"] + batch + ["--limit", str(len(batch) + 5)],
            capture_output=True, text=True, timeout=20
        )
        for line in result.stdout.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    close_prices[(parts[0], td)] = float(parts[1])
                except ValueError:
                    pass

bt2 = bt.copy()
bt2['close'] = bt2.apply(
    lambda r: close_prices.get((r['windcode'], r['trade_dt_str']), np.nan), axis=1)
bt2['premium'] = (bt2['block_price'] - bt2['close']) / bt2['close']
bt2 = bt2.dropna(subset=['premium'])
bt2 = bt2.sort_values(['windcode', 'trade_dt', 'premium']).drop_duplicates(
    subset=['windcode', 'trade_dt'], keep='last')

# ===== 6. 筛选: 溢价>=5% + 市值<100亿 =====
qualify = bt2[(bt2['premium'] > PREMIUM_THRESHOLD)].copy()
print(f"\n近期溢价>5%: {len(qualify)} 条")

if len(qualify) == 0:
    print("暂无符合条件的信号")
    sys.exit(0)

# ===== 7. LLT计算 + 信号筛选 =====
print(f"\n{'=' * 80}")
print("v8g 信号筛选结果")
print("=" * 80)

results = []
for _, row in qualify.iterrows():
    wc = row['windcode']
    tdate = row['trade_dt']
    ds = row['date_str']

    s = dp[dp['windcode'] == wc].copy().reset_index(drop=True)
    if len(s) < 30:
        continue
    prices = s['adj_close'].values
    llt = compute_llt(prices)
    ma5 = s['adj_close'].rolling(5).mean()
    ma10 = s['adj_close'].rolling(10).mean()
    slp = np.gradient(llt)

    m = s[s['trade_dt'] == tdate]
    if len(m) == 0:
        continue
    i = m.index[0]
    px = prices[i]
    llv = llt[i]
    sp = slp[i]

    state = 'NEUTRAL'
    if i >= 2:
        if px > llv and sp > 0:
            state = 'BULL'
        elif px <= llv and sp <= 0:
            state = 'BEAR'

    o5 = px > ma5.iloc[i] if not pd.isna(ma5.iloc[i]) else False
    o10 = px > ma10.iloc[i] if not pd.isna(ma10.iloc[i]) else False

    skip = (state == 'BULL') and (o5 or o10)
    pct = {'BULL': 5, 'NEUTRAL': 25, 'BEAR': 35}.get(state, 25)

    results.append({
        'trade_dt': ds, 'code': wc, 'premium': row['premium'],
        'mkt_cap_亿': row.get('mkt_cap_亿', 0), 'block_amount': row['block_amount'],
        'state': state, 'price': px,
        'ma5': ma5.iloc[i], 'ma10': ma10.iloc[i],
        'over_ma': '是' if (o5 or o10) else '否',
        'action': '❌跳过BULL>MA' if skip else '✅ 买入',
        'pct': pct,
    })

print(f"{'信号日':>10s}  {'代码':>12s}  {'溢价':>6s}  {'LLT':>8s}  {'仓位':>5s}  {'操作':>16s}")
print("-" * 60)
for r in results:
    print(f"{r['trade_dt']:>10s}  {r['code']:>12s}  "
          f"{r['premium']:>+5.1%}  {r['state']:>8s}  "
          f"{r['pct']:>3d}%  {r['action']:>16s}")

print(f"\n共 {len(results)} 条, 可买入 {sum(1 for r in results if '买入' in r['action'])} 条")