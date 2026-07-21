"""
v8g 策略统一配置 — 修改路径后可直接运行
"""
import os

# ===== 数据路径 =====
# 数据目录: 包含 block_trades.csv, daily_prices.csv, derivative_indicators.csv
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "wind_data_full")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# 排除名单
EXCLUDED_PATH = os.path.join(os.path.dirname(__file__), "data", "stock_list", "excluded_stocks.csv")

# ===== 回测参数 =====
INITIAL_CAPITAL = 1_000_000  # 初始资金100万
PREMIUM_THRESHOLD = 0.05     # 溢价阈值 >=5%
MAX_MARKET_CAP = 100         # 市值上限 <100亿

# ===== 交易费用 =====
COMMISSION_RATE = 0.0008     # 佣金 万8
STAMP_TAX_RATE = 0.001      # 印花税 千1
MIN_COMMISSION = 5           # 最低佣金 5元

# ===== LLT仓位映射 =====
POSITION_MAP = {
    'BULL': 0.05,     # BULL: 5%仓位
    'NEUTRAL': 0.25,  # NEUTRAL: 25%仓位
    'BEAR': 0.35,     # BEAR: 35%仓位
}

# BULL状态下价格高于MA5/MA10则跳过不买
BULL_MA_FILTER = True

# LLT参数
LLT_ALPHA = 0.15

# 持有期
HOLD_DAYS = 30