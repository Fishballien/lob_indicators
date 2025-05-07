# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:25:48 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
from numba import typed, types
from datetime import datetime


# from go_through_book import process_a, process_d_or_t, check_relocate_best_px, estimate_theoretical_best_price
from go_through_book import order_dtype, trade_dtype, order_type, trade_type
from go_through_book import loop_until_next_ts


# %%
symbol = '600519.XSHG'
data_dir = Path(r'D:\CNIndexFutures\timeseries\lob_indicators\sample_data')

start = datetime.now()
# %% read data
trade_path = data_dir / 'trade' / f'{symbol}.parquet'
order_path = data_dir / 'order' / f'{symbol}.parquet'
trade_data = pd.read_parquet(trade_path)
order_data = pd.read_parquet(order_path)


# %% preprocess
order_data_extracted = order_data[['OrderTime', 'SeqNum']].rename(columns={'OrderTime': 'time'})
order_data_extracted['index'] = order_data.index
order_data_extracted['data_type'] = 0

trade_data_extracted = trade_data[['datetime', 'SeqNum']].rename(columns={'datetime': 'time'})
trade_data_extracted['index'] = trade_data.index
trade_data_extracted['data_type'] = 1

combined_data = pd.concat([order_data_extracted, trade_data_extracted], ignore_index=True)
combined_data = combined_data.sort_values(by=['time', 'SeqNum', 'data_type']).reset_index(drop=True)
data_type_arr = combined_data['data_type'].values.astype('i4')
index_arr = combined_data['index'].values.astype('i8')
time_arr = combined_data['time'].values.astype('i8')

order_named_arr = np.zeros(len(order_data), dtype=order_dtype)
order_named_arr['orderno'] = order_data['OrderNo'].values
order_named_arr['px'] = order_data['OrderPx'].values
order_named_arr['qty'] = order_data['OrderQty'].values
order_named_arr['side'] = np.where(order_data['Side'].values == b'B', 0, 1)
order_named_arr['ordertype'] = order_data['OrderType'].values.astype('S1')

trade_named_arr = np.zeros(len(trade_data), dtype=trade_dtype)
trade_named_arr['tradp'] = trade_data['tradp'].values
trade_named_arr['tradv'] = trade_data['tradv'].values
trade_named_arr['buyno'] = trade_data['buyno'].values
trade_named_arr['sellno'] = trade_data['sellno'].values
trade_named_arr['side'] = np.where(trade_data['Side'].values == b'B', 0, 1)


# %% containers
unique_orderno = np.sort(np.unique(order_data['OrderNo'])).astype(np.int64)
len_of_orderno = len(unique_orderno)
# orderno_mapping = {orderno: idx for idx, orderno in enumerate(unique_orderno)} # TODO: 改numbadict
orderno_mapping = typed.Dict.empty(
    key_type=types.int64,
    value_type=types.int64
)
for idx, orderno in enumerate(unique_orderno):
    orderno_mapping[orderno] = idx
on_ts_org = np.zeros_like(unique_orderno, dtype='i8') #'M8[ms]'
on_side = np.full_like(unique_orderno, fill_value=-1, dtype='int32')
on_px = np.zeros_like(unique_orderno, dtype=np.int64)
on_qty_org = np.zeros_like(unique_orderno, dtype=np.int64)
on_qty_remain = np.zeros_like(unique_orderno, dtype=np.int64)

unique_prices = np.sort(np.unique(order_data['OrderPx'])).astype(np.int64)
len_of_price = len(unique_prices)
# price_mapping = {px: idx for idx, px in enumerate(unique_prices)}
price_mapping = typed.Dict.empty(
    key_type=types.int64,
    value_type=types.int64
)
for idx, px in enumerate(unique_prices):
    price_mapping[px] = idx
lob_bid = np.zeros(len_of_price, dtype=np.int64)
lob_ask = np.zeros(len_of_price, dtype=np.int64)

best_px = np.zeros(2, dtype=np.int64)
best_if_lost = np.zeros(2, dtype=np.int32)
best_if_lost[0] = 0  # True as integer 1
best_if_lost[1] = 0  # True as integer 1


# %%
# =============================================================================
# ts_pre = 0
# start_idx = 0
# len_combined = len(combined_data)
# 
# for c_i, c_idx in enumerate(range(start_idx, len_combined)):
#     # read target data
#     data_type = data_type_arr[c_idx]
#     idx = index_arr[c_idx]
#     ts = time_arr[c_idx]
#     
#     if c_i != 0 and ts != ts_pre:
#         if best_px[0] != 0 and best_px[1] != 0:
#             # step1: 找当前真实存在挂单的最优价
#             check_relocate_best_px(best_px, best_if_lost, price_mapping, unique_prices, lob_bid, lob_ask)
#             # step2: 模拟撮合后的最优价
#             estimate_theoretical_best_price(best_px, price_mapping, unique_prices, lob_bid, lob_ask)
#         
#     if data_type == b'o':
#         row = order_named_arr[idx]
#         orderno = row['orderno']
#         px = row['px']
#         qty = row['qty']
#         side = row['side']
#         ordertype = row['ordertype']
#         if ordertype == b'A':
#             process_a(orderno, ts, side, px, qty, on_ts_org, on_side, on_px, on_qty_org, on_qty_remain,
#                       lob_bid, lob_ask, best_px, best_if_lost,
#                       orderno_mapping, price_mapping)
#         elif ordertype == b'D':
#             process_d_or_t(orderno, px, side, qty, on_qty_remain,
#                            lob_bid, lob_ask, best_px, best_if_lost,
#                            orderno_mapping, price_mapping)
#     if data_type == b't':
#         row = trade_named_arr[idx]
#         tradp = row['tradp']
#         tradv = row['tradv']
#         buyno = row['buyno']
#         sellno = row['sellno']
#         side = row['side']
#         for target_order_no, target_side in zip((buyno, sellno), (0, 1)):
#             process_d_or_t(target_order_no, tradp, target_side, tradv, on_qty_remain,
#                            lob_bid, lob_ask, best_px, best_if_lost,
#                            orderno_mapping, price_mapping)
#     # print(best_px)
# =============================================================================
    

# %%
nxt_target_ts = 1e18
start_idx = 0
len_combined = len(combined_data)
loop_until_next_ts(start_idx, nxt_target_ts, len_combined, data_type_arr, index_arr, time_arr,
                   order_named_arr, trade_named_arr, 
                   orderno_mapping, price_mapping,
                   on_ts_org, on_side, on_px, on_qty_org, on_qty_remain,
                   best_px, best_if_lost, unique_prices, lob_bid, lob_ask)

end = datetime.now()
print(end-start)




























