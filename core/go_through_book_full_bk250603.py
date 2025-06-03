# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:44:17 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from numba import njit, types, typed, from_dtype
from functools import partial
from numba.types import DictType
from abc import ABC, abstractmethod


from core.loop import FixedTimeIntervalLoop
from utils.market import get_exchange, Action, Exchange, Side, DataType, MINIMUM_SIZE_FILTER, DefaultPx
from utils.timeutils import adjust_timestamp_precision


# %%
class GoThroughBook:
    
    def __init__(self, symbol, order_data, trade_data):
        self.exchange = get_exchange(symbol)
        self._preprocess_data(order_data, trade_data)
        self._init_containers(order_data)
        self.loop_func = self._init_loop_func()
        
    def _preprocess_data(self, order_data, trade_data):
        # print('order', len(order_data))
        # print('trade', len(trade_data))
        order_data_extracted = order_data[['OrderTime', 'SeqNum']].rename(columns={'OrderTime': 'time'})
        order_data_extracted['index'] = order_data.index
        order_data_extracted['data_type'] = DataType.Order.value

        trade_data_extracted = trade_data[['datetime', 'SeqNum']].rename(columns={'datetime': 'time'})
        trade_data_extracted['index'] = trade_data.index
        trade_data_extracted['data_type'] = DataType.Trade.value

        combined_data = pd.concat([order_data_extracted, trade_data_extracted], ignore_index=True)
        combined_data = combined_data.sort_values(by=['time', 'data_type', 'SeqNum'],  
                                                  ascending=[True, True, True]).reset_index(drop=True)
        len_combined = len(combined_data)
        data_type_arr = combined_data['data_type'].values.astype('i4')
        index_arr = combined_data['index'].values.astype('i8')
        time_arr = (combined_data['time'].values.astype('int64') // 1_000).astype('i8')
        time_arr = adjust_timestamp_precision(combined_data['time'].values)

        order_named_arr = np.zeros(len(order_data), dtype=order_dtype)
        order_named_arr['orderno'] = order_data['OrderNo'].values
        order_named_arr['px'] = (np.round(order_data['OrderPx'].values / 10) * 10).astype(np.int64)
        order_named_arr['qty'] = order_data['OrderQty'].values
        order_named_arr['side'] = np.where(order_data['Side'].values == b'B', Side.Bid.value, Side.Ask.value)
        order_named_arr['ordertype'] = np.where(order_data['OrderType'].values == b'D', b'D', b'A').astype('S1')
# =============================================================================
#         seqnum_in_trade = order_data['SeqNum'].isin(trade_data['SeqNum'])
#         ts_after_930 = order_data['OrderTime'].dt.time >= pd.to_datetime('09:30:00').time()
#         order_named_arr['is_trade_fill'] = (seqnum_in_trade & ts_after_930).astype(np.int32)
# =============================================================================

        trade_named_arr = np.zeros(len(trade_data), dtype=trade_dtype)
        trade_named_arr['tradp'] = (np.round(trade_data['tradp'].values / 10) * 10).astype(np.int64)
        trade_named_arr['tradv'] = trade_data['tradv'].values
        trade_named_arr['buyno'] = trade_data['buyno'].values
        trade_named_arr['sellno'] = trade_data['sellno'].values
        conditions = [
            trade_data['Side'].values == b'B',
            trade_data['Side'].values == b'S',
            trade_data['Side'].values == b'N'
        ]
        choices = [Side.Bid.value, Side.Ask.value, Side.N.value]
        trade_named_arr['side'] = np.select(conditions, choices, default=np.nan)
        # trade_named_arr['side'] = np.where(trade_data['Side'].values == b'B', Side.Bid.value, Side.Ask.value)
        # open_auction = trade_data['datetime'].dt.time < pd.to_datetime('09:30:00').time()
        # close_auction = trade_data['datetime'].dt.time > pd.to_datetime('14:57:00').time()
        pre_defined_auction = trade_named_arr['side'] == Side.N.value
        trade_named_arr['is_auction'] = pre_defined_auction.astype(np.int32)
        
        self.len_combined = len_combined
        self.data_type_arr = data_type_arr
        self.index_arr = index_arr
        self.time_arr = time_arr
        self.order_named_arr = order_named_arr
        self.trade_named_arr = trade_named_arr
        
    def _init_containers(self, order_data):
        unique_orderno = np.sort(np.unique(order_data['OrderNo'])).astype(np.int64)
        # orderno_mapping = {orderno: idx for idx, orderno in enumerate(unique_orderno)} # TODO: 改numbadict
        orderno_mapping = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        for idx, orderno in enumerate(unique_orderno):
            orderno_mapping[orderno] = idx
        on_ts_org = np.zeros_like(unique_orderno, dtype='i8') #'M8[ms]'
        on_ts_d = np.zeros_like(unique_orderno, dtype='i8')
        on_ts_t = np.zeros_like(unique_orderno, dtype='i8')
        on_side = np.full_like(unique_orderno, fill_value=-1, dtype='int32')
        on_px = np.zeros_like(unique_orderno, dtype=np.int64)
        on_qty_org = np.zeros_like(unique_orderno, dtype=np.int64)
        on_qty_remain = np.zeros_like(unique_orderno, dtype=np.int64)
        on_qty_d = np.zeros_like(unique_orderno, dtype=np.int64)
        on_qty_t = np.zeros_like(unique_orderno, dtype=np.int64)
        on_amt_t = np.zeros_like(unique_orderno, dtype=np.int64)  # 新增：成交金额

        unique_prices = np.sort(np.unique(self.order_named_arr['px'])).astype(np.int64)
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
        best_px_post_match = np.zeros(2, dtype=np.int64)
        best_if_lost = np.zeros(2, dtype=np.int32)
        best_if_lost[0] = 0  # True as integer 1
        best_if_lost[1] = 0  # True as integer 1
        
        self.unique_orderno = unique_orderno
        self.orderno_mapping = orderno_mapping
        self.on_ts_org = on_ts_org
        self.on_ts_d = on_ts_d
        self.on_ts_t = on_ts_t
        self.on_side = on_side
        self.on_px = on_px
        self.on_qty_org = on_qty_org
        self.on_qty_remain = on_qty_remain
        self.on_qty_d = on_qty_d
        self.on_qty_t = on_qty_t
        self.on_amt_t = on_amt_t  # 新增
        
        self.unique_prices = unique_prices
        self.price_mapping = price_mapping
        self.lob_bid = lob_bid
        self.lob_ask = lob_ask
        self.best_px = best_px
        self.best_px_post_match = best_px_post_match
        self.best_if_lost = best_if_lost

    def _init_loop_func(self):
        loop_func = partial(loop_until_next_ts_wrapper, len_combined=self.len_combined, 
                            data_type_arr=self.data_type_arr, index_arr=self.index_arr, time_arr=self.time_arr,
                            order_named_arr=self.order_named_arr, trade_named_arr=self.trade_named_arr, 
                            orderno_mapping=self.orderno_mapping, price_mapping=self.price_mapping,
                            on_ts_org=self.on_ts_org, on_ts_d=self.on_ts_d, on_ts_t=self.on_ts_t, 
                            on_side=self.on_side, on_px=self.on_px, 
                            on_qty_org=self.on_qty_org, on_qty_remain=self.on_qty_remain,
                            on_qty_d=self.on_qty_d, on_qty_t=self.on_qty_t, on_amt_t=self.on_amt_t,  # 修改：新增on_amt_t
                            best_px=self.best_px, best_px_post_match=self.best_px_post_match, 
                            best_if_lost=self.best_if_lost, 
                            unique_prices=self.unique_prices, lob_bid=self.lob_bid, lob_ask=self.lob_ask,
                            exchange=self.exchange)
        return loop_func
    
    
class GoThroughBookStepper(GoThroughBook, ABC):
    
    def __init__(self, symbol, date, order_data, trade_data, param):
        super().__init__(symbol, order_data, trade_data)
        self.param = param
        target_ts_param = self.param['target_ts']
        self.stepper = FixedTimeIntervalLoop(date, self.loop_func, target_ts_param, self.len_combined)
        self._init_indicator_related()
    
    @abstractmethod
    def _init_indicator_dtype(self):
        pass
    
    @abstractmethod
    def _init_curr_dataset(self):
        pass
        
    def _init_indicator_related(self):
        self.recorded_dtype = self._init_indicator_dtype()
        self.recorded_dataset = np.full(len(self.stepper.target_ts), fill_value=np.nan, dtype=self.recorded_dtype)
        self._init_curr_dataset()
        
    def final(self):
        target_ts = self.stepper.target_ts
        res_descr =  [('timestamp', 'i8')] + self.recorded_dtype.descr
        res_dtype = np.dtype(res_descr)
        res = np.zeros(len(target_ts), dtype=res_dtype)
        res['timestamp'] = target_ts
        for name in list(self.recorded_dtype.names):
            res[name] = self.recorded_dataset[name]
        return res
    

# %%
order_dtype = np.dtype([
    ('orderno', 'int64'), ('px', 'int64'), ('qty', 'int64'),
    ('side', 'int32'), ('ordertype', 'S1') #, ('is_trade_fill', 'int32')
])
order_type = from_dtype(order_dtype)


trade_dtype = np.dtype([
    ('tradp', 'int64'), ('tradv', 'int64'), ('buyno', 'int64'), ('sellno', 'int64'),
    ('side', 'int32'), ('is_auction', 'int32')
])
trade_type = from_dtype(trade_dtype)


# %% loop
@njit(types.void(
    types.int32, types.int64, types.int64, types.int64[:], types.int32[:]
))
def update_best_px(side, price, size, best_px, best_if_lost):
    if side == Side.Bid.value:
        if size > MINIMUM_SIZE_FILTER and (price > best_px[0] or best_px[0] == 0):
            best_px[0] = price
        elif size < MINIMUM_SIZE_FILTER and price == best_px[0]:
            best_if_lost[0] = 1
    elif side == Side.Ask.value:
        if size > MINIMUM_SIZE_FILTER and (price < best_px[1] or best_px[1] == 0):
            best_px[1] = price
        elif size < MINIMUM_SIZE_FILTER and price == best_px[1]:
            best_if_lost[1] = 1
            

@njit(types.void(
    types.int64, types.int64, types.int32, types.int64, types.int64,
    types.int64[:], types.int32[:], types.int64[:], types.int64[:], types.int64[:],
    types.int64[:], types.int64[:], types.int64[:], types.int32[:],
    DictType(types.int64, types.int64), DictType(types.int64, types.int64)
))
def process_a(orderno, ts, side, px, qty, on_ts_org, on_side, on_px, on_qty_org, on_qty_remain,
              lob_bid, lob_ask, best_px, best_if_lost,
              orderno_mapping, price_mapping):
    # update no related
    if orderno not in orderno_mapping:
        print('a', orderno)
    no_idx = orderno_mapping[orderno]
    on_qty_org[no_idx] += qty
    on_qty_remain[no_idx] += qty
    if on_ts_org[no_idx] == 0:
        on_ts_org[no_idx] = ts
        on_side[no_idx] = side
        on_px[no_idx] = px
    if (side == Side.Bid.value and px > on_px[no_idx]) or (side == Side.Ask.value and px < on_px[no_idx]):
        on_px[no_idx] = px
    # update lob related
    lob_idx = price_mapping[px]
    target_lob = lob_bid if side == Side.Bid.value else lob_ask
    target_lob[lob_idx] += qty
    # update best price
    update_best_px(side, px, target_lob[lob_idx], best_px, best_if_lost)
            

@njit(types.void(
    types.int64, types.int64, types.int32, types.int64, types.int64, types.int64[:], types.int64[:], types.int64[:], 
    types.int64[:], types.int64[:], types.int64[:], types.int64[:], types.int64[:], 
    types.int64[:], types.int32[:], types.int64[:],  # 修改：新增on_amt_t
    DictType(types.int64, types.int64), DictType(types.int64, types.int64),
    types.int32, types.int32, types.int32
))
def process_d_or_t(orderno, ts, side, px, qty, on_ts_d, on_ts_t, on_qty_remain, on_qty_d, on_qty_t, on_px,
                   lob_bid, lob_ask, best_px, best_if_lost, on_amt_t,  # 修改：新增on_amt_t参数
                   orderno_mapping, price_mapping, 
                   action_type, exchange, is_auction):
    # update no related
    if orderno not in orderno_mapping:
        print('d or t', orderno)
        return
    target_no_idx = orderno_mapping[orderno]
    on_qty_remain[target_no_idx] -= qty
    if action_type == Action.T.value:
        on_qty_t[target_no_idx] += qty
        # 新增：更新成交金额
        on_amt_t[target_no_idx] += px * qty
        if on_ts_t[target_no_idx] == 0:
            on_ts_t[target_no_idx] = ts
    elif action_type == Action.D.value:
        on_qty_d[target_no_idx] += qty
        if on_ts_d[target_no_idx] == 0:
            on_ts_d[target_no_idx] = ts
    # try:
    #     assert on_qty_remain[target_no_idx] >= 0
    # except:
    #     breakpoint()
    
    # update lob related
    ## 沪市非集合竞价成交，没有对应真实order，只有每笔trade反推出的order，导致每笔order价格不同
    ## 所以要根据trade价格反推找到order px
    use_data_px = action_type == Action.T.value and exchange == Exchange.SH.value and is_auction == 0
    order_px = px if use_data_px else on_px[target_no_idx]
    # try:
    #     assert order_px > 0
    # except:
    #     breakpoint()
    if order_px == 0:
        ## 若order px为0，说明是沪市反推的order，且trade找不到对应的order，则加回一开始消耗掉的qty
        ##（一开始数据没洗好有这个情况，现在可能已经没有了）
        on_qty_remain[target_no_idx] += qty
        # 如果是成交且order_px为0，需要回滚成交金额
        # if action_type == Action.T.value:
        #     on_amt_t[target_no_idx] -= px * qty / 10000.0
        # if action_type == Action.D.value:
        #     on_qty_remain[target_no_idx] += qty
        # else:
        #     try:
        #         assert order_px > 0
        #     except:
        #         breakpoint()
    else:
        lob_idx = price_mapping[order_px]
        target_lob = lob_bid if side == Side.Bid.value else lob_ask
        target_lob[lob_idx] -= qty
        # try:
        assert target_lob[lob_idx] >= 0
        # except:
        #     breakpoint()
        # update best price
        update_best_px(side, order_px, target_lob[lob_idx], best_px, best_if_lost)


@njit(types.void(
    types.int32, types.int64[:], types.int32[:],
    DictType(types.int64, types.int64), types.int64[:], types.int64[:]
))    
def relocate_best_px(side, best_px, best_if_lost, price_mapping, prices, lob_side):
    fake_best_px = best_px[side]
    px_idx = price_mapping[fake_best_px]
    if side == Side.Bid.value:
        while px_idx >= 0 and lob_side[px_idx] <= MINIMUM_SIZE_FILTER:
            px_idx -= 1
        best_px[side] = prices[px_idx] if px_idx >= 0 else DefaultPx.Bid.value
    elif side == Side.Ask.value:
        while px_idx < lob_side.size and lob_side[px_idx] <= MINIMUM_SIZE_FILTER:
            px_idx += 1
        best_px[side] = prices[px_idx] if px_idx < lob_side.size else DefaultPx.Ask.value
    best_if_lost[side] = 0


@njit(types.void(
    types.int64[:], types.int32[:], DictType(types.int64, types.int64),
    types.int64[:], types.int64[:], types.int64[:]
))
def check_relocate_best_px(best_px, best_if_lost, price_mapping, prices, lob_bid, lob_ask):
    if best_if_lost[0] == 1:
        relocate_best_px(0, best_px, best_if_lost, price_mapping, prices, lob_bid)
    if best_if_lost[1] == 1:
        relocate_best_px(1, best_px, best_if_lost, price_mapping, prices, lob_ask)
        

@njit(types.void(
    types.int64[:], types.int64[:], DictType(types.int64, types.int64),
    types.int64[:], types.int64[:], types.int64[:]
))
def estimate_theoretical_best_price(best_px, best_px_post_match, price_mapping, prices, lob_bid, lob_ask):
    best_bid = best_px[0]
    best_ask = best_px[1]
    if best_bid < best_ask:
        best_px_post_match[0] = best_px[0]
        best_px_post_match[1] = best_px[1]
    else:
    # print('best_px', best_px)
    # print('best_px_post_match', best_px_post_match)
        best_bid_idx = price_mapping[best_bid]
        best_ask_idx = price_mapping[best_ask]
        len_lob = prices.size
        best_bid_remain = lob_bid[best_bid_idx]
        best_ask_remain = lob_ask[best_ask_idx]
        while best_bid_idx >= best_ask_idx:
            
            matched = min(best_bid_remain, best_ask_remain)
            best_bid_remain -= matched
            best_ask_remain -= matched
            
            while best_bid_remain == 0 and best_bid_idx > 0:
                best_bid_idx -= 1
                best_bid_remain = lob_bid[best_bid_idx]
            while best_ask_remain == 0 and best_ask_idx < len_lob - 1:
                best_ask_idx += 1
                best_ask_remain = lob_ask[best_ask_idx]
            if best_bid_idx == 0 or best_ask_idx == len_lob - 1:
                break
                
        best_px_post_match[0] = prices[best_bid_idx] if best_bid_idx != 0 else DefaultPx.Bid.value
        best_px_post_match[1] = prices[best_ask_idx] if best_ask_idx != len_lob - 1 else DefaultPx.Ask.value
    if (best_px_post_match[Side.Bid.value] == DefaultPx.Bid.value 
        and best_px_post_match[Side.Ask.value] != DefaultPx.Ask.value):
        best_px_post_match[Side.Bid.value] = best_px_post_match[Side.Ask.value]
    if (best_px_post_match[Side.Ask.value] == DefaultPx.Ask.value 
        and best_px_post_match[Side.Bid.value] != DefaultPx.Bid.value):
        best_px_post_match[Side.Ask.value] = best_px_post_match[Side.Bid.value]


@njit(types.int64(
    types.int64, types.int64, types.int64, types.int32[:], types.int64[:], types.int64[:],
    order_type[:], trade_type[:],
    DictType(types.int64, types.int64), DictType(types.int64, types.int64),
    types.int64[:], types.int64[:], types.int64[:], types.int32[:], types.int64[:], 
    types.int64[:], types.int64[:], types.int64[:], types.int64[:], types.int64[:],  # 修改：新增on_amt_t
    types.int64[:], types.int64[:], types.int32[:], types.int64[:], types.int64[:], types.int64[:], types.int32
))
def loop_until_next_ts(start_idx, nxt_target_ts, len_combined, data_type_arr, index_arr, time_arr,
                       order_named_arr, trade_named_arr, 
                       orderno_mapping, price_mapping,
                       on_ts_org, on_ts_d, on_ts_t, on_side, on_px, 
                       on_qty_org, on_qty_remain, on_qty_d, on_qty_t, on_amt_t,  # 修改：新增on_amt_t
                       best_px, best_px_post_match, best_if_lost, unique_prices, lob_bid, lob_ask, exchange):      
    ts_pre = 0
    
    for c_i, c_idx in enumerate(range(start_idx, len_combined)):
        # read target data
        data_type = data_type_arr[c_idx]
        idx = index_arr[c_idx]
        ts = time_arr[c_idx]
        
        if c_i != 0 and ts != ts_pre:
            if best_px[0] != 0 and best_px[1] != 0:
                # step1: 找当前真实存在挂单的最优价
                check_relocate_best_px(best_px, best_if_lost, price_mapping, unique_prices, lob_bid, lob_ask)
                # step2: 模拟撮合后的最优价
                estimate_theoretical_best_price(best_px, best_px_post_match, price_mapping, unique_prices, 
                                                lob_bid, lob_ask)
            # step3: 检查是否退出
            if ts > nxt_target_ts:
                return c_idx
            
        if data_type == 0:
            row = order_named_arr[idx]
            orderno = row['orderno']
            px = row['px']
            qty = row['qty']
            side = row['side']
            ordertype = row['ordertype']
            if ordertype == b'A':
                process_a(orderno, ts, side, px, qty, on_ts_org, on_side, on_px, on_qty_org, on_qty_remain,
                          lob_bid, lob_ask, best_px, best_if_lost,
                          orderno_mapping, price_mapping)
            elif ordertype == b'D':
                process_d_or_t(orderno, ts, side, px, qty, on_ts_d, on_ts_t, 
                               on_qty_remain, on_qty_d, on_qty_t, on_px,
                               lob_bid, lob_ask, best_px, best_if_lost, on_amt_t,  # 修改：新增on_amt_t
                               orderno_mapping, price_mapping, 
                               Action.D.value, exchange, 0)
        if data_type == 1:
            row = trade_named_arr[idx]
            tradp = row['tradp']
            tradv = row['tradv']
            buyno = row['buyno']
            sellno = row['sellno']
            side = row['side']
            is_auction = row['is_auction']
            for target_order_no, target_side in zip((buyno, sellno), (0, 1)):
                process_d_or_t(target_order_no, ts, target_side, tradp, tradv, on_ts_d, on_ts_t, 
                               on_qty_remain, on_qty_d, on_qty_t, on_px,
                               lob_bid, lob_ask, best_px, best_if_lost, on_amt_t,  # 修改：新增on_amt_t
                               orderno_mapping, price_mapping, 
                               Action.T.value, exchange, is_auction)
    return len_combined
    

def loop_until_next_ts_wrapper(start_idx, nxt_target_ts, len_combined, data_type_arr, index_arr, time_arr,
                               order_named_arr, trade_named_arr, 
                               orderno_mapping, price_mapping,
                               on_ts_org, on_ts_d, on_ts_t, on_side, on_px, 
                               on_qty_org, on_qty_remain, on_qty_d, on_qty_t, on_amt_t,  # 修改：新增on_amt_t
                               best_px, best_px_post_match, best_if_lost, unique_prices, lob_bid, lob_ask, exchange):
    return loop_until_next_ts(start_idx, nxt_target_ts, len_combined, data_type_arr, index_arr, time_arr,
                              order_named_arr, trade_named_arr, 
                              orderno_mapping, price_mapping,
                              on_ts_org, on_ts_d, on_ts_t, on_side, on_px, 
                              on_qty_org, on_qty_remain, on_qty_d, on_qty_t, on_amt_t,  # 修改：新增on_amt_t
                              best_px, best_px_post_match, best_if_lost, unique_prices, lob_bid, lob_ask, exchange)