# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:24:40 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
import traceback
import gc


# %% import self_defined
from core.go_through_book_full import GoThroughBookStepper
from utils.speedutils import timeit
from indicators.chatgpt import *
# from core.plot_lob import visualize_order_book


# %%
class GroupGenerate(GoThroughBookStepper):
    
    def __init__(self, symbol, date, order_data, trade_data, param):
        super().__init__(symbol, date, order_data, trade_data, param)
        
        self.symbol = symbol
        self.date = date
        
        self._init_dataset()
        self._get_ind_funcs()
        self._preprocess_param()
    
    def _init_indicator_related(self):
        factor_idx_mapping = self.param['factor_idx_mapping']
        self.recorded_dataset = np.full((len(self.stepper.target_ts), len(factor_idx_mapping)), 
                                        fill_value=np.nan, dtype=np.float64)
        self._init_curr_dataset()
        
    def _init_indicator_dtype(self):
        pass
    
    def _init_curr_dataset(self):
        indxview_count = self.param['indxview_count']
        
        self.curr_dataset = {}
        for indxview, count in indxview_count.items():
            self.curr_dataset[indxview] = np.full((count, 2), fill_value=np.nan, dtype=np.float64)
    
    def _init_dataset(self):
        self.dataset = {}
        self.dataset['on_ts_org'] = self.on_ts_org
        self.dataset['on_ts_d'] = self.on_ts_d
        self.dataset['on_ts_t'] = self.on_ts_t
        self.dataset['on_side'] = self.on_side
        self.dataset['on_px'] = self.on_px
        self.dataset['on_qty_org'] = self.on_qty_org
        self.dataset['on_qty_remain'] = self.on_qty_remain
        self.dataset['on_qty_d'] = self.on_qty_d
        self.dataset['on_qty_t'] = self.on_qty_t
        self.dataset['on_amt_t'] = self.on_amt_t  # 成交金额
        
        # 新增：主动、被动和集合竞价成交量及金额
        self.dataset['on_qty_t_a'] = self.on_qty_t_a  # 主动成交量
        self.dataset['on_amt_t_a'] = self.on_amt_t_a  # 主动成交金额
        self.dataset['on_qty_t_p'] = self.on_qty_t_p  # 被动成交量
        self.dataset['on_amt_t_p'] = self.on_amt_t_p  # 被动成交金额
        self.dataset['on_qty_t_n'] = self.on_qty_t_n  # 集合竞价成交量
        self.dataset['on_amt_t_n'] = self.on_amt_t_n  # 集合竞价成交金额
        
        self.dataset['best_px'] = self.best_px_post_match
        self.dataset['ts'] = 0
        
        self.list_to_check_valid = ['on_ts_org', 'on_ts_d', 'on_ts_t', 'on_side', 'on_px', 
                                    'on_qty_org', 'on_qty_remain', 'on_qty_d', 'on_qty_t', 'on_amt_t',
                                    'on_qty_t_a', 'on_amt_t_a', 'on_qty_t_p', 'on_amt_t_p', 
                                    'on_qty_t_n', 'on_amt_t_n']  # 修改：新增主动被动集合竞价字段
   
    def _get_ind_funcs(self):
        ind_cates = self.param['ind_cates']

        self.ind_funcs = {}
        for ind_cate in ind_cates:
            self.ind_funcs[ind_cate] = globals()[ind_cate]
      
    def _preprocess_param(self):
        ind_cates = self.param['ind_cates']
        
        for ind_cate in ind_cates:
            ind_param = self.param[ind_cate]
            inputs = ind_param['inputs']
            param_dict = ind_param.get('param', {})
            for ipt_name in inputs:
                assert ipt_name in self.dataset or ipt_name in param_dict or ipt_name == 'curr_dataset', ipt_name
                assert not (ipt_name in self.dataset and ipt_name in param_dict), ipt_name
            ind_param['param'] = {k: np.array(v, dtype=np.float64) for k, v in param_dict.items()}

    def run(self):
        ind_cates = self.param['ind_cates']
        indxview_count = self.param['indxview_count']
        view_infos = self.param['view_infos']
        factor_idx_mapping = self.param['factor_idx_mapping']

        for ts_idx, ts in self.stepper:
            ts_dataset = self._update_valid_data(ts)
            # visualize_order_book(ts_dataset)
            # if ts_idx == 41:
            #     breakpoint()
            for view_name, view_info in view_infos.items():
                view_dataset, status = self._cut_view(view_name, view_info, ts_dataset)
                
                if status != 0:
                    continue
                
                for ind_cate in ind_cates:
                    ind_func = self.ind_funcs[ind_cate]
                    input_dict = self._fill_ind_x_view_input(ind_cate, view_name, view_dataset)
                    indxview_len = indxview_count[(ind_cate, view_name)]

                    try:
                        ind_func(*input_dict.values())
                        for idx in range(indxview_len):
                            for side in (0, 1):
                                self.recorded_dataset[ts_idx, factor_idx_mapping[(ind_cate, view_name, idx, side)]] = (
                                    self.curr_dataset[(ind_cate, view_name)][idx, side])
                    except:
                        print(ind_func.__name__)
                        traceback.print_exc()
        return self.final()

    def _update_valid_data(self, ts):
        valid_idx = (self.on_side != -1) & (self.on_ts_org > 0) #& (self.on_qty_remain > 0)
        dataset = {}
        for col in self.dataset:
            if col in self.list_to_check_valid:
                dataset[col] = self.dataset[col][valid_idx]
            elif col == 'ts':
                dataset[col] = ts
            else:
                dataset[col] = self.dataset[col]
        return dataset

    def _fill_ind_x_view_input(self, ind_cate, view_name, view_dataset):
        input_dict = {}
        ind_param = self.param[ind_cate]
        inputs = ind_param['inputs']
        param_dict = ind_param['param']

        for ipt_name in inputs:
            if ipt_name in view_dataset:
                input_dict[ipt_name] = view_dataset[ipt_name]
            elif ipt_name in param_dict:
                input_dict[ipt_name] = param_dict[ipt_name]
            elif ipt_name == 'curr_dataset':
                input_dict[ipt_name] = self.curr_dataset[(ind_cate, view_name)]
        return input_dict
             
    def final(self):
        target_ts = self.stepper.target_ts.astype('i8')
        res = {'ts': target_ts, 'indicator': self.recorded_dataset}
        return res
                    
    def _cut_view(self, view_name, view_info, dataset):
        return dataset, 0
    
    
# %% cut price range
class GGCutPriceRange(GroupGenerate):
    # @timeit
    def _cut_view(self, view_name, view_info, ts_dataset):
        price_range = view_info['price_range']
        
        best_px = ts_dataset['best_px']
        on_px = ts_dataset['on_px']
        
        bid1 = best_px[0]
        ask1 = best_px[1]
        
        if bid1 == 0 or ask1 == 0:
            return None, 1
        
        mid_price = (bid1 + ask1) / 2
        lower_bound = mid_price * (1 - price_range)
        upper_bound = mid_price * (1 + price_range)
        
        idx_in_price_range = (on_px >= lower_bound) & (on_px <= upper_bound)
        # print(np.sum(idx_in_price_range))
        
        view_dataset = {}
        for col in ts_dataset:
            if col in self.list_to_check_valid:
                view_dataset[col] = ts_dataset[col][idx_in_price_range]
            else:
                view_dataset[col] = ts_dataset[col]
        return view_dataset, 0
    

class GGCutPriceRangeNOrderAmount(GroupGenerate):
    # @timeit
    def _cut_view(self, view_name, view_info, ts_dataset):
        price_range = view_info['price_range']
        amount_thres = view_info['amount_thres']
        
        best_px = ts_dataset['best_px']
        on_px = ts_dataset['on_px']
        on_qty_org = ts_dataset['on_qty_org']
        
        bid1 = best_px[0]
        ask1 = best_px[1]
        
        if bid1 == 0 or ask1 == 0:
            return None, 1
        
        mid_price = (bid1 + ask1) / 2
        lower_bound = mid_price * (1 - price_range)
        upper_bound = mid_price * (1 + price_range)
        
        idx_in_price_range = (on_px >= lower_bound) & (on_px <= upper_bound)
        idx_in_amount_thres = (on_px * on_qty_org / 10000 >= amount_thres)
        
        view_dataset = {}
        for col in ts_dataset:
            if col in self.list_to_check_valid:
                view_dataset[col] = ts_dataset[col][idx_in_price_range & idx_in_amount_thres]
            else:
                view_dataset[col] = ts_dataset[col]
        return view_dataset, 0
    
    
class GGCutOrderAmount(GroupGenerate):
    # @timeit
    def _cut_view(self, view_name, view_info, ts_dataset):
        amount_thres = view_info['amount_thres']
        
        best_px = ts_dataset['best_px']
        on_px = ts_dataset['on_px']
        on_qty_org = ts_dataset['on_qty_org']
        
        bid1 = best_px[0]
        ask1 = best_px[1]
        
        if bid1 == 0 or ask1 == 0:
            return None, 1

        idx_in_amount_thres = (on_px * on_qty_org / 10000 >= amount_thres)
        
        view_dataset = {}
        for col in ts_dataset:
            if col in self.list_to_check_valid:
                view_dataset[col] = ts_dataset[col][idx_in_amount_thres]
            else:
                view_dataset[col] = ts_dataset[col]
        return view_dataset, 0


# %% 基于成交类型的视图切分类
class GGCutTradeType(GroupGenerate):
    """基于成交类型进行数据筛选的视图切分器（主动/被动/集合竞价）"""
    
    def _cut_view(self, view_name, view_info, ts_dataset):
        trade_types = view_info.get('trade_types', ['active', 'passive', 'auction'])  # 默认包含所有类型
        
        best_px = ts_dataset['best_px']
        
        bid1 = best_px[0]
        ask1 = best_px[1]
        
        if bid1 == 0 or ask1 == 0:
            return None, 1
        
        # 根据成交类型筛选订单
        idx_mask = np.zeros(len(ts_dataset['on_px']), dtype=bool)
        
        if 'active' in trade_types:
            idx_mask |= (ts_dataset['on_qty_t_a'] > 0)
        if 'passive' in trade_types:
            idx_mask |= (ts_dataset['on_qty_t_p'] > 0)
        if 'auction' in trade_types:
            idx_mask |= (ts_dataset['on_qty_t_n'] > 0)
        
        view_dataset = {}
        for col in ts_dataset:
            if col in self.list_to_check_valid:
                view_dataset[col] = ts_dataset[col][idx_mask]
            else:
                view_dataset[col] = ts_dataset[col]
        return view_dataset, 0


class GGCutPriceRangeNTradeType(GroupGenerate):
    """结合价格范围和成交类型进行数据筛选的视图切分器"""
    
    def _cut_view(self, view_name, view_info, ts_dataset):
        price_range = view_info['price_range']
        trade_types = view_info.get('trade_types', ['active', 'passive', 'auction'])
        
        best_px = ts_dataset['best_px']
        on_px = ts_dataset['on_px']
        
        bid1 = best_px[0]
        ask1 = best_px[1]
        
        if bid1 == 0 or ask1 == 0:
            return None, 1
        
        mid_price = (bid1 + ask1) / 2
        lower_bound = mid_price * (1 - price_range)
        upper_bound = mid_price * (1 + price_range)
        
        idx_in_price_range = (on_px >= lower_bound) & (on_px <= upper_bound)
        
        # 根据成交类型筛选订单
        idx_in_trade_type = np.zeros(len(on_px), dtype=bool)
        
        if 'active' in trade_types:
            idx_in_trade_type |= (ts_dataset['on_qty_t_a'] > 0)
        if 'passive' in trade_types:
            idx_in_trade_type |= (ts_dataset['on_qty_t_p'] > 0)
        if 'auction' in trade_types:
            idx_in_trade_type |= (ts_dataset['on_qty_t_n'] > 0)
        
        view_dataset = {}
        for col in ts_dataset:
            if col in self.list_to_check_valid:
                view_dataset[col] = ts_dataset[col][idx_in_price_range & idx_in_trade_type]
            else:
                view_dataset[col] = ts_dataset[col]
        return view_dataset, 0


# %% 移除原有的基于成交方向的视图切分类，替换为基于成交类型的实现
# 以下是原来基于 trade_direction 的类，现在已被上面基于成交类型的类替代

# class GGCutTradeDirection(GroupGenerate):
#     """已废弃：原基于成交方向进行数据筛选的视图切分器"""
#     pass

# class GGCutPriceRangeNTradeDirection(GroupGenerate):
#     """已废弃：原结合价格范围和成交方向进行数据筛选的视图切分器"""
#     pass