# -*- coding: utf-8 -*-
"""
Market Order Book Analyzer

This script analyzes order book data for all stocks on a given date at specified
time intervals. It creates aggregated snapshots and visualizations using the
existing go_through_book infrastructure.

Author: Claude
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gc
import seaborn as sns
import time
import importlib
import toml

# Add project directory to path
import sys
file_path = Path(__file__).resolve()
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

from utils.load_data import load_data
from utils.dirutils import list_symbols_in_folder, load_path_config
from utils.timeutils import get_a_share_intraday_time_series

# Import necessary modules from the provided code
from core.go_through_book import GoThroughBookStepper
from core.auto_generate import GGCutPriceRange

class OrderBookAnalyzer(GGCutPriceRange):
    """
    Modified version of GGCutPriceRange to analyze and capture order book snapshots
    """
    
    def __init__(self, symbol, date, order_data, trade_data, param):
        super().__init__(symbol, date, order_data, trade_data, param)
        self.symbol = symbol
        self.date = date
        self.snapshots = {}
        
    def run(self):
        """
        Override of the run method to capture order book snapshots at each timestamp
        """
        view_infos = self.param['view_infos']
        
        for ts_idx, ts in self.stepper:
            ts_dataset = self._update_valid_data(ts)
            
            # Skip if no valid data at this timestamp
            if len(ts_dataset['on_px']) == 0:
                continue
                
            for view_name, view_info in view_infos.items():
                view_dataset, status = self._cut_view(view_name, view_info, ts_dataset)
                
                if status != 0:
                    continue
                
                # Capture snapshot for this timestamp
                self.capture_snapshot(ts, view_dataset)
                    
        return self.snapshots
    
    def capture_snapshot(self, ts, view_dataset):
        """
        Capture an order book snapshot at a specific timestamp
        
        Args:
            ts (int): Timestamp
            view_dataset (dict): Dataset containing order book data
        """
        # Extract relevant data
        on_side = view_dataset['on_side']
        on_px = view_dataset['on_px']
        on_qty_remain = view_dataset['on_qty_remain']
        best_px = view_dataset['best_px']
        
        # Calculate mid price
        mid_price = (best_px[0] + best_px[1]) / 2
        
        # Calculate time distance from order placement to current time
        time_distance = (ts - view_dataset['on_ts_org']) / 1000 / 60  # minutes
        
        # Calculate price distance as percentage from mid price
        price_distance_percent = (on_px - mid_price) / mid_price * 100
        
        # Calculate remaining order value
        order_value = on_px * on_qty_remain
        
        # Create dataframe
        df = pd.DataFrame({
            'time_distance': time_distance,
            'price_distance_percent': price_distance_percent,
            'order_value': order_value,
            'side': on_side
        })
        
        # Aggregate into time and price bins
        snapshot = self.aggregate_snapshot(df)
        
        # Store snapshot
        self.snapshots[ts] = snapshot
    
    def aggregate_snapshot(self, df, time_bin_width=1.0, price_bin_width=0.1):
        """
        Aggregate snapshot data into time and price bins with fixed widths
        
        Args:
            df (DataFrame): DataFrame with order book data
            time_bin_width (float): Width of each time bin in minutes
            price_bin_width (float): Width of each price bin in percentage points
            
        Returns:
            dict: Aggregated snapshot data
        """
        # Define max ranges
        max_time = min(df['time_distance'].max(), 480)  # Cap at 480 minutes (8 hours) for visualization
        min_price = np.floor(df['price_distance_percent'].min())
        max_price = np.ceil(df['price_distance_percent'].max())
        
        # Create bin edges with fixed widths
        time_edges = np.arange(0, max_time + time_bin_width, time_bin_width)
        price_edges = np.arange(min_price, max_price + price_bin_width, price_bin_width)
        
        # Separate buy and sell sides
        df_buy = df[df['side'] == 0]
        df_sell = df[df['side'] == 1]
        
        # Aggregate with 2D histogram
        heatmap_all, xedges, yedges = np.histogram2d(
            df['price_distance_percent'], 
            df['time_distance'], 
            bins=[price_edges, time_edges], 
            weights=df['order_value']
        )
        
        heatmap_buy, _, _ = np.histogram2d(
            df_buy['price_distance_percent'] if len(df_buy) > 0 else [], 
            df_buy['time_distance'] if len(df_buy) > 0 else [], 
            bins=[price_edges, time_edges], 
            weights=df_buy['order_value'] if len(df_buy) > 0 else None
        )
        
        heatmap_sell, _, _ = np.histogram2d(
            df_sell['price_distance_percent'] if len(df_sell) > 0 else [], 
            df_sell['time_distance'] if len(df_sell) > 0 else [], 
            bins=[price_edges, time_edges], 
            weights=df_sell['order_value'] if len(df_sell) > 0 else None
        )
        
        return {
            'all': heatmap_all.T,
            'buy': heatmap_buy.T,
            'sell': heatmap_sell.T,
            'xedges': xedges,
            'yedges': yedges,
            'price_range': (min_price, max_price)
        }

def process_symbol(symbol, date, order_dir, trade_dir, target_ts, param):
    """
    Process a single symbol at a specific timestamp
    
    Args:
        symbol (str): Stock symbol
        date (str): Date string
        order_dir (Path): Directory containing order data
        trade_dir (Path): Directory containing trade data
        target_ts (list): List of target timestamps
        param (dict): Parameters for OrderBookAnalyzer
        
    Returns:
        dict: Order book snapshots for this symbol
    """
    try:
        start_time = time.time()
        
        # Load order and trade data
        order_data = load_data(symbol, order_dir)
        trade_data = load_data(symbol, trade_dir)
        
        # Initialize OrderBookAnalyzer
        analyzer = OrderBookAnalyzer(symbol, date, order_data, trade_data, param)
        
        # Run analysis
        snapshots = analyzer.run()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processed {symbol} in {processing_time:.2f} seconds, got {len(snapshots)} snapshots")
        
        return snapshots
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def visualize_market_snapshot(combined_snapshot, output_path, ts_str):
    """
    Visualize a market-wide order book snapshot
    
    Args:
        combined_snapshot (dict): Combined snapshot data
        output_path (Path): Path to save visualizations
        ts_str (str): Timestamp string for naming
    """
    heatmap_all = combined_snapshot['all']
    heatmap_buy = combined_snapshot['buy']
    heatmap_sell = combined_snapshot['sell']
    xedges = combined_snapshot['xedges']
    yedges = combined_snapshot['yedges']
    price_range = combined_snapshot['price_range']
    
    # Create time and price ticks for better visualization
    price_ticks = np.arange(np.floor(price_range[0]), np.ceil(price_range[1]) + 1, 1)
    time_ticks = np.linspace(0, yedges[-1], 10)
    
    # Combined visualization
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(xedges, yedges, heatmap_all, norm=LogNorm(), cmap='viridis')
    
    plt.xticks(price_ticks, price_ticks.astype(int))
    plt.yticks(time_ticks, np.round(time_ticks, 1))
    
    plt.xlabel('Price Distance from Mid-price (%)')
    plt.ylabel('Time Distance from Current Time (minutes)')
    plt.title(f'Market-wide Order Book Heatmap at {ts_str}')
    
    # Add mid-price line
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Mid Price (Bid/Ask Boundary)')
    
    # Add bid/ask avg indicators if possible
    buy_avg_price = None
    sell_avg_price = None
    
    buy_totals = heatmap_buy.sum(axis=0)
    if buy_totals.sum() > 0:
        buy_indices = np.where(buy_totals > 0)[0]
        if len(buy_indices) > 0:
            buy_avg_price = np.average(
                [(xedges[i] + xedges[i+1])/2 for i in buy_indices], 
                weights=buy_totals[buy_indices]
            )
    
    sell_totals = heatmap_sell.sum(axis=0)
    if sell_totals.sum() > 0:
        sell_indices = np.where(sell_totals > 0)[0]
        if len(sell_indices) > 0:
            sell_avg_price = np.average(
                [(xedges[i] + xedges[i+1])/2 for i in sell_indices], 
                weights=sell_totals[sell_indices]
            )
    
    if buy_avg_price is not None:
        plt.axvline(x=buy_avg_price, color='blue', linestyle='--', alpha=0.7,
                   label=f'Average Bid Position ({buy_avg_price:.2f}%)')
    
    if sell_avg_price is not None:
        plt.axvline(x=sell_avg_price, color='red', linestyle='--', alpha=0.7,
                    label=f'Average Ask Position ({sell_avg_price:.2f}%)')
    
    plt.legend()
    plt.colorbar(label='Total Order Value')
    plt.tight_layout()
    plt.savefig(output_path / f'market_order_book_{ts_str}_combined.png', dpi=300)
    plt.close()
    
    # Separate bid and ask visualizations
    plt.figure(figsize=(16, 8))
    
    # Bid side
    plt.subplot(1, 2, 1)
    plt.pcolormesh(xedges, yedges, heatmap_buy, norm=LogNorm(), cmap='Blues')
    
    plt.xticks(price_ticks, price_ticks.astype(int))
    plt.yticks(time_ticks, np.round(time_ticks, 1))
    
    plt.xlabel('Price Distance from Mid-price (%)')
    plt.ylabel('Time Distance from Current Time (minutes)')
    plt.title(f'Market-wide Bid Orders at {ts_str}')
    
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Mid Price')
    plt.legend()
    plt.colorbar(label='Order Value')
    
    # Ask side
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xedges, yedges, heatmap_sell, norm=LogNorm(), cmap='Reds')
    
    plt.xticks(price_ticks, price_ticks.astype(int))
    plt.yticks(time_ticks, np.round(time_ticks, 1))
    
    plt.xlabel('Price Distance from Mid-price (%)')
    plt.ylabel('Time Distance from Current Time (minutes)')
    plt.title(f'Market-wide Ask Orders at {ts_str}')
    
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Mid Price')
    plt.legend()
    plt.colorbar(label='Order Value')
    
    plt.tight_layout()
    plt.savefig(output_path / f'market_order_book_{ts_str}_sides.png', dpi=300)
    plt.close()

def combine_snapshots(snapshots_list, ts):
    """
    Combine snapshots from multiple symbols for a specific timestamp
    
    Args:
        snapshots_list (list): List of snapshot dictionaries
        ts (int): Target timestamp
        
    Returns:
        dict: Combined snapshot
    """
    # Find all valid snapshots for this timestamp
    valid_snapshots = []
    for snapshots in snapshots_list:
        if ts in snapshots and snapshots[ts] is not None:
            valid_snapshots.append(snapshots[ts])
    
    if not valid_snapshots:
        return None
    
    # 固定价格范围为中间价的正负20%
    fixed_price_range = 20.0  # 百分比
    price_bin_width = 0.1  # 价格区间宽度为0.1%
    
    # 计算价格区间
    min_price = -fixed_price_range
    max_price = fixed_price_range
    price_bins = int((max_price - min_price) / price_bin_width) + 1
    std_xedges = np.linspace(min_price, max_price, price_bins)
    
    # 计算时间区间
    # 假设市场开盘时间是9:15
    market_open_time = pd.Timestamp(pd.to_datetime(ts, unit='ms').date()).replace(hour=9, minute=15)
    market_open_ts = int(market_open_time.timestamp() * 1000)
    
    # 计算从开盘到当前时间的分钟数
    elapsed_minutes = max(0, (ts - market_open_ts) / (60 * 1000))
    time_bin_width = 1.0  # 时间区间宽度为1分钟
    time_bins = int(elapsed_minutes / time_bin_width) + 1
    std_yedges = np.linspace(0, elapsed_minutes, time_bins)
    
    # 创建合并数组
    combined_all = np.zeros((time_bins - 1, price_bins - 1))
    combined_buy = np.zeros((time_bins - 1, price_bins - 1))
    combined_sell = np.zeros((time_bins - 1, price_bins - 1))
    
    # 对每个快照重新取样并合并
    for snapshot in valid_snapshots:
        # 创建新的直方图用于当前股票数据
        # 使用相同的区间定义但是数据来自原始快照
        orig_all = snapshot['all']
        orig_buy = snapshot['buy']
        orig_sell = snapshot['sell']
        orig_xedges = snapshot['xedges']
        orig_yedges = snapshot['yedges']
        
        # 提取原始数据的坐标点和权重
        # 为了重新取样，我们需要把直方图转换回点数据
        # 简化处理：根据原始网格的中心值重新创建数据点
        
        # 我们将把每个原始网格的值分配到新网格中
        # 这是一个简化的方法，对于每个网格单元，我们将其视为一个点数据
        
        # 创建新的直方图用于重新采样
        new_all = np.zeros((time_bins - 1, price_bins - 1))
        new_buy = np.zeros((time_bins - 1, price_bins - 1))
        new_sell = np.zeros((time_bins - 1, price_bins - 1))
        
        # 计算原始数据可用的最大时间索引
        max_time_idx = min(orig_all.shape[0], time_bins - 1)
        
        # 将原始数据的价格范围映射到新的价格范围
        orig_price_min = orig_xedges[0]
        orig_price_max = orig_xedges[-1]
        
        # 对每个原始数据的网格单元，找到它在新网格中的位置
        for t_idx in range(max_time_idx):
            # 获取原始时间值
            t_orig = (orig_yedges[t_idx] + orig_yedges[t_idx + 1]) / 2
            
            # 找到在新网格中的时间索引
            t_new_idx = min(int(t_orig / time_bin_width), time_bins - 2)
            
            for p_idx in range(orig_all.shape[1]):
                # 获取原始价格值
                p_orig = (orig_xedges[p_idx] + orig_xedges[p_idx + 1]) / 2
                
                # 将原始价格值限制在我们的固定范围内
                if p_orig < min_price or p_orig > max_price:
                    continue
                
                # 找到在新网格中的价格索引
                p_new_idx = min(int((p_orig - min_price) / price_bin_width), price_bins - 2)
                
                # 将原始值加到新网格中
                new_all[t_new_idx, p_new_idx] += orig_all[t_idx, p_idx]
                new_buy[t_new_idx, p_new_idx] += orig_buy[t_idx, p_idx]
                new_sell[t_new_idx, p_new_idx] += orig_sell[t_idx, p_idx]
        
        # 将重新采样的数据添加到合并结果中
        combined_all += new_all
        combined_buy += new_buy
        combined_sell += new_sell
    
    return {
        'all': combined_all,
        'buy': combined_buy,
        'sell': combined_sell,
        'xedges': std_xedges,
        'yedges': std_yedges,
        'price_range': (min_price, max_price)
    }

def main(date, time_interval, output_dir, n_workers=4, price_range=0.01):
    """
    Main function to analyze order books across all symbols
    
    Args:
        date (str): Date string in format 'YYYY-MM-DD'
        time_interval (dict): Time interval parameters (e.g., {'minutes': 15})
        output_dir (str): Directory to save results
        n_workers (int): Number of parallel workers
        price_range (float): Price range as percentage of mid price
    """
    # Setup directories
    path_config = load_path_config(project_dir)
    trade_dir = Path(path_config['trade']) / date
    order_dir = Path(path_config['order']) / date
    
    # Create output directory
    output_path = Path(output_dir) / date
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get available symbols
    trade_symbols = list_symbols_in_folder(trade_dir)
    order_symbols = list_symbols_in_folder(order_dir)
    symbols = list(set(trade_symbols) & set(order_symbols))
    
    print(f"Found {len(symbols)} symbols for date {date}")
    
    # Generate timestamps for the day
    date_dt = datetime.strptime(date, '%Y-%m-%d')
    timestamps = get_a_share_intraday_time_series(date_dt, time_interval)
    
    print(f"Processing {len(timestamps)} timestamps at {time_interval} intervals")
    
    # Prepare parameters
    param = {
        'target_ts': time_interval,
        'view_infos': {'': {'price_range': price_range}},
        'ind_cates': [],
        'indxview_count': {},
        'factor_idx_mapping': {}
    }
    
    # Process symbols in parallel
    process_func = partial(
        process_symbol,
        date=date,
        order_dir=order_dir,
        trade_dir=trade_dir,
        target_ts=timestamps,
        param=param
    )
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for symbol in symbols:
            results.append(executor.submit(process_func, symbol))
    
    # Collect results
    snapshots_list = []
    for future in results:
        try:
            snapshots = future.result()
            if snapshots:
                snapshots_list.append(snapshots)
        except Exception as e:
            print(f"Error collecting result: {e}")
    
    print(f"Collected snapshots from {len(snapshots_list)} symbols")
    
    # Combine and visualize snapshots for each timestamp
    for ts in timestamps:
        print(f"Combining and visualizing timestamp: {ts}")
        timestamp_str = pd.to_datetime(ts, unit='ms').strftime('%H%M%S')
        
        # Combine snapshots
        combined_snapshot = combine_snapshots(snapshots_list, ts)
        
        if combined_snapshot is None:
            print(f"No valid data for timestamp {ts}")
            continue
        
        # Save combined snapshot
        np.savez(
            output_path / f'market_order_book_{timestamp_str}.npz',
            **combined_snapshot
        )
        
        # Visualize combined snapshot
        visualize_market_snapshot(combined_snapshot, output_path, timestamp_str)
        
        # Clear memory
        gc.collect()
    
    print(f"Analysis completed. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze market-wide order book snapshots")
    parser.add_argument("--date", required=True, help="Date in format YYYY-MM-DD")
    parser.add_argument("--interval", type=int, default=15, help="Time interval in minutes")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--price-range", type=float, default=1.0, help="Price range as percentage of mid price")
    
    args = parser.parse_args()
    
    # Convert interval to param dict
    time_interval = {"minutes": args.interval}
    
    main(args.date, time_interval, args.output, args.workers, args.price_range)