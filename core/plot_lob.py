# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 22:19:57 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sns


def visualize_order_book(view_dataset):
    # 从字典中提取数据
    on_ts_org = view_dataset['on_ts_org']
    on_side = view_dataset['on_side']
    on_px = view_dataset['on_px']
    on_qty_remain = view_dataset['on_qty_remain']
    best_px = view_dataset['best_px']
    ts = view_dataset['ts']
    
    # 计算中间价
    mid_price = (best_px[0] + best_px[1]) / 2
    
    # 计算时间距离（分钟）
    time_distance = (ts - on_ts_org) / 1000 / 60
    
    # 计算价格距离中间价的百分比
    price_distance_percent = (on_px - mid_price) / mid_price * 100
    
    # 计算每个订单的金额（价格 * 剩余数量）
    order_value = on_px * on_qty_remain
    
    # 创建数据框
    df = pd.DataFrame({
        'time_distance': time_distance,
        'price_distance_percent': price_distance_percent,
        'order_value': order_value,
        'side': on_side
    })
    
    # 确定价格距离的边界，确保能以1%为间隔
    min_price_dist = np.floor(min(price_distance_percent))
    max_price_dist = np.ceil(max(price_distance_percent))
    # 创建更细粒度的价格区间用于绘图，但标签仍然是1%间隔
    price_bins = np.linspace(min_price_dist, max_price_dist, 100)  # 更细的粒度，100个区间
    price_ticks = np.arange(min_price_dist, max_price_dist + 1, 1)  # 每1%一个标签
    
    # 定义时间的分箱范围，增加区间数以提高精度
    time_bins = np.linspace(0, max(time_distance), 100)  # 更细的粒度
    # 创建合适的时间标签，确保均匀分布
    time_ticks_count = 10  # 控制显示的标签数量
    time_ticks = np.linspace(0, max(time_distance), time_ticks_count)
    
    # 创建二维直方图以聚合数据
    heatmap_data, xedges, yedges = np.histogram2d(
        df['price_distance_percent'], 
        df['time_distance'], 
        bins=[price_bins, time_bins], 
        weights=df['order_value']
    )
    
    # 转置以便时间在y轴，价格距离在x轴
    heatmap_data = heatmap_data.T
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    
    # 使用对数标准化以便更好地显示值的分布
    # 确保使用正确的参数组合，确保色图正确渲染
    hm = plt.pcolormesh(xedges, yedges, heatmap_data, 
                norm=LogNorm(), cmap='viridis')
    
    # 手动设置X轴刻度和标签（每1%一个标签）
    plt.xticks(price_ticks, price_ticks.astype(int))
    
    # 手动设置Y轴刻度和标签，确保均匀分布
    plt.yticks(time_ticks, np.round(time_ticks, 1))
    
    plt.xlabel('Price Distance from Mid-price (%)')
    plt.ylabel('Time Distance from Current Time (minutes)')
    plt.title('Order Book Heatmap: Time vs Price Distance, Color Represents Order Value')
    
    # 添加买卖方向的标记
    buy_avg_price = np.mean(price_distance_percent[on_side == 0])
    sell_avg_price = np.mean(price_distance_percent[on_side == 1])
    
    # 添加中间价线
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Mid Price (Bid/Ask Boundary)')
    
    # 添加平均买卖单位置
    plt.axvline(x=buy_avg_price, color='blue', linestyle='--', alpha=0.7, label=f'Average Bid Position ({buy_avg_price:.2f}%)')
    plt.axvline(x=sell_avg_price, color='red', linestyle='--', alpha=0.7, label=f'Average Ask Position ({sell_avg_price:.2f}%)')
    plt.legend()
    
    plt.colorbar(hm, label='Total Order Value')
    plt.tight_layout()
    plt.show()
    
    # 分别查看买单和卖单的分布
    plt.figure(figsize=(12, 6))
    
    # 买单热力图
    plt.subplot(1, 2, 1)
    df_buy = df[df['side'] == 0]
    heatmap_buy, xedges_buy, yedges_buy = np.histogram2d(
        df_buy['price_distance_percent'], 
        df_buy['time_distance'], 
        bins=[price_bins, time_bins], 
        weights=df_buy['order_value']
    )
    heatmap_buy = heatmap_buy.T
    
    # 使用plt.pcolormesh替代sns.heatmap以避免潜在的cmap问题
    hm_buy = plt.pcolormesh(xedges_buy, yedges_buy, heatmap_buy, 
                  norm=LogNorm(), cmap='Blues')
    
    # 手动设置X轴刻度和标签
    plt.xticks(price_ticks, price_ticks.astype(int))
    
    # 手动设置Y轴刻度和标签
    plt.yticks(time_ticks, np.round(time_ticks, 1))
    
    plt.xlabel('Price Distance from Mid-price (%)')
    plt.ylabel('Time Distance from Current Time (minutes)')
    plt.title('Bid Orders Heatmap')
    
    # 添加中间价线
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Mid Price')
    plt.legend()
    
    plt.colorbar(hm_buy, label='Order Value')
    
    # 卖单热力图
    plt.subplot(1, 2, 2)
    df_sell = df[df['side'] == 1]
    heatmap_sell, xedges_sell, yedges_sell = np.histogram2d(
        df_sell['price_distance_percent'], 
        df_sell['time_distance'], 
        bins=[price_bins, time_bins], 
        weights=df_sell['order_value']
    )
    heatmap_sell = heatmap_sell.T
    
    # 使用plt.pcolormesh替代sns.heatmap
    hm_sell = plt.pcolormesh(xedges_sell, yedges_sell, heatmap_sell, 
                  norm=LogNorm(), cmap='Reds')
    
    # 手动设置X轴刻度和标签
    plt.xticks(price_ticks, price_ticks.astype(int))
    
    # 手动设置Y轴刻度和标签
    plt.yticks(time_ticks, np.round(time_ticks, 1))
    
    plt.xlabel('Price Distance from Mid-price (%)')
    plt.ylabel('Time Distance from Current Time (minutes)')
    plt.title('Ask Orders Heatmap')
    
    # 添加中间价线
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Mid Price')
    plt.legend()
    
    plt.colorbar(hm_sell, label='Order Value')
    
    # 确保所有子图的布局完美
    plt.tight_layout()
    plt.show()

