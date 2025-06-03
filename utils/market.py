# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:03:07 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from enum import Enum


# %%
PX_MULTIPLIER = 10_000
MINIMUM_SIZE_FILTER = 1


# %%
class Action(Enum):
    A = 0
    D = 1
    T = 2
    

class Exchange(Enum):
    SH = 0
    SZ = 1
    
    
class Side(Enum):
    Bid = 0
    Ask = 1
    N = 2
    
    
class TradeDirection(Enum):
    AB = 0  # 主买 (Active Buy)
    PB = 1  # 被动买 (Passive Buy)
    AS = 2  # 主卖 (Active Sell)
    PS = 3  # 被动卖 (Passive Sell)
    N = 4   # 集合竞价 (Auction)
    
    
class DataType(Enum):
    Order = 0
    Trade = 1
    

class DefaultPx(Enum):
    Bid = 0
    Ask = int(1e13)
    

# %%
def get_exchange(symbol):
    if symbol.startswith('6') or symbol.startswith('5'):
        return 0
    else:
        return 1