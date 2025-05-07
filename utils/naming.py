# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:57:39 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from itertools import product


# %%
def generate_factor_names(factor_category, param_dict):
    """
    生成因子名列表。
    
    参数:
        factor_category (str): 因子大类名。
        param_dict (dict): 参数字典，键为参数全名，值为参数值列表。
        
    返回:
        list: 因子名列表。
    """
    # 如果参数字典为空，直接返回因子大类名作为唯一因子名
    if not param_dict:
        return [factor_category]
    
    def get_unique_abbreviation(param_names):
        """
        根据参数名生成唯一的1-3字母缩写，避免重复。
        """
        abbreviations = {}
        used_abbrs = set()
        
        for name in param_names:
            abbr = None
            for i in range(1, len(name) + 1):  # 动态从1个字母到完整长度逐步尝试
                abbr = name[:i]  # 截取前i个字母
                if abbr not in used_abbrs:
                    used_abbrs.add(abbr)
                    break
            else:  # 如果全部截取都冲突，直接使用完整名字
                abbr = name
            abbreviations[name] = abbr
    
        # 检查是否有重复，并解决冲突
        for name, abbr in abbreviations.items():
            while list(abbreviations.values()).count(abbr) > 1:  # 如果有重复
                conflict_count = list(abbreviations.values()).count(abbr)
                abbr = f"{abbr}_{conflict_count}"  # 添加后缀使其唯一
                abbreviations[name] = abbr
    
        return abbreviations

    # 获取参数缩写
    param_abbreviations = get_unique_abbreviation(param_dict.keys())
    
    # 提取参数组合
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())
    param_combinations = product(*param_values)
    
    factor_names = []
    for combination in param_combinations:
        factor_name = factor_category  # 初始化因子名为因子大类
        for param_name, param_value in zip(param_names, combination):
            # if len(param_dict[param_name]) > 1:  # 参数值仅有一个时忽略参数名
            abbr = param_abbreviations[param_name]
            factor_name += f"_{abbr}{param_value}"
        factor_names.append(factor_name)
    
    return factor_names

# =============================================================================
# # 示例用法
# factor_category = "FactorA"
# param_dict = {
#     "alpha": [0.1, 0.2],
#     "beta": [1],
#     "gamma": [10, 20, 30],
#     "ganna": [10, 20, 30],
#     "gamna": [10, 20, 30],
#     "gamca": [10, 20, 30],
# }
# 
# factor_names = generate_factor_names(factor_category, param_dict)
# print(factor_names)
# =============================================================================
