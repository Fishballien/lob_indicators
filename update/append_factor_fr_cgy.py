import os
os.environ["POLARS_MAX_THREADS"] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
from datetime import datetime, timedelta
from multiprocessing import Pool

import polars as pl
from loguru import logger

from cccTools import safe_to_parquet_pl

def check_overlap_days(df_old: pl.DataFrame, df_inc: pl.DataFrame, overlap_days: int, max_err: float) -> tuple[bool, str]:
    """
    检查两个任务：
    1. df_old的最后的overlap_days的数据, 是否能在df_inc里找到。
    2. 如果能找到, 检查这两个重叠部分的数值是否小于max_err。

    Args:
        df_old (pl.DataFrame): 旧数据表。
        df_inc (pl.DataFrame): 新数据表。
        overlap_days (int): 要检查的重叠天数。
        max_err (float): 最大允许误差。

    Returns:
        bool: 如果满足条件，返回True，否则返回False。
    """
    # Step 1: 确保两个数据框按时间戳排序
    df_old = df_old.sort("timestamp")
    df_inc = df_inc.sort("timestamp")

    # Step 2: 提取df_old的最后overlap_days的数据
    last_timestamp = df_old[-1, "timestamp"]
    overlap_start = last_timestamp - timedelta(days=overlap_days)

    old_overlap = df_old.filter(pl.col("timestamp") >= overlap_start)

    if old_overlap.is_empty():
        raise ValueError("Old DataFrame does not have sufficient data for the specified overlap_days.")

    # Step 3: 检查df_inc是否包含所有重叠的时间戳
    new_overlap = df_inc.filter(
        (pl.col("timestamp") >= overlap_start) & (pl.col("timestamp") <= last_timestamp)
    )
    
    if new_overlap.height != old_overlap.height:
        # 如果时间戳数量不匹配，说明重叠部分不完整
        return False, f"no enough {overlap_days} overlap days. old days: {old_overlap.height}, inc_days: {new_overlap.height}"

    # Step 4: 对齐重叠部分并比较数值差异
    old_overlap = old_overlap.sort("timestamp")
    new_overlap = new_overlap.sort("timestamp")

    # 提取数值列（排除timestamp列）
    factor_cols = [col for col in df_old.columns if col != "timestamp"]

    # 去掉nan, inf, 方便计算准确
    old_overlap = old_overlap.with_columns(
        pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
        .then(1.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in factor_cols
    )
    new_overlap = new_overlap.with_columns(
        pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
        .then(1.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in factor_cols
    )

    # # 检查是否所有差值均小于max_err
    # # 计算绝对百分比差异
    abs_percent_diff_df = old_overlap.select([
        ((pl.col(col) - new_overlap[col]).abs() / (pl.col(col).abs() + 1e-20)).alias(col)
        for col in factor_cols
    ])

    # 检查是否所有百分比差异均小于指定阈值
    exceeds_error = abs_percent_diff_df.select([pl.col(col).max() > max_err for col in factor_cols]).row(0)

    # 找出超过 max_err 的列名
    exceeding_columns = [factor_cols[i] for i, exceeded in enumerate(exceeds_error) if exceeded]

    # 如果任一列的最大值超过 max_err，返回 False 并打印超出误差的列名
    if exceeding_columns:
        return False, f"Overlap days error too large in columns: {', '.join(exceeding_columns)}"

    return True, "success"



def append_factor(old_factor_path: str, incremental_factor_path: str, 
                  output_factor_path: str,
                  overlap_days: int = 3,
                  max_err: float = 1e-6):
    """往旧因子文件追加incremental_factor。将新的部分追加到old_factor中, 并保存old_factor。
    因子文件的格式为[timestamp, symbol1, symbol2, ..., symboln]

    注意:
    1. 行: 时间轴为timestamp, old_factor的timestamp表示的最后3天的数据, 和incremental_factor有完全重叠的部分(最大误差小于max_err)。
    2. 列：列可能会增多, 应当拓展。列的顺序有可能不一样, 应当排序。

    Args:
        old_factor_path (str): 旧因子文件路径
        incremental_factor_path (str): 增量因子文件路径
        output_factor_path (str): 保存的文件路径
        overlap_days (int): 应当重叠最少3天
        max_err (float): 允许的最大误百分比

    Returns:
        tuple[bool, str]:
            bool: True表示有超过max_err的误差, False表示没有误差并成功追加
            str: 错误信息或空字符串
    """
    # Step 0: Read the old and incremental data
    df_old = pl.read_parquet(old_factor_path)
    df_inc = pl.read_parquet(incremental_factor_path)
    df_old = df_old.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    df_inc = df_inc.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    
    # Ensure timestamp is sorted
    df_old = df_old.sort("timestamp")
    df_inc = df_inc.sort("timestamp")
    
    # Step1: union columns
    old_columns = set(df_old.columns)
    inc_columns = set(df_inc.columns)
    missing_in_old = inc_columns - old_columns
    missing_in_inc = old_columns - inc_columns

    if missing_in_old:
        df_old = df_old.with_columns([pl.lit(None).cast(pl.Float32).alias(col) for col in missing_in_old])
    if missing_in_inc:
        df_inc = df_inc.with_columns([pl.lit(None).cast(pl.Float32).alias(col) for col in missing_in_inc])
    # 按列名排序
    common_columns = [col for col in df_old.columns if col != "timestamp"]
    common_columns.sort()
    df_old = df_old.select(["timestamp"] + common_columns)
    df_inc = df_inc.select(["timestamp"] + common_columns)

    # Step2: 检查overlap
    valid_overlap, info = check_overlap_days(df_old=df_old, df_inc=df_inc, 
                                             overlap_days=overlap_days, max_err=max_err)
    if not valid_overlap:
        logger.error(f"Err occurred while processing {old_factor_path}, {incremental_factor_path}")
        logger.error(info)
        # raise
        return

    # Step 2: Append new (non-overlapping) data from incremental to old and save
    df_inc_new = df_inc.filter(pl.col("timestamp") > df_old[-1, "timestamp"])
    
    # 拼接并保存
    df_new = pl.concat([df_old, df_inc_new], how="vertical").unique(subset=["timestamp"], keep="first").sort("timestamp")
    safe_to_parquet_pl(df_new, output_factor_path)

def construct_tasks(historical_dir: str, incremental_dir: str, updated_dir: str) -> list[tuple]:
    historical_files = set(os.listdir(historical_dir))  # 转为集合
    incremental_files = set(os.listdir(incremental_dir))  # 转为集合
    updated_files = os.listdir(updated_dir)  # 可选，不需要转为集合
    
    # 1. historical 和 incremental应当相同
    if historical_files != incremental_files:
        # 找出差异
        only_in_historical = historical_files - incremental_files
        only_in_incremental = incremental_files - historical_files

        # 抛出详细的错误信息
        raise ValueError(
            f"Historical files not equal to incremental files:\n"
            f"In historical but not in incremental: {only_in_historical}\n"
            f"In incremental but not in historical: {only_in_incremental}"
        )

    # 2. 如果updated_dir文件数量小于historical_files, 直接构造全部任务
    if len(updated_files) < len(historical_files):
        res = []
        for factor_filename in historical_files:
            old_factor_path = os.path.join(historical_dir, factor_filename)
            inc_factor_path = os.path.join(incremental_dir, factor_filename)
            upd_factor_path = os.path.join(updated_dir, factor_filename)
            res.append((old_factor_path, inc_factor_path, upd_factor_path))
        return res
    # 3. 如果文件数量相同, 说明更新过
    elif len(updated_files) == len(historical_files):
        res = []
        for factor_filename in historical_files:
            old_factor_path = os.path.join(updated_dir, factor_filename)
            inc_factor_path = os.path.join(incremental_dir, factor_filename)
            upd_factor_path = os.path.join(updated_dir, factor_filename)
            res.append((old_factor_path, inc_factor_path, upd_factor_path))
        return res
    # 4. 如果文件数量不同, 有奇怪bug
    else:
        raise ValueError("来debug")


def process_task(task: tuple):
    """处理单个因子合并任务"""
    old_factor_path, inc_factor_path, upd_factor_path = task
    append_factor(
            old_factor_path=old_factor_path,
            incremental_factor_path=inc_factor_path,
            output_factor_path=upd_factor_path
        )
    return True, upd_factor_path
    try:
        append_factor(
            old_factor_path=old_factor_path,
            incremental_factor_path=inc_factor_path,
            output_factor_path=upd_factor_path
        )
        return True, upd_factor_path
    except Exception as e:
        logger.error(f"Failed to process {upd_factor_path}: {e}")
        return False, upd_factor_path

def append_factors_main_R2():
    historical_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/RunningFactors/historical_factors/LOB_2024-12-13_valid0.2_R2'
    incremental_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/RunningFactors/incremental_factors/LOB_2024-12-13_valid0.2_R2'
    updated_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/LOB_2024-12-13_valid0.2_R2'

    # 构建任务列表
    task_list = construct_tasks(historical_dir, incremental_dir, updated_dir)

    # 任务总数
    total_tasks = len(task_list)
    logger.info(f"Total tasks: {total_tasks}")

    # 多进程处理
    num_processes = 32  # 使用所有可用的 CPU 核心
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_task, task_list), 1):
            success, path = result
            if success:
                logger.success(f"Finished {i}/{total_tasks}: {path}")
            else:
                logger.error(f"Task {i}/{total_tasks} failed: {path}")
                
def append_factors_main_R1():
    historical_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/RunningFactors/historical_factors/LOB_2024-12-13_valid0.2_R1'
    incremental_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/RunningFactors/incremental_factors/LOB_2024-12-13_valid0.2_R1'
    updated_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/LOB_2024-12-13_valid0.2_R1'

    # 构建任务列表
    task_list = construct_tasks(historical_dir, incremental_dir, updated_dir)

    # 任务总数
    total_tasks = len(task_list)
    logger.info(f"Total tasks: {total_tasks}")

    # 多进程处理
    num_processes = 32  # 使用所有可用的 CPU 核心
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_task, task_list), 1):
            success, path = result
            if success:
                logger.success(f"Finished {i}/{total_tasks}: {path}")
            else:
                logger.error(f"Task {i}/{total_tasks} failed: {path}")
def append_factors_main():
    append_factors_main_R1()
    append_factors_main_R2()
    
def merge_test():
    historical_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/LOB_2024-12-19_f32_R1'
    incremental_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/LOB_2024-12-19_f32_R1_append'
    updated_dir = '/Data/guangyao/FactorDataMerged/LimitOrderBook/1'

    # 构建任务列表
    task_list = construct_tasks(historical_dir, incremental_dir, updated_dir)

    # 任务总数
    total_tasks = len(task_list)
    logger.info(f"Total tasks: {total_tasks}")

    # 多进程处理
    num_processes = 32  # 使用所有可用的 CPU 核心
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_task, task_list), 1):
            success, path = result
            if success:
                logger.success(f"Finished {i}/{total_tasks}: {path}")
            else:
                logger.error(f"Task {i}/{total_tasks} failed: {path}")

if __name__ == "__main__":
    merge_test()