#!/bin/bash

# 定义变量
#ind_ver_name="cc_top5_ver0"
ind_ver_name="Batch3_241126"
processor='IndicatorProcessorByL2Batch'
start_date='2021-01-01'
end_date='2024-09-30'
#start_date='2024-01-01'
#end_date='2021-01-30'
n_workers=150

# 创建日志目录（如果不存在）
log_dir="./.logs"
mkdir -p "$log_dir"

# 定义日志文件
log_file="$log_dir/${ind_ver_name}.log"

# 定义命令
cmd="stdbuf -oL python3 generate_indicator.py -indv $ind_ver_name -p $processor -sd $start_date -ed $end_date -nw $n_workers" # -rp

# 打印日志文件名
echo "Log file: $log_file"

# 打印命令
echo "Command: $cmd"

# 执行命令并将输出重定向到日志文件
eval "nohup $cmd" > "$log_file" 2>&1 &
