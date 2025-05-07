#!/bin/bash

# 定义变量
ind_ver_name="cc_top5_ver0"
n_workers=30

# 创建日志目录（如果不存在）
log_dir="./.logs"
mkdir -p "$log_dir"

# 定义日志文件
log_file="$log_dir/trans_${ind_ver_name}.log"

# 定义命令
cmd="python3 trans_to_cs.py -indv $ind_ver_name -nw $n_workers"

# 打印日志文件名
echo "Log file: $log_file"

# 打印命令
echo "Command: $cmd"

# 执行命令并将输出重定向到日志文件
eval "$cmd" &> "$log_file" &
