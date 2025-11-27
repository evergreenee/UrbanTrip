#!/bin/bash

# --- 1. 配置 (根据你的日志自动填写) ---

# 原始文件的完整路径
ORIGINAL_FILE_PATH="chinatravel/evaluation/default_splits/tpc_phase1.txt"

# Python脚本期望文件所在的目录 (从你的错误日志中提取)
WORK_DIR="chinatravel/evaluation/default_splits"

# 原始文件的"词根" (不含.txt)
ORIGINAL_STEM="tpc_phase1"

# 我们将要创建的新文件的"词根"前缀
# 最终文件会像: tpc_phase1_part_00.txt, tpc_phase1_part_01.txt
NEW_STEM_PREFIX="${ORIGINAL_STEM}_part_"

# 并行配置
LINES_PER_FILE=10
MAX_JOBS=30

# Python命令
CMD_PREFIX="python run_exp.py"
CMD_ARGS="--agent TPCAgent --llm TPCLLM --oracle_translation --skip 1"


# --- 2. 脚本核心逻辑 ---

echo "--- 步骤 1: 清理可能存在的旧分割文件 ---"
# trap命令确保即使脚本被中断(Ctrl+C)，清理也会尝试执行
# '|| true' 确保在找不到文件时脚本不会因错误而退出
trap "echo '--- 捕获到退出信号, 执行清理... ---'; find '$WORK_DIR' -name '${NEW_STEM_PREFIX}*.txt' -delete || true; echo '清理完毕。'; exit" INT TERM EXIT

echo "清理 ${WORK_DIR}/${NEW_STEM_PREFIX}*.txt ..."
find "$WORK_DIR" -name "${NEW_STEM_PREFIX}*.txt" -delete || true


echo "--- 步骤 2: 在Python期望的目录中创建带.txt后缀的分割文件 ---"
# --additional-suffix=.txt : 确保生成的文件名是 part_00.txt
# 我们在WORK_DIR目录中, 以NEW_STEM_PREFIX为前缀创建文件
split -l "$LINES_PER_FILE" -d "$ORIGINAL_FILE_PATH" \
      "${WORK_DIR}/${NEW_STEM_PREFIX}" --additional-suffix=.txt

echo "分割文件创建完毕。"


echo "--- 步骤 3: 查找所有新创建的'词根'(Stems) ---"
# 我们需要一个列表, 内容是:
# tpc_phase1_part_00
# tpc_phase1_part_01
# ...
# find: 查找所有匹配的文件
# sed 's|^.*/||': 移除路径 (e.g., .../default_splits/tpc_phase1_part_00.txt -> tpc_phase1_part_00.txt)
# sed 's|.txt$||': 移除.txt后缀 (e.g., tpc_phase1_part_00.txt -> tpc_phase1_part_00)
STEM_LIST=$(find "$WORK_DIR" -name "${NEW_STEM_PREFIX}*.txt" | sed 's|^.*/||' | sed 's|.txt$||')

if [ -z "$STEM_LIST" ]; then
    echo "错误: 未能创建或找到任何分割文件。请检查路径和权限。"
    exit 1
fi


echo "--- 步骤 4: 使用 xargs 并行执行 (只传递'词根') ---"
# xargs 将会接收 "tpc_phase1_part_00", "tpc_phase1_part_01" 等
# 并执行: python run_exp.py --splits tpc_phase1_part_00 ...
# Python 内部会将其拼接为: .../default_splits/tpc_phase1_part_00.txt
# 这个文件是存在的! 成功!
echo "$STEM_LIST" | xargs -P "$MAX_JOBS" -I {} \
$CMD_PREFIX --splits {} $CMD_ARGS


echo "--- 步骤 5: 所有任务执行完毕 ---"
# 'trap' 命令会在脚本正常退出时自动执行清理
# 我们不再需要'temp_splits'目录, 因为文件是原地创建和删除的