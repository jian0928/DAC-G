#!/bin/bash

set -e

# --- 配置 ---
# 定义需要循环处理的数据集列表
DATASETS=(
   "CaiHarmless"
    "HarmlessHH"
   "HelpfulHH"
   "HelpSteer"
    "HHrlhf"
    "SafeRLHF"
)
BASE_DATASET_DIR="./datasets"

echo "🚀 开始执行所有 Python 脚本，循环处理 ${#DATASETS[@]} 个数据集..."
echo "=========================================="

# --- 主循环 ---
for DATASET_NAME in "${DATASETS[@]}"
do
    echo "---"
    echo "Processing Dataset: **${DATASET_NAME}**"
    echo "---"

    # 根据数据集名称动态构建路径
    WEAK_SUP_DATASET_PATH="${BASE_DATASET_DIR}/${DATASET_NAME}/train_trans_train_5000.json"
    ANN_DATASET_PATH="${BASE_DATASET_DIR}/${DATASET_NAME}/train_trans_ann_5000.json"
    ANN_DATASET_NAME="train_trans_ann_5000.json" # annotate.py 中的文件名称参数

    # 1. 运行 create_weak_supervior.py (弱监督模型训练)
    echo "▶️ 1/5: create_weak_supervior.py"
    python create_weak_supervior.py \
        --weak_model_name 'Qwen3-4B' \
        --dataset_name "${DATASET_NAME}" \
        --seed  \
        --quantization_bit  \
        --num_train_epochs  \
        --per_device_train_batch_size  \
        --gradient_accumulation_steps  \
        --learning_rate  \
        --warmup_ratio  \
        --logging_steps  \
        --beta  \
        --max_length  \
        --max_prompt_length  \
        --r  \
        --lora_alpha  \
        --lora_dropout 
    echo "✅ create_weak_supervior.py for ${DATASET_NAME} 运行完成"

    # 2. 运行 extract_key.py (提取关键信息)
    echo "▶️ 2/5: extract_key.py"
    python extract_key.py \
        --dataset_name "${DATASET_NAME}"

    echo "✅ extract_key.py for ${DATASET_NAME} 运行完成"

    # 3. 运行 detect.py (检测)
    echo "▶️ 3/5: detect.py"
    python detect.py \
        --dataset_name "${DATASET_NAME}"

    echo "✅ detect.py for ${DATASET_NAME} 运行完成"

    # 4. 运行 annotate.py (标注)
    echo "▶️ 4/5: annotate.py"
    python annotate.py \
        --weak_model_name "Qwen3-4B" \
        --train_dataset_name "${DATASET_NAME}" \
        --quantization_bit  \
        --beta  \
        --max_length  \
        --max_prompt_length 

    echo "✅ annotate.py for ${DATASET_NAME} 运行完成"

    python baseline_STF_strong_student.py \
        --strong_model_name 'Qwen3-8B' \
        --dataset_name "${DATASET_NAME}" \
        --seed  \
        --quantization_bit  \
        --num_train_epochs  \
        --per_device_train_batch_size  \
        --gradient_accumulation_steps  \
        --learning_rate  \
        --warmup_ratio  \
        --logging_steps  \
        --beta  \
        --max_length  \
        --max_prompt_length  \
        --r  \
        --lora_alpha  \
        --lora_dropout 
    echo "✅ baseline_STF_strong_student.py for ${DATASET_NAME} 运行完成"
    echo "------------------------------------------"
    echo "完成数据集: **${DATASET_NAME}** 的所有步骤"
    echo "------------------------------------------"

done

echo "=========================================="
echo "🎉 所有脚本已对所有 ${#DATASETS[@]} 个数据集循环运行完毕！"