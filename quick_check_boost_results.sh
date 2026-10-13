#!/bin/bash
# 快速检查 boost 优化实验的结果

echo "================================================================================"
echo "快速结果检查：attack_boost 优化实验"
echo "================================================================================"

check_experiment() {
    local name=$1
    local dir=$2

    if [ ! -f "$dir/summary.csv" ]; then
        echo "⏳ $name: 还未完成"
        return
    fi

    echo ""
    echo "【$name】"
    echo "--------------------------------------------------------------------------------"

    # 提取 sm9rrs 的 ASR 结果
    awk -F',' 'NR==1 {
        for(i=1;i<=NF;i++) {
            if($i=="partition") p=i;
            if($i=="ratio") r=i;
            if($i=="final_acc") a=i;
            if($i=="target_asr") asr=i;
            if($i=="blacklisted") b=i;
        }
        print "分布          ratio    ASR      Accuracy  撤销"
        print "----------------------------------------------------------------"
    }
    NR>1 && $3=="sm9rrs" {
        printf "%-12s  %.1f    %5.1f%%   %5.1f%%    %3d\n", $p, $r, $asr*100, $a*100, $b
    }' "$dir/summary.csv"

    # 计算平均 ASR
    echo ""
    avg_asr=$(awk -F',' 'NR>1 && $3=="sm9rrs" && $4>0 {sum+=$55; count++} END {if(count>0) print sum/count*100; else print "N/A"}' "$dir/summary.csv")
    echo "平均 ASR (攻击场景): ${avg_asr}%"
}

# 检查三个实验
check_experiment "attack_boost=25 (最激进)" "outputs/cifar10_v9_boost25_test"
check_experiment "attack_boost=20 (中等)" "outputs/cifar10_v9_boost20_test"
check_experiment "平衡策略 (boost=20 + threshold=2.0)" "outputs/cifar10_v9_balanced_test"

echo ""
echo "================================================================================"
echo "【对比基准】"
echo "================================================================================"
echo "原始配置 (boost=15):"
echo "  seed=2026101301: IID 0.4 ASR=80.5%, 平均=33.1%"
echo "  seed=2026101302: IID 0.4 ASR=10.5%, 平均=~15%"
echo ""
echo "竞争对手 (多方法对比):"
echo "  Krum:     平均 ASR=10.0%"
echo "  AlignIns: 平均 ASR=11.4%"
echo "  Ding13:   平均 ASR=16.7%"
echo ""
echo "【成功标准】"
echo "  最低目标: IID 0.4 < 30%, 平均 < 20%"
echo "  理想目标: IID 0.4 < 15%, 平均 < 12%"
echo "  完美目标: 所有 < 10%, 平均 < 10%"
echo "================================================================================"
