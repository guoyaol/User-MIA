这里，我们把
group-level attack
改造使用shadow model提供训练集

流程：
0. ./datasets/Market-1501-2/shadow_attack/split_all.py

1.run_train_shadow & run_train_victim

2.generate_attack_train & generate_victim_train

3.apply_attack