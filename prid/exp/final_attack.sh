cd ../
CUDA_VISIBLE_DEVICES=0 \
nohup python -u apply_attack_ablation.py \
    --train_feature ./attack_set/train_semi_test/attack_input.csv \
    --train_label ./attack_set/train_semi_test/attack_labels.csv \
    --victim_feature ./attack_set/semi_test/victim_input.csv \
    --victim_label ./attack_set/semi_test/victim_labels.csv \
    --feature_1 1 \
    --feature_2 1 \
    --feature_3 0 \
    --feature_4 0 \
    --feature_5 0 \
    --attack_name final_1 \
    --lr=1e-4 \
    --epochs=300 \
> ./saved_logs/final_attack_1.log 2>&1 &

nohup python -u apply_attack_ablation.py \
    --train_feature ./attack_set/train_semi_test/attack_input.csv \
    --train_label ./attack_set/train_semi_test/attack_labels.csv \
    --victim_feature ./attack_set/semi_test/victim_input.csv \
    --victim_label ./attack_set/semi_test/victim_labels.csv \
    --feature_1 1 \
    --feature_2 1 \
    --feature_3 0 \
    --feature_4 0 \
    --feature_5 0 \
    --attack_name final_2 \
    --lr=1e-4 \
    --epochs=300 \
> ./saved_logs/final_attack_2.log 2>&1 &

nohup python -u apply_attack_ablation.py \
    --train_feature ./attack_set/train_semi_test/attack_input.csv \
    --train_label ./attack_set/train_semi_test/attack_labels.csv \
    --victim_feature ./attack_set/semi_test/victim_input.csv \
    --victim_label ./attack_set/semi_test/victim_labels.csv \
    --feature_1 1 \
    --feature_2 1 \
    --feature_3 0 \
    --feature_4 0 \
    --feature_5 0 \
    --attack_name final_3 \
    --lr=1e-4 \
    --epochs=300 \
> ./saved_logs/final_attack_3.log 2>&1 &

nohup python -u apply_attack_ablation.py \
    --train_feature ./attack_set/train_semi_test/attack_input.csv \
    --train_label ./attack_set/train_semi_test/attack_labels.csv \
    --victim_feature ./attack_set/semi_test/victim_input.csv \
    --victim_label ./attack_set/semi_test/victim_labels.csv \
    --feature_1 1 \
    --feature_2 1 \
    --feature_3 0 \
    --feature_4 0 \
    --feature_5 0 \
    --attack_name final_4 \
    --lr=1e-4 \
    --epochs=300 \
> ./saved_logs/final_attack_4.log 2>&1 &

nohup python -u apply_attack_ablation.py \
    --train_feature ./attack_set/train_semi_test/attack_input.csv \
    --train_label ./attack_set/train_semi_test/attack_labels.csv \
    --victim_feature ./attack_set/semi_test/victim_input.csv \
    --victim_label ./attack_set/semi_test/victim_labels.csv \
    --feature_1 1 \
    --feature_2 1 \
    --feature_3 0 \
    --feature_4 0 \
    --feature_5 0 \
    --attack_name final_5 \
    --lr=1e-4 \
    --epochs=300 \
> ./saved_logs/final_attack_5.log 2>&1 &