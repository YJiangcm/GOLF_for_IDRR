#!/usr/bin/python3
# python3 run.py --seed 0 --cuda 1 --model_name_or_path bert-base-uncased
# sleep 1m
# python3 run.py --seed 1 --cuda 1 --model_name_or_path bert-base-uncased
# sleep 1m
# python3 run.py --seed 2 --cuda 1 --model_name_or_path bert-base-uncased
# sleep 1m
# python3 run.py --seed 3 --cuda 1 --model_name_or_path bert-base-uncased
# sleep 1m
# python3 run.py --seed 4 --cuda 1 --model_name_or_path bert-base-uncased


python3 run.py --seed 0 --cuda 1 --num_gcn_layer 2
sleep 1m
python3 run.py --seed 1 --cuda 1 --num_gcn_layer 2
sleep 1m
python3 run.py --seed 2 --cuda 1 --num_gcn_layer 2
sleep 1m
python3 run.py --seed 3 --cuda 1 --num_gcn_layer 2
sleep 1m
python3 run.py --seed 4 --cuda 1 --num_gcn_layer 2

# python3 run.py --seed 0 --cuda 1 --num_gcn_layer 1
# sleep 1m
# python3 run.py --seed 1 --cuda 1 --num_gcn_layer 1
# sleep 1m
# python3 run.py --seed 2 --cuda 1 --num_gcn_layer 1
# sleep 1m
# python3 run.py --seed 3 --cuda 1 --num_gcn_layer 1
# sleep 1m
# python3 run.py --seed 4 --cuda 1 --num_gcn_layer 1

###################################################
# sleep 1m
# python3 run.py --seed 0 --cuda 1 --lambda_local 0.1
# sleep 1m
# python3 run.py --seed 1 --cuda 1 --lambda_local 0.1
# sleep 1m
# python3 run.py --seed 2 --cuda 1 --lambda_local 0.1
# sleep 1m
# python3 run.py --seed 3 --cuda 1 --lambda_local 0.1
# sleep 1m
# python3 run.py --seed 4 --cuda 1 --lambda_local 0.1

# ##################################################
# sleep 1m
# python3 run.py --seed 0 --cuda 1 --lambda_local 0.5
# sleep 1m
# python3 run.py --seed 1 --cuda 1 --lambda_local 0.5
# sleep 1m
# python3 run.py --seed 2 --cuda 1 --lambda_local 0.5
# sleep 1m
# python3 run.py --seed 3 --cuda 1 --lambda_local 0.5
# sleep 1m
# python3 run.py --seed 4 --cuda 1 --lambda_local 0.5

# ###################################################
# sleep 1m
# python3 run.py --seed 0 --cuda 1 --lambda_local 2.0
# sleep 1m
# python3 run.py --seed 1 --cuda 1 --lambda_local 2.0
# sleep 1m
# python3 run.py --seed 2 --cuda 1 --lambda_local 2.0
# sleep 1m
# python3 run.py --seed 3 --cuda 1 --lambda_local 2.0
# sleep 1m
# python3 run.py --seed 4 --cuda 1 --lambda_local 2.0