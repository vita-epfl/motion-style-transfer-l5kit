# Ratio 0.008 --> 24 batches
# Ratio 0.007 --> 21 batches
# Ratio 0.006 --> 18 batches
# Ratio 0.005 --> 15 batches
# Ratio 0.004 --> 12 batches
# Ratio 0.003 --> 9 batches

@ ft_batches_c0:
	# python examples/dro/finetune.py --ratio 0.002 --output 'adapter_ds12_batches' --strategy 'adapter'
	# python examples/dro/finetune.py --ratio 0.002 --output 'ft_batches' --strategy 'all'
	# python examples/dro/finetune.py --ratio 0.002 --output 'ln_batches' --strategy 'norm'
	# python examples/dro/finetune.py --ratio 0.005 --output 'adapter_ds12_batches' --strategy 'adapter'
	# python examples/dro/finetune.py --ratio 0.005 --output 'ft_batches' --strategy 'all'
	# python examples/dro/finetune.py --ratio 0.005 --output 'ln_batches' --strategy 'norm'
	# python examples/dro/finetune.py --ratio 0.005 --output 'adapter_ds12_batches' --strategy 'adapter'
	# python examples/dro/finetune.py --ratio 0.005 --output 'ft_batches' --strategy 'all'
	# python examples/dro/finetune.py --ratio 0.005 --output 'layer0_batches' --strategy 'layer' --layer_num 0
	# python examples/dro/finetune.py --ratio 0.005 --output 'layer11_batches' --strategy 'layer' --layer_num 11
	# python examples/dro/finetune.py --ratio 0.005 --output 'layer5_batches' --strategy 'layer' --layer_num 5
	# python examples/dro/finetune.py --ratio 0.005 --output 'adapter_ds12_longer_batches' --strategy 'adapter'
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'memory_num20_batches' --strategy 'memory'
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.007 --output 'scratch_batches' --strategy 'scratch'


	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapter_ds24_config1_batches' --strategy 'adapter' --num_adapters 1
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.003 --output 'adapter_ds24_config1_batches' --strategy 'adapter' --num_adapters 1
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.007 --output 'adapter_ds24_config1_batches' --strategy 'adapter' --num_adapters 1
	CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'lora_r16_batches' --strategy 'lora'
	CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.003 --output 'lora_r16_batches' --strategy 'lora'

@ ft_batches_c1:
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapter_ds24_config3_batches' --strategy 'adapter' --num_adapters 3
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.003 --output 'adapter_ds24_config3_batches' --strategy 'adapter' --num_adapters 3
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.007 --output 'adapter_ds24_config3_batches' --strategy 'adapter' --num_adapters 3
	CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.007 --output 'lora_r16_batches' --strategy 'lora'



@ ft_batches_pred_c0:
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.012 --strategy 'adapter' --output 'cycs_adapter_ds24_batches'
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.012 --strategy 'all' --output 'cycs_ft_batches'
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.012 --strategy 'lora' --output 'cycs_lora_r16_batches'
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.3 --strategy 'all' --output 'cycs_lr5e5_ft_batches' --lr 5e-5
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.3 --strategy 'adapter' --output 'cycs_lr5e5_adap_batches' --lr 5e-5
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr5e4_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 5e-4 -s 42
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr2e4_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 2e-4 -s 42
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr5e5_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 5e-5 -s 42
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr1e3_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 42
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr3e3_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 42
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_ds12_lr1e3_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 42 --adapter_downsample 12
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_ds12_lr3e3_s42_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 42 --adapter_downsample 12
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_ds48_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420 --adapter_downsample 48
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_ds48_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420 --adapter_downsample 48
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.002 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.003 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.004 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.006 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.007 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420

	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.01 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.01 --output 'adapt_p0_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.015 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.015 --output 'adapt_p0_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 10 --output 'adapt_step10_p0_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 10 --output 'adapt_step10_p0_lr3e4_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-4 -s 420

	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'adapt_step2_p0_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 20 --output 'adapt_step20_p0_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'adapt_step2_p0_lr3e4_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-4 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 20 --output 'adapt_step20_p0_lr3e4_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-4 -s 420

	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'lora_step2_r8_p0_lr1e3_s420_batches' --strategy 'lora' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 10 --output 'lora_step10_r8_p0_lr1e3_s420_batches' --strategy 'lora' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 20 --output 'lora_step20_r8_p0_lr1e3_s420_batches' --strategy 'lora' --perturb 0.0 --lr 1e-3 -s 420

	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'lora_step2_r8_p0_lr3e3_s6_batches' --strategy 'lora' --perturb 0.0 --lr 3e-3 -s 6
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'lora_step2_r8_p0_lr3e3_s9_batches' --strategy 'lora' --perturb 0.0 --lr 3e-3 -s 9
	# Longer lora
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 1 --step 5 -e 15 -ev 1 --output 'lora_full_step5_p05_lr1e3_batches' --strategy 'lora' --perturb 0.5 --lr 1e-3
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 1 --step 5 -e 15 -ev 1 --output 'lora_full_step5_p05_lr1e4_batches' --strategy 'lora' --perturb 0.5 --lr 1e-4

	# Prediction 5 secs LORA
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py -e 50 -ev 5 --output '5sec_lora_step500_lr1e3_batches' --strategy 'lora' --lr 1e-3
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py -e 50 -ev 5 --output '5sec_lora_step500_lr3e4_batches' --strategy 'lora' --lr 3e-4
	# CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py -e 50 -ev 5 --output '5sec_lora_step500_lr3e3_batches' --strategy 'lora' --lr 3e-3
	CUDA_VISIBLE_DEVICES=0 python examples/prediction/finetune.py -e 15 -ev 1 --step 10 --output '5sec_ft_step10_lr1e4_batches' --strategy 'all' --lr 1e-4

@ ft_batches_pred_c1:
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.008 --strategy 'adapter' --output 'cycs_adapter_ds24_batches'
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.008 --strategy 'all' --output 'cycs_ft_batches'
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.008 --strategy 'lora' --output 'cycs_lora_r16_batches'
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.3 --strategy 'all' --output 'cycs_lr5e6_ft_batches' --lr 5e-6
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py --filter_type "cycs" --ratio 0.3 --strategy 'adapter' --output 'cycs_lr5e6_adap_batches' --lr 5e-6
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr5e4_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 5e-4 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr2e4_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 2e-4 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p05_lr5e5_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 5e-5 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_lr1e3_s42_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 42
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_lr3e3_s42_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-3 -s 42
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_ds12_lr1e3_s42_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 42 --adapter_downsample 12
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_ds12_lr3e3_s42_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-3 -s 42 --adapter_downsample 12
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_ds48_lr1e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 1e-3 -s 420 --adapter_downsample 48
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --output 'adapt_p0_ds48_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-3 -s 420 --adapter_downsample 48
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.002 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.003 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.004 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.006 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.007 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420

	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.01 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.01 --output 'adapt_p0_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.015 --output 'adapt_p05_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.5 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.015 --output 'adapt_p0_lr3e3_s420_batches' --strategy 'adapter' --perturb 0.0 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 10 --output 'ft_step10_p0_lr1e4_s420_batches' --strategy 'all' --perturb 0.0 --lr 1e-4 -s 420

	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'ft_step2_p0_lr1e4_s420_batches' --strategy 'all' --perturb 0.0 --lr 1e-4 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 10 --output 'ft_step10_p0_lr5e5_s420_batches' --strategy 'all' --perturb 0.0 --lr 5e-5 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 20 --output 'ft_step20_p0_lr5e5_s420_batches' --strategy 'all' --perturb 0.0 --lr 5e-5 -s 420

	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'lora_step2_r8_p0_lr3e3_s7_batches' --strategy 'lora' --perturb 0.0 --lr 3e-3 -s 7
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'lora_step2_r8_p0_lr3e3_s8_batches' --strategy 'lora' --perturb 0.0 --lr 3e-3 -s 8

	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 10 --output 'lora_step10_r8_p0_lr3e3_s420_batches' --strategy 'lora' --perturb 0.0 --lr 3e-3 -s 420
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 20 --output 'lora_step20_r8_p0_lr3e3_s420_batches' --strategy 'lora' --perturb 0.0 --lr 3e-3 -s 420

	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'ft_step2_p0_lr5e5_s7_batches' --strategy 'all' --perturb 0.0 --lr 5e-5 -s 7
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 2 --output 'ft_step2_p0_lr5e5_s7_batches' --strategy 'all' --perturb 0.0 --lr 5e-5 -s 7

	# Longer lora
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 1 --step 5 -e 15 -ev 1 --output 'lora_full_step5_p05_lr3e3_batches' --strategy 'lora' --perturb 0.5 --lr 3e-3
	# CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 1 --step 5 -e 15 -ev 1 --output 'lora_full_step5_p05_lr3e4_batches' --strategy 'lora' --perturb 0.5 --lr 3e-4

	# Prediction 5 secs FT
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py -e 50 -ev 2 --step 50 --output '5sec_ft_step50_lr1e4_batches' --strategy 'all' --lr 1e-4
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py -e 50 -ev 5 --output '5sec_ft_step500_lr1e5_batches' --strategy 'all' --lr 1e-5
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py -e 50 -ev 5 --output '5sec_lora_step500_lr1e4_batches' --strategy 'lora' --lr 1e-4
	# CUDA_VISIBLE_DEVICES=1 python examples/prediction/finetune.py -e 50 -ev 2 --step 50 --output '5sec_lora_step50_lr1e3_batches' --strategy 'lora' --lr 1e-3

	CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'debug_batches' --strategy 'mosa' --lr 3e-4

@ agent_dict:
	python scripts/categorize_agents.py --data_path scenes/train.zarr --output train_agent_metadata.csv
	python scripts/categorize_agents.py --data_path scenes/validate.zarr --output v

@ reproduce_2:
	CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_ft_batches' --strategy 'all' --lr 1e-4
	CUDA_VISIBLE_DEVICES=1 python examples/dro/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_layer11_batches' --strategy 'layer' --lr 3e-4

@ reproduce_1:
	CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_mosa_batches' --strategy 'mosa' --lr 3e-3
	CUDA_VISIBLE_DEVICES=0 python examples/dro/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_norm_batches' --strategy 'norm' --lr 1e-4
