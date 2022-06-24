@ scene_transfer_l5kit:
	python examples/adaptation/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_ft_batches' --strategy 'all' --lr 1e-4
	python examples/adaptation/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_layer11_batches' --strategy 'layer' --lr 3e-4
	python examples/adaptation/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_mosa_batches' --strategy 'mosa' --lr 3e-3
	python examples/adaptation/finetune.py --ratio 0.005 --step 5 -e 250 -ev 25 --output 'reproduce_norm_batches' --strategy 'norm' --lr 1e-4
