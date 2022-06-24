**Scripts for reproducing Scene Style Transfer experiment on Level 5 Prediction Dataset.**

## Setup

Install `pipenv` 

After pipenv installation:

`cd l5kit`

`pipenv install --dev -e .`

`pipenv shell`

`cd ..`


## Data Download

Due to License issues, we cannot provide data from the L5Kit dataset. Please follow the instructions from the L5Kit authors for downloading and setting the path to data directory. 


## Running adaptation scripts (using pre-trained model)

### Get Model

Run: `sh get_pretrained_model.sh`

### Run Script

For Full model finetuning: `make scene_transfer_full_finetune`

For partial model finetuning (last layers): `make scene_transfer_partial_finetune`

For adaptive normalization: `make scene_transfer_adaptive_layernorm`

For motion style adapters (ours): `make scene_transfer_mosa`

 
## Model Pre-training (takes 1 day)

`make pretrain_l5kit`

Note: Larger batchsize speeds up the training process.