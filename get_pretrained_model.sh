# Get zip file from drive
pip install gdown && gdown https://drive.google.com/uc?id=1XrWBvJj8RJcnVPxTuHWCbG3A9jXRFk8G

# Extract contents
unzip data_checkpoints_SDD_inD_L5.zip && rm data_checkpoints_SDD_inD_L5.zip

# Shift model to checkpoint file
mkdir examples/adaptation/checkpoints
mv data_checkpoints_SDD_inD_L5/vit_tiny_split_73140_steps.pth  examples/adaptation/checkpoints
rm -r data_checkpoints_SDD_inD_L5/
