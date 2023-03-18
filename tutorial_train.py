from share import *
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = sys.argv[1]
batch_size = int(sys.argv[2])
logger_freq = int(sys.argv[3])
cldm_config_file = sys.argv[4]
num_gpu = int(sys.argv[5])
num_workers = int(sys.argv[6])
prompt_file = sys.argv[7]
data_dir = sys.argv[8]
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(cldm_config_file).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(prompt_file, data_dir)
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=num_gpu, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader)
