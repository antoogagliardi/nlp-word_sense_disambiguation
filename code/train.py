import os
from pprint import pprint
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import nltk
nltk.download('wordnet')

from src.utils import load_json_data, read_config_file
from src.data import WSD_Dataset, WSD_DataModule
from src.model import WSD_Model


## Main Program
os.makedirs("ckpt", exist_ok=True)
os.makedirs("wandb", exist_ok=True)

# Read configuration file
cfg = read_config_file("configs/config.yaml")
print("== Configuration File ==")
pprint(cfg)



# Setup the device for training
device = torch.device(cfg["project"]["device"])
print(f"Device used: {device}")



# Paths configuration
root_path = "../"
cwd = os.getcwd()
data_path = os.path.join(root_path, cfg["paths"]["data"])   # "data/data"
    # Coarse-Grain WSD Datasets
coarse_grain_train = os.path.join(data_path, "coarse-grained/train_coarse_grained.json")
coarse_grain_val = os.path.join(data_path, "coarse-grained/val_coarse_grained.json")
coarse_grain_test = os.path.join(data_path, "coarse-grained/test_coarse_grained.json")
    # Fine-Grain WSD Datasets
fine_grain_train = os.path.join(data_path, "fine-grained/train_fine_grained.json")
fine_grain_val = os.path.join(data_path, "fine-grained/val_fine_grained.json")
fine_grain_test = os.path.join(data_path, "fine-grained/test_fine_grained.json")
    # Fine-Coarse-Grain Mapping WSD Datasets
coarse_fine_grain_map = os.path.join(data_path, "map/coarse_fine_defs_map.json")

# Retrieve metadata from JSON files
coarse_train_metadata = load_json_data(coarse_grain_train)
coarse_val_metadata = load_json_data(coarse_grain_val)
coarse_test_metadata = load_json_data(coarse_grain_test)

fine_train_metadata = load_json_data(fine_grain_train)
fine_val_metadata = load_json_data(fine_grain_val)
fine_test_metadata = load_json_data(fine_grain_test)

coarse_fine_mapping = load_json_data(coarse_fine_grain_map)
print("Length Fine-Coarse Grain Mapping Dataset: ", len(coarse_fine_mapping))



# Dataset creation
LOAD_DATA = cfg["paths"]["load_existing_data"]
WEAK_SUPERVISION = cfg["paths"]["weak_supervision"]

if LOAD_DATA == True:
    name = f"datasets_weak.data" if WEAK_SUPERVISION == True else f"datasets_noWeak.data"
    train_data = torch.load(os.path.join(data_path, f"{name}"), weights_only=False)["training_set"]
    val_data = torch.load(os.path.join(data_path, f"{name}"), weights_only=False)["validation_set"]
    test_data = torch.load(os.path.join(data_path, f"{name}"), weights_only=False)["testing_set"]
    print("Datasets Correctly Loaded")
    print(" - Length of Training Dataset: ", len(train_data))
    print(" - Length of Validation Dataset: ", len(val_data))
    print(" - Length of Test Dataset: ", len(test_data))
else:
    print("Training Data")
    train_data = WSD_Dataset(coarse_train_metadata, fine_train_metadata, coarse_fine_mapping, weak_supervision=WEAK_SUPERVISION)
    print("Validation Data")
    val_data = WSD_Dataset(coarse_val_metadata, fine_val_metadata, coarse_fine_mapping, weak_supervision=WEAK_SUPERVISION)
    print("Test Data")
    test_data = WSD_Dataset(coarse_test_metadata, fine_test_metadata, coarse_fine_mapping, weak_supervision=WEAK_SUPERVISION)
    torch.save({"training_set": train_data,
                "validation_set": val_data,
                "testing_set": test_data},
                os.path.join(data_path, f"datasets_weak.data" if WEAK_SUPERVISION else f"datasets_noWeak.data"))
    torch.save({"coarse-fine": coarse_fine_mapping}, os.path.join(root_path, "model/coarse-fine_map.data"))
    print("Datasets Saved")
# for i in range(10):
#   print(train_data.dataset[i])


BATCH_SIZE = 32
data_manager = WSD_DataModule(train_dataset=train_data,
                              valid_dataset=val_data,
                              test_dataset=None, batch_size=BATCH_SIZE, train_shuffle=True)
# for i ,(sent, label) in enumerate(data_manager.train_dataloader(), 0):
#     if i < 1:
#         print(f"Row: {i}")
#         print(f" Sentences: \n  {sent}")
#         print(f" Sentences Shape: {sent['input_ids'].shape}")
#         print(f" Labels: \n  {label}")
#         print(f" Labels Shape: {label.shape}")
#     else:
#         break



# Model Training Loop

# Prepare the training
config = cfg["training"]
ckpt_path = os.path.join(os.getcwd(), config["ckpt_path"])
ckpt_path = os.path.join(ckpt_path, "weak") if WEAK_SUPERVISION else os.path.join(ckpt_path, "noWeak")
print("ckpt path: ", ckpt_path)
PROJECT_NAME = config["wandb_proj"]
RUN_NAME = config["wandb_run"]
# LAST_RUN_ID is discovered once the project is created
EPOCHS = config["epochs"]
ACCUMULATE_BATCH = config["grad_accumulation"]


# Model Instantiation
model = WSD_Model(fine_tuning=True, token_path="../model/bert_tokenizer/", model_path="../model/bert_model/")
if device.type == "mps":    # Note: MPS Backend doesn't support torch.DoubleTensor(=float64)
    model = model.to(device=device, dtype=torch.float32)
else:
    model = model.to(device=device)
pprint(model)


# Training loop
RESUME_TRAIN = config["resume"]
if RESUME_TRAIN == True:
    LAST_RUN_ID = config["wandb_runID"]            # LAST_RUN_ID is discovered once the project is created
    LAST_EPOCH = config["last_epoch"]
    ADD_EPOCHS = config["add_epoch"]
    run = wandb.init(project=PROJECT_NAME, name=RUN_NAME,
                     config=config,
                     resume=True, id=LAST_RUN_ID)
    ckpt = os.path.join(ckpt_path, f"{LAST_RUN_ID}/model-epoch={LAST_EPOCH-1}.ckpt")
    EPOCHS = LAST_EPOCH + ADD_EPOCHS
else:
    run = wandb.init(project=PROJECT_NAME,
                     name=RUN_NAME,config=config)

wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=ckpt_path,
                           offline=False,
                           project=PROJECT_NAME, log_model=False)   # log_model = "all"/True/False

callback_list = [ModelCheckpoint(dirpath=f"{ckpt_path}/{wandb.run.id}",
                                 filename='model-{epoch}',
                                 save_top_k=-1,
                                 every_n_epochs=1),
                EarlyStopping(monitor="valid_loss", min_delta=0.001, patience=5, mode="min"),
                EarlyStopping(monitor="valid_F1score", min_delta=0.001, patience=5, mode="max")]

trainer = pl.Trainer(accelerator=device.type,                   # gpu, cpu, mps
                     num_sanity_val_steps=1,
                     limit_train_batches=10,
                     limit_val_batches=3,
                     accumulate_grad_batches=ACCUMULATE_BATCH,
                     logger= wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks = callback_list,
                     log_every_n_steps=1)
trainer.wandb_id    = wandb.run.id
trainer.device      = device
trainer.batch_size  = BATCH_SIZE
# trainer.learning_rate = LEARNING_RATE
# trainer.gradient_accumulation = GRADIENT_ACCUMULATION

training_dataloader = data_manager.train_dataloader()
validation_dataloader = data_manager.val_dataloader()

# Start the training procedure
if RESUME_TRAIN == True:
    trainer.fit(model,
                train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader,
                ckpt_path=ckpt)
else:
    trainer.fit(model,
                train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)
wandb.finish()
print("Training Complete")