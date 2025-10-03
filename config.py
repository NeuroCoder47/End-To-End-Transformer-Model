from pathlib import Path
def get_config():
    return{
        "dropout":0.1,
        "d_ff" :1024,
        "layers": 4,
        "heads": 4,
        "batch_size": 4,
        "num_epoch": 2,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 256,
        "datasource": ' ',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\weights/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
