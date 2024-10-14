from yaml import safe_load

class Config:
    def __init__(self, config_path: str):
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.img_size = self.config["train"]["image_size"]
        self.img_channels = self.config["train"]["image_channels"]
        self.train_time_steps = self.config["train"]["steps"]
        self.inf_time_steps = self.config["inf"]["steps"]
        self.beta_start = self.config["train"]["beta"][0]
        self.beta_end = self.config["train"]["beta"][1]

        self.batch_size = self.config["train"]["batch_size"]
        self.epochs = self.config["train"]["epochs"]
        
        self.save_interval = self.config["save_interval"]

        self.lr = self.config["train"]["lr"]
        
        self.clip: float = self.config["clip"]

        self.device: str = self.config["device"]

        self.model_channels: int = int(self.config["model"]["base_channels"])
        self.ts_embed_dims: int = int(self.config["model"]["timestep_embed_dims"])
        self.ts_proj_dims: int = int(self.config["model"]["timestep_proj_dims"])
        self.layers: int = int(self.config["model"]["layers"])