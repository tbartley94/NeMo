from omegaconf import OmegaConf
from nemo.core.config import hydra_runner

@hydra_runner(config_path="./", config_name="fast-conformer_aed")
def main(cfg):