import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path='./conf', config_name='eval', version_base=None)
def main(cfg):
    assert cfg.name == 'eval'
    assert cfg.one == 1
    assert cfg.two == 2
    assert cfg.three == 3

main()
