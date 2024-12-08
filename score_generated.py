import warnings

import hydra
from wvmos import get_wvmos

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="score_generated")
def main(config):
    model = get_wvmos(cuda=True)
    mos = model.calculate_dir(config.path_to_score, mean=True)
    print(f"Mos is {mos}")


if __name__ == "__main__":
    main()
