import yaml


def load_yaml(path: str) -> dict:
    yaml_data = None
    with open("pybullet_config.yaml", "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data
