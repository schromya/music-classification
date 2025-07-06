from data_processors.GtzanProcessor import GtzanProcessor

import yaml

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    path = config["data"]["path"]
    data_processor = GtzanProcessor(path)

    data_processor.download_data()


if __name__ == "__main__":
    main()


    