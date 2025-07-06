import yaml

if __name__ == "__main__":


    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    gtzan_path = config["data"]["path"]
    print("GTZAN path:", gtzan_path)