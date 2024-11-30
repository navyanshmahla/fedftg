import argparse
import yaml
from utils import *

def parse_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config file")
    args = parser.parse_args()
    config = parse_config(args.config)

    print("Configuration Loaded:")
    for key, value in config.items():
        print(f"{key}: {value}")

    device_list = config["device_list"]
    batch_size = config["batch_size"]
    n_clients = config["n_clients"]
    target_module_list = config["target_module_list"]
    framework = config["framework"]
    dataframe_paths = config["dataframe_paths"]
    dataset = config["dataset"]
    hf_model = config["hf_model"]
    config["lr"] = [float(lr) for lr in config["lr"]]
    lr = config["lr"]
    rank = config["rank"]
    lora_alpha = config["lora_alpha"]
    test_size_fraction = config["test_size_fraction"]
    epochs = config["epochs"]
    local_iter = config["local_iter"]

    
    models, global_model, tokenizers, optimizers, param_groups, global_param_group = load_model_and_tokenizer_and_optimizer(
        hf_model, device_list, framework, lr, rank, lora_alpha, target_module_list
    )

    train_dataloaders, eval_dataloaders = get_client_dataloaders(dataframe_paths, n_clients, batch_size, test_size_fraction, tokenizers)

    metrics_at_aggregation = []
    if framework == "fedftg":
        fedavg_train_fedftg(
            models, global_model, tokenizers, optimizers, train_dataloaders, eval_dataloaders,
            epochs, local_iter, device_list, metrics_at_aggregation, config
        )
    elif framework == "flexlora":
        fedavg_train_flexlora(
            models, global_model, tokenizers, optimizers, train_dataloaders, eval_dataloaders,
            epochs, local_iter, device_list, metrics_at_aggregation, config
        )
    elif framework == "ffa-lora":
        for model in models:
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    param.requires_grad = False

        for name, param in global_model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False

        fedavg_train_ffalora(
            models, global_model, tokenizers, optimizers, train_dataloaders, eval_dataloaders,
            epochs, local_iter, device_list, metrics_at_aggregation, config
        )

    steps_list = list(range(len(metrics_at_aggregation)))
    graph_save_path = "./results"
    create_and_save_graph(
        steps_list, metrics_at_aggregation, 'Aggregation Steps', 'ROUGE_L Score',
        'ROUGE_L F1 Score vs Steps', f"{graph_save_path}/agg.png"
    )
