import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from galore_torch import GaLoreAdamW
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import gc
from sklearn.manifold import TSNE
from rouge_score import rouge_scorer
from datasets import Dataset
import datasets
from tqdm import tqdm
import json
import pickle
from datasets import concatenate_datasets
import pandas as pd
import concurrent.futures
from peft import LoraModel, LoraConfig



def load_galore_optimizer(model, target_modules_list):
    galore_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(target_key in module_name for target_key in target_modules_list):
            continue
        #print(module_name)
        galore_params.append(module.weight)

    id_galore_params = {id(p) for p in galore_params}
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    for p in regular_params:
        p.requires_grad = False
    return galore_params


def get_embeddings(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def init_lr(learning_rate, num_clients):
    return [learning_rate for _ in range(num_clients)]

def load_model_and_tokenizer_and_optimizer(model_path, device_list, framework, init_lr, rank, lora_alpha, target_modules_list):
    models = []
    
    for device in device_list[:-1]:
        models.append(AutoModelForCausalLM.from_pretrained(model_path, device_map = device))
    global_model = AutoModelForCausalLM.from_pretrained(model_path, device_map = device_list[len(device_list)-1])

    if framework=="flexlora" or framework=="ffa-lora":
        config = LoraConfig(
            r = rank,
            lora_alpha = lora_alpha,
            target_modules = target_modules_list
        )
        models = [LoraModel(model, config, "default") for model in models]
        global_model = LoraModel(global_model, config, "default")
        param_groups = []
        for idx, model in enumerate(models):
            param_list = []
            for name,param in model.named_parameters():
                if param.requires_grad:
                    param_list.append(param)
            param_groups.append({f'params':param_list})
        
        optimizers = [torch.optim.AdamW(param_groups[i][f'params'], lr=init_lr[i]) for i in range(len(models))]
        global_param_group = []
        global_param_list = []
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                global_param_list.append(param)
        global_param_group.append({'params': global_param_list})


    elif framework=="fedftg":
        param_groups=[load_galore_optimizer(model, target_modules_list) for model in models]
        optimizers = [GaLoreAdamW(param_groups[i], init_lr[i]) for i in range(len(models))]
        global_param_group = [load_galore_optimizer(global_model, target_modules_list)]

    tokenizers = [AutoTokenizer.from_pretrained(model_path) for _ in range(len(models))]
    if "Llama" in model_path:
        for tokenizer in tokenizers:
            tokenizer.pad_token = tokenizer.eos_token
    
    return models, global_model, tokenizers, optimizers, param_groups, global_param_group



def create_prompt(question, answer):
    if len(answer) < 1:
        answer = "Cannot Find Answer"
    else:
        answer = answer
    prompt_template = f"### INSTRUCTION\nAnswer the Question\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
    return prompt_template



def evaluate_model(model, eval_dataloader, tokenizer, device):
    model.eval()
    total_loss = 0
    predictions = []
    references = []
    f1_scores = []
    rouge_l_scores = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()

            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            batch_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            batch_references = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            for pred, ref in zip(batch_predictions, batch_references):
                predictions.append(word_tokenize(pred))
                references.append([word_tokenize(ref)])
                
                # Calculate F1 score
                pred_tokens = word_tokenize(pred)
                ref_tokens = word_tokenize(ref)
                pred_set = set(pred_tokens)
                ref_set = set(ref_tokens)

                precision = len(pred_set & ref_set) / len(pred_set) if len(pred_set) > 0 else 0
                recall = len(pred_set & ref_set) / len(ref_set) if len(ref_set) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
                
                # Calculate ROUGE-L score
                rouge_l = scorer.score(ref, pred)['rougeL'].fmeasure
                rouge_l_scores.append(rouge_l)
                #f1_scores.append(rouge_l.fmeasure)

    average_loss = total_loss / len(eval_dataloader)
    perplexity = torch.exp(torch.tensor(average_loss))
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothie)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return average_loss, perplexity, bleu_score, avg_rouge_l



def evaluate_model_small(model, eval_dataloader, tokenizer, device):
    model.eval()
    rouge_l_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with torch.no_grad():
        # Iterate through the first 5 samples of eval_dataloader
        num_sample_batch = 5
        for batch_idx, batch in enumerate(eval_dataloader):
            if (batch_idx+1) * eval_dataloader.batch_size >= num_sample_batch:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch['input_ids'])

            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            batch_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            batch_references = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            for pred, ref in zip(batch_predictions, batch_references):
                # Calculate ROUGE-L score
                rouge_l = scorer.score(ref, pred)['rougeL'].fmeasure
                rouge_l_scores.append(rouge_l)

    avg_rouge_l_f1 = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    return avg_rouge_l_f1


def get_client_dataloaders(dataframe_paths, n_clients, batch_size, test_size_fraction, tokenizers):
    client_dataframes = []
    for i in range(n_clients):
        path = dataframe_paths + f"/client_{i+1}.pkl"
        df = pd.read_pickle(path)
        client_dataframes.append(df)

    client_datasets = [Dataset.from_pandas(df) for df in client_dataframes]

    trainsets = []

    for i in range(len(client_dataframes)):
        trainset = client_datasets[i].train_test_split(test_size= test_size_fraction)
        client_datasets[i] = datasets.DatasetDict({
            'train': trainset['train'],
            'test': trainset['test']
        })
    
    mapped_datasets = []
    for idx, dataset in enumerate(client_datasets):
        mapped_dataset = dataset.map(lambda samples: tokenizers[idx](create_prompt(samples['question'], samples['answer']), truncation=True, padding='max_length', max_length=512))
        mapped_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
        mapped_datasets.append(mapped_dataset)
        # mapped_datasets.append(dataset.map(lambda samples: tokenizers[idx](create_prompt(samples['question'], samples['answer']), truncation=True, padding='max_length', max_length=512)))
        # mapped_datasets = [mapped_dataset.set_format('torch',columns = ['input_ids', 'attention_mask']) for mapped_dataset in mapped_datasets]
    
    train_datasets = [mapped_dataset['train'] for mapped_dataset in mapped_datasets]
    eval_datasets = [mapped_dataset['test'] for mapped_dataset in mapped_datasets]

    train_dataloaders = [DataLoader(train_dataset, batch_size, shuffle=True) for train_dataset in train_datasets]
    eval_dataloaders = [DataLoader(eval_dataset, batch_size) for eval_dataset in eval_datasets]

    return train_dataloaders, eval_dataloaders


def average_model_weights(model1, model2, target_model):
    # Averaging the parameters with requires_grad=True
    with torch.no_grad():
        for param1, param2, param_target in zip(model1.parameters(), model2.parameters(), target_model.parameters()):
            if param1.requires_grad and param2.requires_grad:
                param_target.data = (param1.data + param2.data) / 2


def generate_sequence(model, input_ids, attention_mask):
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    return generated_ids


def create_and_save_graph(x, y, xlabel, ylabel, title, filename, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.hf_model}_{args.framework}_{args.dataset}_{args.n_clients}clients_{args.epochs}epochs_{args.local_iter}localiter_{args.rank}rank_{timestamp}"
    folder_path = os.path.join(os.getcwd(), folder_name)

    os.makedirs(folder_path, exist_ok=True)

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plot_path = os.path.join(folder_path, filename)
    plt.savefig(plot_path)
    plt.close()

    # Save metadata
    metadata = {
        'x': x,
        'y': y,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'title': title,
        'filename': plot_path,
        'args': vars(args)  #saving arguments in the metadata as well
    }
    metadata_filename = os.path.join(folder_path, filename.rpartition('.png')[0] + '.pkl')
    with open(metadata_filename, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Graph and metadata saved in: {folder_path}")



def save_entire_model(filename, global_model, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{config['hf_model']}_{config['framework']}_{config['dataset']}_{config['n_clients']}clients_{config['epochs']}epochs_{config['local_iter']}localiter_{config['rank']}rank_{timestamp}"
    folder_path = os.path.join(os.getcwd(), folder_name)

    os.makedirs(folder_path, exist_ok=True)

    model_path = os.path.join(folder_path, filename) # assuming .pth is provided as filename by the user
    torch.save(global_model, model_path)
    
    print(f"Model and its weights are saved in: {folder_path}")


def fedavg_train_fedftg(models, global_model, tokenizers, optimizers, train_dataloaders, eval_dataloaders, epochs, local_iter, device_list, metrics_at_aggregation, config):
    
    for model in models:
        model.train()
    
    global_model.train()
    
    

    for epoch in range(epochs):
        total_batches = len(train_dataloaders[0])
        for i,batches in enumerate(tqdm(zip(*train_dataloaders), total = total_batches)):
            losses = []

        # Process each client's batch
            for client_idx, (batch, model, optimizer, device) in enumerate(zip(batches, models, optimizers, device_list)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (i + 1) % local_iter == 0 or (i + 1) == total_batches:
                print(f"Performing global aggregation FedAvg step...", flush=True)
                with torch.no_grad():
                    for params in zip(*(model.parameters() for model in models), global_model.parameters()):
                        avg_param = sum(param.to(device_list[len(device_list)-1]) for param in params[:-1]) / len(models)
                        params[-1].data.copy_(avg_param)

                # Copy global model parameters back to clients
                for model, device in zip(models, device_list):
                    for param, global_param in zip(model.parameters(), global_model.parameters()):
                        param.data.copy_(global_param.to(device).data)
                print("Aggregation complete, now evaluating the global model")

                # Evaluate global model
                rouge_L = evaluate_global_model(global_model, eval_dataloaders, tokenizers, device_list[len(device_list)-1])
                metrics_at_aggregation.append(rouge_L)

        print(f"epoch {epoch+1}/{epochs} completed", flush=True)
        # now evaluate the global model for other metrics after the complete epoch
        avg_losses, perplexities, bleus, rouge = evaluate_clients(global_model, eval_dataloaders, tokenizers, device_list)
    
    save_entire_model("global_model.pth", global_model, config)
    
    del models
    del global_model

    torch.cuda.empty_cache()
    gc.collect()
    
    # return metrics_at_aggregation
    


# def evaluate_global_model(global_model, dataloaders, tokenizers, device):
#     with torch.no_grad():
#         metrics = []
        
#         def evaluate_and_append(dataloader, tokenizer):
#             avg_rouge_l = evaluate_model_small(global_model, dataloader, tokenizer, device)
#             return avg_rouge_l

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_to_dataloader = {executor.submit(evaluate_and_append, dataloader, tokenizer): (dataloader, tokenizer)
#                                     for dataloader, tokenizer in zip(dataloaders, tokenizers)}

#             for future in concurrent.futures.as_completed(future_to_dataloader):
#                 avg_rouge_l = future.result()
#                 metrics.append(avg_rouge_l)

#     return sum(metrics) / len(metrics)

def evaluate_global_model(global_model, dataloaders, tokenizers, device):
    with torch.no_grad():
        metrics = []
        for dataloader, tokenizer in zip(dataloaders, tokenizers):
            avg_rouge_l=evaluate_model_small(global_model, dataloader, tokenizer, device) # can also use evaluate_model_samll instead of evaluate_model at each aggregation would take a lot of time
            metrics.append(avg_rouge_l) 
    return sum(metrics) / len(metrics)




def evaluate_clients(global_model, dataloaders, tokenizers, device_list):
    """Evaluate each dataset for the current epoch."""
    avg_losses, perplexities, bleus, rouge = [], [], [], []

    for idx, (dataloader, tokenizer, device) in enumerate(zip(dataloaders, tokenizers, device_list)):
        loss, perplexity, bleu, rouge_l = evaluate_model(global_model, dataloader, tokenizer, device)
        print(f"Dataset{idx} - Loss: {loss:.4f}, Perplexity: {perplexity:.4f}, BLEU: {bleu:.4f}, F1: {f1:.4f}, ROUGE_L: {rouge_l:.4f}", flush=True)
        avg_losses.append(loss)
        perplexities.append(perplexity)
        bleus.append(bleu)
        rouge.append(rouge_l)

    return avg_losses, perplexities, bleus, rouge


## Functions specific to FlexLoRA FedAvg implementation########
def extract_lora_params(model):
    lora_A_params = {}
    lora_B_params = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            lora_A_params[name] = param
        elif 'lora_B' in name:
            lora_B_params[name] = param
    return lora_A_params, lora_B_params

def decompose_matrix(W, K):
    U, S, V = torch.svd(W)
    U_k = U[:, :K]
    S_k = S[:K]
    V_k = V[:, :K]

    B = U_k @ torch.diag(torch.sqrt(S_k))
    A = torch.diag(torch.sqrt(S_k)) @ V_k.T

    B = B[:, :K]
    A = A[:K, :]
    
    return B, A

def average_lora_params(lora_params_list, device, rank):
    num_clients = len(lora_params_list)
    W_avg_list = []
    
    for key in lora_params_list[0]['lora_A']:
        W_sum = None
        for lora_params in lora_params_list:
            A = lora_params['lora_A'][key].to(device)
            B_key = key.replace('lora_A', 'lora_B')
            B = lora_params['lora_B'][B_key].to(device)
            
            W = torch.mm(B, A)
            if W_sum is None:
                W_sum = W
            else:
                W_sum += W
        
        W_avg = W_sum / num_clients
        W_avg_list.append((key, W_avg))
    
    avg_lora_A_params = {}
    avg_lora_B_params = {}
    
    for key, W_avg in W_avg_list:
        B, A = decompose_matrix(W_avg, rank)
        avg_lora_A_params[key] = A
        avg_lora_B_params[key.replace('lora_A', 'lora_B')] = B
    
    return avg_lora_A_params, avg_lora_B_params

def update_model_with_lora_params(global_model, avg_lora_A_params, avg_lora_B_params):
    for key, avg_A in avg_lora_A_params.items():
        global_model.state_dict()[key].copy_(avg_A)
    
    for key, avg_B in avg_lora_B_params.items():
        global_model.state_dict()[key].copy_(avg_B)

##############


def fedavg_train_flexlora(models, global_model, tokenizers, optimizers, train_dataloaders, eval_dataloaders, epochs, local_iter, device_list, metrics_at_aggregation, config):
    
    for model in models:
        model.train()
    
    global_model.train()
    
    

    for epoch in range(epochs):
        total_batches = len(train_dataloaders[0])
        for i,batches in enumerate(tqdm(zip(*train_dataloaders), total = total_batches)):
            losses = []

        # Process each client's batch
            for client_idx, (batch, model, optimizer, device) in enumerate(zip(batches, models, optimizers, device_list)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (i + 1) % local_iter == 0 or (i + 1) == total_batches:
                print(f"Performing global aggregation FedAvg step...", flush=True)
                lora_params_dict_list = []
                for model in models:
                    lora_params_A, lora_params_B = extract_lora_params(model)
                    lora_params_dict_list.append({'lora_A':lora_params_A, 'lora_B':lora_params_B})
                # now coming to averaging
                avg_lora_A_params, avg_lora_B_params = average_lora_params(lora_params_dict_list, device_list[len(device_list)-1], config['rank'])

                print("Averaging done!", flush=True)

                update_model_with_lora_params(global_model, avg_lora_A_params, avg_lora_B_params)

                with torch.no_grad():
                    for param_global, *param_clients in zip(global_model.parameters(), *(model.parameters() for model in models)):
                        for param_client in param_clients:
                            param_client.data = param_global.data.clone().to(param_client.device)

                del avg_lora_A_params, avg_lora_B_params
                del lora_params_dict_list
                torch.cuda.empty_cache()
                gc.collect()

                # Evaluate global model
                rouge_L = evaluate_global_model(global_model, eval_dataloaders, tokenizers, device_list[len(device_list)-1])
                metrics_at_aggregation.append(rouge_L)

        print(f"epoch {epoch+1}/{epochs} completed", flush=True)
        # now evaluate the global model for other metrics after the complete epoch
        avg_losses, perplexities, bleus, rouge = evaluate_clients(global_model, eval_dataloaders, tokenizers, device_list)
    
    save_entire_model("global_model.pth", global_model, config)

    del models
    del global_model

    torch.cuda.empty_cache()
    gc.collect()
    
    # return metrics_at_aggregation


def fedavg_train_ffalora(models, global_model, tokenizers, optimizers, train_dataloaders, eval_dataloaders, epochs, local_iter, device_list, metrics_at_aggregation, config):
    
    for model in models:
        model.train()
    
    global_model.train()
    
    

    for epoch in range(epochs):
        total_batches = len(train_dataloaders[0])
        for i,batches in enumerate(tqdm(zip(*train_dataloaders), total = total_batches)):
            losses = []

        # Process each client's batch
            for client_idx, (batch, model, optimizer, device) in enumerate(zip(batches, models, optimizers, device_list)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (i + 1) % local_iter == 0 or (i + 1) == total_batches:
                print(f"Performing global aggregation FedAvg step...", flush=True)
                with torch.no_grad():
                    for param_list in zip(*(model.parameters() for model in models), global_model.parameters()):
                        *client_params, global_param = param_list
                        if all(param.requires_grad for param in client_params + [global_param]):
                            client_params_on_device = [param.to(device_list[len(device_list)-1]) for param in client_params]
                            global_param.data = sum(param.data for param in client_params_on_device) / len(client_params_on_device)

                with torch.no_grad():
                    for global_param, *client_params in zip(global_model.parameters(), *(model.parameters() for model in models)):
                        if all(param.requires_grad for param in [global_param] + client_params):
                            for param, device in zip(client_params, device_list[:-1]):
                                param.data = global_param.to(device).data

                # Evaluate global model
                rouge_L = evaluate_global_model(global_model, eval_dataloaders, tokenizers, device_list[len(device_list)-1])
                metrics_at_aggregation.append(rouge_L)

        print(f"epoch {epoch+1}/{epochs} completed", flush=True)
        # now evaluate the global model for other metrics after the complete epoch
        avg_losses, perplexities, bleus, rouge = evaluate_clients(global_model, eval_dataloaders, tokenizers, device_list)
    
    save_entire_model("global_model.pth", global_model, config)
    del models
    del global_model

    torch.cuda.empty_cache()
    gc.collect()
    
    # return metrics_at_aggregation

