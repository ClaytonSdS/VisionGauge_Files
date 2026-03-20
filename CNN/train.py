import torch
from torch.utils.data import DataLoader
from mlflow.entities import LoggedModelStatus
from databricks.sdk import WorkspaceClient
from .utils import train_one_epoch, validate_one_epoch, test_one_epoch, inferir_assinatura, print_logs
import mlflow
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
import os

def train_model(
                # Loaders
                train_loader:torch.utils.data.DataLoader, valid_loader:torch.utils.data.DataLoader, test_loader:torch.utils.data.DataLoader,
                
                # Parametros do Modelo e Treinamento
                model: torch.nn.Module, hyperparameters:dict, criterion: torch.nn.Module, optimizer:torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                checkpoint_save_path:str, 
                
                # MLFLOW Parâmetros
                mlflow_experiment_path:str, mlflow_author:str, mlflow_run_name:str, mlflow_model_name:str,
                mlflow_training_dataset:pd.DataFrame, mlflow_training_dataset_name:str, mlflow_training_dataset_source:str, 

                # Setados
                mlflow_start_saving_checkpoints_after:int=10, mlflow_tracking_uri:str="databricks", mlflow_register_model:bool=False,
                mlflow_model_type:str="CNN", mlflow_model_tags:dict=None, mlflow_run_tags:dict=None, training_framework:str="pytorch",  patience:int=15,
                ):
    
    # Verificar qual o device e mover o modelo para ele
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inicializar o histórico para adição dos resultados
    history = []

    # Inicializar a variável de paciência
    current_patience = 0

    # Setar a melhor loss como +inf
    best_loss = float("inf")

    # Pegar os hiperparametros do dicionário
    learning_rate = hyperparameters.get("learning_rate", None)
    epochs = hyperparameters.get("epochs", None)
    batch_size = hyperparameters.get("batch_size", None)

    # Criar o model tags default caso seja none para mlflow_model_tags e  mlflow_run_tags
    mlflow_model_tags = mlflow_model_tags if mlflow_model_tags else{"architecture":mlflow_model_type, "dataset": mlflow_training_dataset_name}
    mlflow_run_tags = mlflow_run_tags if mlflow_run_tags else {"architecture":mlflow_model_type, "dataset": mlflow_training_dataset_name}

    # Verificar os catalogos disponíveis
    catalogs = []
    for catalog in WorkspaceClient().catalogs.list():
        catalogs.append(catalog.name)


    # Conectar o mlflow a um tracking server, i.e., databricks
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Definir um experimento para salvar as runs
    mlflow.set_experiment(mlflow_experiment_path)

    # Inicializar uma run no mlflow
    with mlflow.start_run(run_name=mlflow_run_name):
        # Inicializar o modelo, com nome, tipo e tags próprias
        model_mlflow = mlflow.initialize_logged_model(name=mlflow_model_name,  model_type=mlflow_model_type,  tags= mlflow_model_tags)

        # Adicionar tags a run
        mlflow.set_tags(mlflow_run_tags)

        # Criar um dataset mlflow
        training_dataset = mlflow.data.from_pandas(mlflow_training_dataset, name=mlflow_training_dataset_name, source=mlflow_training_dataset_source)

        # Adicionar Dataset ao Run
        mlflow.log_input(training_dataset, context=mlflow_training_dataset_name)

        # Adicionar os hiperparâmetros na run
        mlflow.log_params(hyperparameters)

        # Inferir assinature e input_example para registro do modelo
        input_example, signature = inferir_assinatura(loader=train_loader)

        # Rodar as epocas
        for e in tqdm(range(epochs), desc="Epochs", leave=False):
            # Rodar treinamento, validação e teste por uma epoca
            train_loss = train_one_epoch(model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch = e, device=device)
            valid_loss = validate_one_epoch(model, val_loader=valid_loader, criterion=criterion, epoch = e, device=device)
            test_loss = test_one_epoch(model, test_loader=test_loader, criterion=criterion, epoch = e, device=device) if test_loader else 0

            # Organizar as métricas
            metrics = {"train_loss":train_loss, "valid_loss":valid_loss, "test_loss":test_loss, "learning_rate":learning_rate}

            # Adicionar ao histórico as logs
            history.append(metrics)

            # Atualizar o valor do learning_rate
            learning_rate = optimizer.param_groups[0]["lr"]

            # Ativar scheduler para avançar
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()


            # Adicionar as métricas na run e no modelo
            mlflow.log_metrics(metrics, model_id=model_mlflow.model_id, step=e, dataset= training_dataset)

            if valid_loss < best_loss:
                # Atualizar o valor da melhor loss
                best_loss = valid_loss
                
                # Resetar a paciência
                current_patience = 0

                # Criar as tags para utilizar no modelo registrado
                register_tags = {"train_loss":f"{train_loss:.2f}", "valid_loss":f"{valid_loss:.2f}", "test_loss":f"{test_loss:.2f}",
                                 "framework": training_framework, "created_by": mlflow_author, "dataset": mlflow_training_dataset_name, "model_type": mlflow_model_type}
                
                # Pegar o model state atual
                model_state = model.state_dict()

                # Criar o checkpoint e salvar como torch
                best_model_path = os.path.join(checkpoint_save_path, f"checkpoint_epoch_{e}.pt")
                torch.save(model_state, best_model_path)

                # Salvar o checkpoint com artefato na pasta "/checkpoints" a cada "mlflow_start_saving_checkpoints_after" epocas
                if e >= mlflow_start_saving_checkpoints_after:
                    mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

                # Printar as logs
                print_logs(isthebest=True, logs=metrics)

            # Modelo sem melhora na validação
            else:
                current_patience += 1

                # Printar as logs
                print_logs(logs=metrics)

                if current_patience >= patience:
                    print(f"Modelo atingiu o limite de espera na época {e}")
                    break

        # Finalizar o modelo e colocar como pronto
        try:
            mlflow.finalize_logged_model(model_mlflow.model_id, LoggedModelStatus.READY)

        except Exception as e:
            mlflow.finalize_logged_model(model_mlflow.model_id, LoggedModelStatus.FAILED)
            raise e
            
        # Carrega os pesos do melhor modelo antes de salvar
        model.load_state_dict(torch.load(best_model_path))


        # Adicionar o modelo como artefato
        try:
            mlflow.pytorch.log_model(pytorch_model=model, signature=signature, input_example=input_example, model_id=model_mlflow.model_id)

            # Registrar o modelo caso true
            if mlflow_register_model:
                mlflow_register_name = f"workspace.default.{mlflow_model_name}"
                mlflow.register_model(model_uri=f"models:/{model_mlflow.model_id}", name=mlflow_register_name, tags=register_tags)
        
        except Exception as e:
            raise e
            



                




            




