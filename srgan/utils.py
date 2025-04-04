import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import einsum
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchinfo import summary
from torch import GradScaler, autocast

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import os
from PIL import Image
from functools import partial
from tqdm import tqdm
from datetime import datetime
import pytz
import copy
import time
import glob
import gc

print("imports done!")


def get_torch_version():
    torch_version = torch.__version__.split("+")[0]
    torch_number = torch_version.split(".")[:2]
    torch_number_float = torch_number[0] + "." + torch_number[1]
    torch_number_float = float(torch_number_float)
    return torch_number_float


def set_seed(seed=42):
    """
    Seeds basic parameters for reproducibility of results
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # if get_torch_version() <= 1.7:
        #     torch.set_deterministic(True)
        # else:
        #     torch.use_deterministic_algorithms(True)
    print(f"seed {seed} set!")
    

def compute_accuracy(y_pred, y):
    assert len(y_pred)==len(y), "length of y_pred and y must be equal"
    acc = torch.eq(y_pred, y).sum().item()
    acc = acc/len(y_pred)
    return acc


def train_validation_split(train_dataset):
    X_train, X_valid, y_train, y_valid = train_test_split(train_dataset.data, train_dataset.targets, 
                                                          test_size=0.2, random_state=42, shuffle=True, 
                                                          stratify=train_dataset.targets)
    X_train = torch.tensor(X_train, dtype=torch.float64).permute(0, 3, 1, 2)
    X_valid = torch.tensor(X_valid, dtype=torch.float64).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_valid = torch.tensor(y_valid, dtype=torch.int64)
    return X_train, X_valid, y_train, y_valid
    

def predict(model, img_path, device):
    img = cv2.imread(img_path)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img2 = img.copy()
    img = torch.tensor(img)
    img = img.permute(1,2,0)
    img = img.unsqueeze(dim=0)
    img = img.to(device)
    model.eval()
    with torch.inference_mode():
        logit = model(img)
    pred_prob = torch.softmax(logit, dim=1)
    pred_label = pred_prob.argmax(dim=1)
    plt.imshow(img2)
    plt.axis("off")
    plt.label(f"Prediction: {classes[pred_label]}\t\tProbability: {round(pred_prob)}")
    plt.show()


def set_scheduler(scheduler, results, scheduler_on):
    """Makes the neccessary updates to the scheduler."""
    if scheduler_on == "valid_acc":
        scheduler.step(results["valid_acc"][-1])
    elif scheduler_on == "valid_loss":
        scheduler.step(results["valid_loss"][-1])
    elif scheduler_on == "train_acc":
        scheduler.step(results["train_acc"][-1])
    elif scheduler_on == "train_loss":
        scheduler.step(results["train_loss"][-1])
    else:
        raise ValueError("Invalid `scheduler_on` choice.")
    return scheduler


def visualize_results(results, plot_name=None):
    """Plot the training and validation loss and accuracy, given the results dictionary"""
    train_loss, train_acc = results["train_loss"], results["train_acc"]
    val_loss, val_acc = results["valid_loss"], results["valid_acc"]
    cls = ["no", "vort", "sphere"]
    x = np.arange(len(train_loss))  # this is the number of epochs
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
    # ax[0,0].set_title("Loss")
    ax[0,0].set_xlabel("Epochs")
    ax[0,0].set_ylabel("Loss")
    ax[0,0].plot(x, train_loss, label="train_loss", color="orange")
    ax[0,0].plot(x, val_loss, label="valid_loss", color="blue")
    ax[0,0].legend()
    # ax[0,1].set_title("Accuracy")
    ax[0,1].set_xlabel("Epochs")
    ax[0,1].set_ylabel("Accuracy")
    ax[0,1].plot(x, train_acc, label="train_acc", color="orange")
    ax[0,1].plot(x, val_acc, label="valid_acc", color="blue")
    ax[0,1].legend()
    # ax[1,0].set_title("Train ROC AUC Plot")
    ax[1,0].set_xlabel("Epochs")
    ax[1,0].set_ylabel("Train ROC AUC Score")
    ax[1,0].plot(x, results["train_roc_auc_0"], label=cls[0])
    ax[1,0].plot(x, results["train_roc_auc_1"], label=cls[1])
    ax[1,0].plot(x, results["train_roc_auc_2"], label=cls[2])
    ax[1,0].legend()
    # ax[1,1].set_title("Valid ROC AUC Plot")
    ax[1,1].set_xlabel("Epochs")
    ax[1,1].set_ylabel("Valid ROC AUC Score")
    ax[1,1].plot(x, results["valid_roc_auc_0"], label=cls[0])
    ax[1,1].plot(x, results["valid_roc_auc_1"], label=cls[1])
    ax[1,1].plot(x, results["valid_roc_auc_2"], label=cls[2])
    ax[1,1].legend()
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()
    

def train_step(model, loss_fn, optimizer, dataloader, device, scaler=None):
    model.train()
    train_loss = 0
    train_acc = 0
    all_labels = []
    all_preds = []
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if scaler is not None:           # do automatic mixed precision training
            with autocast(device):       # mixed precision forward pass
                logit = model(X)
                pred_prob = torch.softmax(logit, dim=1)
                pred_label = pred_prob.argmax(dim=1)
                # note: first put logit and then y in the loss_fn
                # otherwise, if you put y first and then logit, then it will raise an error
                loss = loss_fn(logit, y)
            scaler.scale(loss).backward()      # mixed precision backward pass
            scaler.step(optimizer)             # updating optimizer
            scaler.update()                    # updating weights
        else:                     # don't do any mixed precision training
            logit = model(X)
            pred_prob = torch.softmax(logit, dim=1)
            pred_label = pred_prob.argmax(dim=1)
            loss = loss_fn(logit, y)
            loss.backward()
            optimizer.step()
        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(pred_prob.detach().cpu().numpy())
        train_loss += loss.item()
        acc = compute_accuracy(pred_label, y)
        train_acc += acc
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    return train_loss, train_acc, all_labels, all_preds
        

def valid_step(model, loss_fn, dataloader, device):
    model.eval()
    valid_loss = 0
    valid_acc = 0
    all_labels = []
    all_preds = []
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logit = model(X)
            pred_prob = torch.softmax(logit, dim=1)
            pred_label = pred_prob.argmax(dim=1)
            # note: first put logit and then y in the loss_fn
            # otherwise, if you put y first and then logit, then it will raise an error
            loss = loss_fn(logit, y)
            valid_loss += loss.item()
            acc = compute_accuracy(pred_label, y)
            valid_acc += acc
            all_labels.extend(y.detach().cpu().numpy())
            all_preds.extend(pred_prob.detach().cpu().numpy())
    valid_loss = valid_loss / len(dataloader)
    valid_acc = valid_acc / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    return valid_loss, valid_acc, all_labels, all_preds


def training_fn1(model, loss_fn, optimizer, train_dataloader, valid_dataloader, device, 
                 epochs, scheduler=None, scheduler_on="val_acc", verbose=False, scaler=None,
                 save_best_model=False, path=None, model_name=None, optimizer_name=None, 
                 scheduler_name=None):
    """
    Does model training and validation for one fold in a k-fold cross validation setting.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "train_roc_auc_0": [],
        "valid_roc_auc_0": [],
        "train_roc_auc_1": [],
        "valid_roc_auc_1": [],
        "train_roc_auc_2": [],
        "valid_roc_auc_2": [],
    }
    best_valid_roc_auc = 0.0
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_labels, train_preds = train_step(model, loss_fn, optimizer, 
                                                                      train_dataloader, device, scaler)
        valid_loss, valid_acc, valid_labels, valid_preds = valid_step(model, loss_fn, valid_dataloader, 
                                                                      device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)
        bin_train_labels = label_binarize(train_labels, classes=[0,1,2])
        bin_valid_labels = label_binarize(valid_labels, classes=[0,1,2])
        mean_train_roc_auc = np.mean([results["train_roc_auc_0"], results["train_roc_auc_1"],
                                          results["train_roc_auc_2"]])
        mean_valid_roc_auc = np.mean([results["valid_roc_auc_0"], results["valid_roc_auc_1"],
                                          results["valid_roc_auc_2"]])
        for i in range(3):
            try:
                train_roc_auc = roc_auc_score(bin_train_labels[:, i], train_preds[:, i])
                valid_roc_auc = roc_auc_score(bin_valid_labels[:, i], valid_preds[:, i])
                results[f"train_roc_auc_{i}"].append(train_roc_auc)
                results[f"valid_roc_auc_{i}"].append(valid_roc_auc)
            except ValueError:
                print(f"Warning: AUC computation failed for class {i}")
        if verbose:
            print(
                    f"Epoch: {epoch+1} | Train_loss: {train_loss:.5f} | "
                    f"Train_acc: {train_acc:.5f} | Val_loss: {valid_loss:.5f} | "
                    f"Val_acc: {valid_acc:.5f} | Train_roc_auc: {mean_train_roc_auc:.5f} | "
                    f"Val_roc_auc: {mean_valid_roc_auc:.5f}"
                )
        if scheduler is not None:
            scheduler = set_scheduler(scheduler, results, scheduler_on)
            if mean_valid_roc_auc > best_valid_roc_auc:
                best_valid_roc_auc = mean_valid_roc_auc
                plot_name = path + "/" + model_name[:-3] + "_" + optimizer_name[:-3] + "_" + scheduler_name[:-3] + ".pdf"
                save_model_info(path, device, model, model_name, optimizer, optimizer_name, 
                    scheduler, scheduler_name)
        else:
            if mean_valid_roc_auc > best_valid_roc_auc:
                best_valid_roc_auc = mean_valid_roc_auc
                plot_name = path + "/" + model_name[:-3] + "_" + optimizer_name[:-3] + ".pdf"
                save_model_info(path, device, model, model_name, optimizer, optimizer_name) 
    visualize_results(results, plot_name)


def training_fn2(model, loss_fn, optimizer, train_dataset, device, epochs, 
                 scheduler=None, scheduler_on="val_acc", verbose=False, n_splits=5, scaler=None):
    """
    Does the training and validation for all the folds in a k-fold cross validation setting.
    """
    kf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    MODELS = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=train_dataset.data, y=train_dataset.targets)):
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, 
                                      sampler=SubsetRandomSampler(train_idx))
        valid_dataloader = DataLoader(dataset=train_dataset, batch_size=64, 
                                      sampler=SubsetRandomSampler(val_idx))
        results = training_fn1(model, loss_fn, optimizer, train_dataloader, valid_dataloader, device, 
                                epochs, scheduler=scheduler, scheduler_on=scheduler_on, verbose=verbose)
        train_loss = np.mean(results["train_loss"])
        valid_loss = np.mean(results["valid_loss"])
        train_acc = np.mean(results["train_acc"])
        valid_acc = np.mean(results["valid_acc"])
        print(
                f"Fold: {fold+1} | Train_loss: {train_loss:.5f} | "
                f"Train_acc: {train_acc:.5f} | Val_loss: {valid_loss:.5f} | "
                f"Val_acc: {valid_acc:.5f}"
            )
        visualize_results(results)
        MODELS.append(model)
    return MODELS


def training_function(model, loss_fn, optimizer, train_dataset, device, epochs, scheduler=None, 
                      scheduler_on="val_acc", verbose=False, validation_strategy="train test split",
                      n_splits=5, scaler=None):
    """
    validation_strategy: choose one of the following: 
        - "train test split"
        - "k-fold cross validation"
    """
    if validation_strategy == "train test split":
        X_train, X_valid, y_train, y_valid = train_validation_split(train_dataset)
        train_dataset = CustomDataset(features=X_train, targets=y_train)
        valid_dataset = CustomDataset(features=X_valid, targets=y_valid)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=CONFIG["batchsize"], shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=CONFIG["batchsize"], shuffle=False)
        training_fn1(model, loss_fn, optimizer, train_dataloader, valid_dataloader, device, epochs, 
                     scheduler=scheduler, scheduler_on=scheduler_on, verbose=verbose)
    elif validation_strategy == "k-fold cross validation":
        training_fn2(model, loss_fn, optimizer, train_dataset, device, epochs, scheduler=scheduler, 
                     scheduler_on=scheduler_on, verbose=verbose, n_splits=n_splits)
    else:
        raise ValueError("Invalid validation strategy.\nChoose either \"train test split\" \
        or \"k-fold cross validation\"")
    

def save_model_info(path: str, device, model, model_name, optimizer, optimizer_name, 
                    scheduler=None, scheduler_name=""):
    model.to(device)
    torch.save(model.state_dict(), os.path.join(path,model_name))
    torch.save(optimizer.state_dict(), os.path.join(path,optimizer_name))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path,scheduler_name))    
    print("Model info saved!")
    
    
def load_model_info(PATH, device, model, model_name, optimizer, optimizer_name, 
                    scheduler=None, scheduler_name=""):
    model.load_state_dict(torch.load(os.path.join(path,model_name)))
    model.to(device)
    optimizer.load_state_dict(torch.load(os.path.join(path,optimizer_name)))
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(os.path.join(path,scheduler_name)))
    print("Model info loaded!")
    
    
def get_current_time():
    """Returns the current time in Toronto."""
    now = datetime.now(pytz.timezone('Canada/Eastern'))
    current_time = now.strftime("%d_%m_%Y__%H_%M_%S")
    return current_time


print("Utility functions created!")
