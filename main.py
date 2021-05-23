import os
import glob
import pickle
from tqdm import tqdm
import datetime

import numpy as np
import torch
import torchaudio
import neptune

from loader import *
from network import *

seed_num = 42

def main(args):

    ## load data
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
            
    train_dataset = CustomDataset(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)

    test_dataset = CustomDataset(args, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)
    
    device = torch.device(f"cuda:{args['GPU_NUM']}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("Current cuda device :", torch.cuda.current_device(), "\n")
    
    model = Network(args, device)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    
    if device == "cuda":
        torch.cuda.manual_seed(seed_num)
        
    if args["neptune"]:
        neptune.init("*")
        neptune.create_experiment(name="SED")        
    
    epochs = 100
    for epoch in range(epochs):
        
        model.train()
        loss_per_batch = 0
        total_batch = len(train_dataloader)
        for X_train, y_train in tqdm(train_dataloader, ncols=60):
            X_train = X_train.to(device)
            y_train = y_train.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            loss.backward()
            optimizer.step()
            
            loss_per_batch += loss.item() / total_batch
            
            
        model.eval()
        correct = 0
        total = 0
        with torch.set_grad_enabled(False):
            for X_test, y_test in tqdm(test_dataloader, ncols=60):
                X_test = X_test.to(device)
                y_hat = model(X_test).cpu().detach().numpy()
                predict = np.where(y_hat >= .5, 1, 0)
                
                total += y_test.size(0)
                correct += (predict == y_test.numpy()).sum()
        
        torch.save(model.state_dict(), args["save_filepath"])   
        
        accuracy = (correct / total) * 100
        print(f"epoch : {epoch + 1}    loss : {loss_per_batch:.6f}    accuracy : {accuracy:.6f}%")
        
        if args["neptune"]:
            neptune.log_metric("loss", loss_per_batch)
            neptune.log_metric("accuracy", accuracy)
 
            
            
            
            
if __name__ == "__main__":
    
    random.seed(42)
    
    train_audio_files = glob.glob("/data/project_2/DCASE2020_task1-A/train_audio/*.wav")
    test_audio_files = glob.glob("/data/project_2/DCASE2020_task1-A/test_audio/*.wav")
    
    event_audio_files = glob.glob("/data/project_2/DCASE2020_task1-A/event_audio/*.wav")
    random.shuffle(event_audio_files)
    
    train_event_audio_files = event_audio_files[:40]
    test_event_audio_files = event_audio_files[40:]
    
    args = {
        "train_filepath" : train_audio_files,
        "test_filepath" : test_audio_files,
        "train_event_filepath" : train_event_audio_files,
        "test_event_filepath" : test_event_audio_files,
        
        "weight_directory" : "/code/models/parameters/Joint/weights/best_SED.pt",
        "weight_freeze" : True,
        
        "save_filepath" : "/code/result/trained_model_MLP_label_prop_0.5_learning_rate_1e-2.pt",
        'GPU_NUM' : 0,
        "batch_size" : 6,   # 6 or 12
        
        "samp_rate" : 44100,
        "utt_length" : 5000,
        "win_length" : 40,
        "hop_length" : 20,
        "n_fft" : 2048,
        "n_mels" : 128,
        
        "DcaseNet_features" : 256,
        "learning_rate" : 1e-2,
        "neptune" : True,
    }
    
    main(args)
