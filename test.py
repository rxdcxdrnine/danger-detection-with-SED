import glob
import torch
import torchaudio

from network import *


def pre_emphasis(X):
    return X[1:] - 0.97 * X[:-1]


def local_cep_mvn(X):
    X_mean = X.mean(dim=-1, keepdim=True)
    X_std = X.std(dim=-1, keepdim=True)
    X_std[X_std < .001] = .001
    result = (X - X_mean) / X_std
    return result


def feature_extractor(X):
    log_mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
        args["sample_rate"],
        n_fft=args["n_fft"],
        win_length=int(args["sample_rate"] * 0.001 * args["mel_window"]),
        hop_length=int(args["sample_rate"] * 0.001 * args["mel_hop"]),
        window_fn=torch.hamming_window,
        n_mels=args["n_mels"]
    )
    
    X = log_mel_spec_extractor(X)
    X = torch.log(X)
    X = local_cep_mvn(X)
    X = X.unsqueeze(1)
    return X


def record(result, file):
    with open(file, 'a') as f:
        for i in range(len(result)):
            f.write(f"{result[i]:.6f}/")
    


def main(args):
    demo_audio_directory = args["demo_audio_directory"]
    demo_audio, _ = torchaudio.load(demo_audio_directory, normalization=True)
    demo_audio = demo_audio.squeeze(0)
    
    window_length = int(args["sample_rate"] * 0.001 * args["audio_window"])
    shift_length = int(args["sample_rate"] * 0.001 * args["audio_hop"])
    audio_length = demo_audio.size(0)
    
    list_segment = []
    for i in range(0, audio_length, shift_length):
        if i + window_length > audio_length:
            list_segment.append(demo_audio[audio_length - window_length:])
            break
        else :
            list_segment.append(demo_audio[i:(i + window_length)])
    X = torch.stack(list_segment)
    X = feature_extractor(X)
    
    device = torch.device(f"cuda:{args['GPU_NUM']}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    X = X.to(device)    
    model = Network(args, device)
    model.load_state_dict(torch.load(args["save_directory"]))
    
    for i in range(0, X.size(0), args["batch_size"]):
        y_hat = model(X[i:(i + args["batch_size"]), :]).cpu().detach().numpy()
            
        demo_text_directory = args["demo_text_directory"]
        record(y_hat, demo_text_directory)
 
    
        

if __name__ == "__main__":
    args = {
        "demo_audio_directory" : "/code/demo/test_0.wav",
        "demo_text_directory" : "/code/demo/result.txt",
        "weight_directory" : '/code/models/parameters/Joint/weights/best_SED.pt',
        "weight_freeze" : True,
        
        "save_directory" : "/code/result/trained_model_GAT_label_prop_0.5_learning_rate_1e-5.pt",
        "GPU_NUM" : 0,
        "batch_size" : 6,
        
        "sample_rate" : 44100,
        "audio_window" : 5000,
        "audio_hop" : 100,
        
        "mel_window" : 40,
        "mel_hop" : 20,
        "n_fft" : 2048,
        "n_mels" : 128,
        
        "DcaseNet_features" : 256,
    }
    
    main(args)
    
    