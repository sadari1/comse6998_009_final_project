#%%
import skvideo
skvideo.setFFmpegPath("/usr/bin")
import skvideo.io 


import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import PIL
# from Attack_Models.CNN_Attack import Attack
# import skvideo.io 
import json 
import torchvision.models as models
import time
from TREMBA.FCN import *
from TREMBA.utils import *
from TREMBA.dataloader import *
import pandas as pd
import os
from pretrainedmodels import utils as ptm_utils
from pathlib import Path
from utils import * 

from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from video_caption_pytorch.process_features import process_batches
from video_caption_pytorch.process_features import create_batches as create_batches_eval
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils
#%%



def EmbedBA(function, encoder, decoder, image, label, config, latent=None):
#     device = image.device

    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config['sample_size'])).cuda()#, device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config['lr']
    for iter in range(config['num_iters']+1):
        
        print(f"\tIteration {iter} / {config['num_iters']+1}")

        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*config['epsilon'], -config['epsilon'], config['epsilon'])
        logit, loss = function(torch.clamp(image+perturbation, 0, 1), label)
        if config['target']:
            success = torch.argmax(logit, dim=1) == label
        else:
            success = torch.argmax(logit, dim=1) !=label
        last_loss.append(loss.item())

        if function.current_counts > 50000:
            break
        
        if bool(success.item()):
            return True, perturbation#torch.clamp(image+perturbation, 0, 1)

        nn.init.normal_(noise)
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]
        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        # if iter % config['log_interval'] == 0 and config['print_log']:
        if iter % config['log_interval'] == 0 :
            print("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()), float(torch.norm(perturbation))))

        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config['plateau_length']:]
        if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            if lr > config['lr_min']:
                lr = max(lr / config['lr_decay'], config['lr_min'])
            last_loss = []

    return False, perturbation#origin_image

   
def TREMBA_attack(tremba_dict):
    
    count_success = 0
    count_total = 0
    
    stats = {}

    torch.cuda.empty_cache()

    print("Starting attack")
    
    #function, encoder, decoder, image, label, config, latent=None
    with torch.no_grad():
        success, adv = EmbedBA(**tremba_dict)

    F = tremba_dict["function"]
    state = tremba_dict["config"]

    count_success += int(success)
    count_total += 1#int(correct)
    
    print("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(1, F.current_counts,
                                                                                           success,
                                                                                           F.get_average(),
                                                                                           float(
                                                                                               count_success) / float(
                                                                                               count_total)))
    F.new_counter()

    success_rate = float(count_success) / float(count_total)
    
    stats["F_current_counts"] = F.current_counts
    stats["success"] = success
    stats["F_average_eval_count"] = F.get_average()
    stats["success_rate"] = float(count_success) / float(count_total)
    
    return adv, success, stats, F.counts

#%%

def main(config):

#     config = json.load(open(opt["config"]))
#     import os
#     print(opt['gpuid'])
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(opt['gpuid'])) if type(opt['gpuid']) is list else opt['gpuid']

    ####################################### LOAD CONFIG #######################################################
    # Load the configurations

    

    #%%


    video_path = config["video_path"]

    video_name = config["video_name"]
    conv_model = config["source_model"]
    target_class = config["target"]

    TREMBA_path = config["generator_path"]
    # TREMBA_config = f"attack_target_{target_class}.json"

    generator_name = config["generator_name"]

    lower_frame = config["lower_frame"]
    upper_frame = config["upper_frame"]

    eps = config["epsilon"]


    config["DIM"] = 224
    #%%


    # Append the extension to this depending on whether it is the npy arrays or if it is the video
    save_path = f"{config['adv_save_path']}/{conv_model}_{target_class}_{video_name}"

    csv_save_path = f"{save_path}_{config['epsilon']}_run_summary.csv"
    run_done = Path(csv_save_path)
    if run_done.is_file():
        print(f"\nRun already found at {csv_save_path}, skipping")
        return  

    #%%

    print("Loaded configs")

    model_dict = {

        "resnet152": models.resnet152(pretrained=True),
        "densenet121": models.densenet121(pretrained=True),
        # "densenet161": models.densenet161(pretrained=True),
        "vgg16": models.vgg16(pretrained=True),
        # "vgg19": models.vgg19_bn(pretrained=True),
        # "inceptionv3": models.inception_v3(pretrained=True),
        # "googlenet": models.googlenet(pretrained=True),
        # "squeezenet": models.squeezenet1_1(pretrained=True),
        # "mnasnet": models.mnasnet1_0(pretrained=True),
        # "mobilenet_v2": models.mobilenet_v2(pretrained=True),
        # "nasnetalarge": pmodels.nasnetalarge(num_classes=1000, pretrained='imagenet'),
    }
    #%%


    ####################################### PRE-ATTACK CONFIGURATION ##############################################

    frames = skvideo.io.vread(f"{video_path}/{video_name}")
    print("Total frames: ", len(frames))

    # Artificially limit the amount of frames
    lower_frame = lower_frame
    upper_frame = upper_frame  # len(frames)

    frames = frames[lower_frame:upper_frame]

    plt.imshow(frames[0])
    #%%

    conv = model_dict[conv_model]
    conv.eval()


    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
        
    conv_model = nn.Sequential(
            Normalize(mean, std),
            conv
        )

    print("Got model")

    #%%

    tf_img_fn = TransformImage()#ptm_utils.TransformImage(conv)
    load_img_fn = PIL.Image.fromarray

    original = original_create_batches(frames_to_do=frames, batch_size=config["batch_size"], tf_img_fn=tf_img_fn,
                                        load_img_fn=load_img_fn)

    #%%

    # Get the original predictions
    with torch.no_grad():
        original_output = conv(original[0]).cuda()#.to(device))
        # print(original_output, original_output.shape)
        print("Original classes: ")
        for f in original_output:
            print(" {} ".format(np.argmax(np.round(f.detach().cpu().numpy()))), end='')
            
    #%%
    frames = torch.Tensor(frames).cuda().float()#.unsqueeze(0)

    # try:
    #     check= open(f"{TREMBA_path}/config/{TREMBA_config}")# as config_file
    #     check.close()
    # except:
    #     import shutil
    #     shutil.copy2(f"{TREMBA_path}/config/attack_target.json", f"{TREMBA_path}/config/{TREMBA_config}")
            

    # with open(f"{TREMBA_path}/config/{TREMBA_config}") as config_file:
        # state = json.load(config_file)

    nlabels = 1000

    # eps = 0.03125
    #         eps = 0.0625

    # Load up the pretrained generator
    generator_path = f"{config['generator_path']}/{config['generator_name']}"
    weight = torch.load(generator_path)#.cuda()##, map_location=device)

    print("Loaded generator")
    #%%
    # Get the encoder and decoder weights
    encoder_weight = {}
    decoder_weight = {}
    for key, val in weight.items():
        if key.startswith('0.'):
            encoder_weight[key[2:]] = val
        elif key.startswith('1.'):
            decoder_weight[key[2:]] = val

    encoder = Imagenet_Encoder()
    decoder = Imagenet_Decoder()
    encoder.load_state_dict(encoder_weight)
    decoder.load_state_dict(decoder_weight)

    encoder.cuda()##.to(device)
    encoder.eval()
    decoder.cuda()##.to(device)
    decoder.eval()

    print("Encoder, decoder loaded")
    #%%

    F = Function(conv, config['batch_size'], config['margin'], nlabels, config['target'])
    F.cuda()

    print("F function loaded")
    #%%


    if config['target']:
        labels = config['target']
    #         labels = torch.Tensor(np.array([labels])).cuda()
    #function, encoder, decoder, image, label, config, latent=None
    tremba_dict = {
        "function": F,
        "encoder": encoder,
        "decoder": decoder,
        "label": labels,

        "config":config,
        "latent":None,
    }
        
    adversarial_images = []
    adversarial_frames = []
    #%%

    ####################################### ATTACK #######################################################
    pd_array = []

    column_names = ["conv_model", "video_name", "epsilon", "frame_num", "target", "success", "average_count", "success_rate", "time"]
    tic = time.time()
    # Launching the attack over each frame
    for f in range(len(frames)):
        print("\n-------------------------------\nFrame number: ", f, end='\n')

        image = (create_batches(frames[f].unsqueeze(0), config["DIM"]) / 255.)
    #         image.cuda()##.to(device)

        tremba_dict['image'] = image.squeeze(0).cuda()#[0].cuda()

        adv_frames, success, stats, F.counts = TREMBA_attack(tremba_dict=tremba_dict)

        adv_im = torch.clamp(image + adv_frames.unsqueeze(0), 0, 1)

        adv_frames = detach(adv_frames.unsqueeze(0))
        adversarial_images.append(detach(adv_im))
        adversarial_frames.append(adv_frames)

        row_array = [config["source_model"], video_name, config['epsilon'], f, target_class, stats["success"], stats["F_average_eval_count"], stats["success_rate"], ""]
                        
        pd_array.append(row_array)

    print("Attack done")
    toc=  time.time()

    attack_time = toc-tic
    time_row = [config["source_model"], video_name, config['epsilon'], f, target_class, stats["success"], stats["F_average_eval_count"], stats["success_rate"], attack_time]

    pd_array.append(time_row)
    print(f"Attack took: {toc-tic} seconds")
    #%%
    ####################################### POST-ATTACK ##################################################

    adversarial_frames = np.concatenate(adversarial_frames, axis=0)
    adversarial_images = np.concatenate(adversarial_images, axis=0)

    # plt.imshow(torch.Tensor(adversarial_frames[0]).permute(1,2,0))
    # plt.show()

    #     plt.imshow(torch.Tensor(adversarial_images[0]).permute(1,2,0))
    #     plt.show()
    #%%
    try:
        import os
        print("Creating save path: ", config["adv_save_path"])
        os.makedirs(config["adv_save_path"])
    except:
        print("Save path already exists, skipping creation")
    #%%

    df = pd.DataFrame(pd_array, columns=column_names)
    csv_save_path = f"{save_path}_{config['epsilon']}_run_summary.csv"
    df.to_csv(csv_save_path, index=False)
    print("Pandas results table saved at: ", csv_save_path)

    np_save_path = f"{save_path}_{config['epsilon']}_perturbations.npy"
    np.save(np_save_path, adversarial_frames * 255.)
    print(f"Adversarial Perturbations Saved at: {np_save_path}")

    np_save_path = f"{save_path}_{config['epsilon']}_adv_images.npy"
    np.save(np_save_path, adversarial_images * 255.)
    print(f"Adv. Images Saved at: {np_save_path}")

    adv_save_path = f"{save_path}_{config['epsilon']}_adv_video.avi"
    print(f"Writing adversarial video to: {adv_save_path}")
    # Writing the adversarial frames to video
    writer = skvideo.io.FFmpegWriter(adv_save_path, outputdict={
        '-c:v': 'huffyuv',  # r210 huffyuv r10k
    })

    for f in adversarial_images:
        writer.writeFrame(f * 255.)

    writer.close()
    print("Finished writing adversarial video.")


#%%
if __name__ == '__main__':

    config_path = "configs/attack"
    configs = os.listdir(config_path)
    for c in range(len(configs)):
        print(f"Config {c}/{len(configs)}")
        cpath = configs[c]
        
            
        config_name = os.path.join(config_path, cpath)
        with open(config_name, 'r') as reader:
            config = json.load(reader)
        # if config['target'] == 701:
        #     continue
        if config['epsilon'] == 0.0625:
            config['epsilon'] += 0.03125  
        if config['epsilon'] == 0.03125:
            config['epsilon'] *= 2
 
        main(config)