# Source: TREMBA Repo at  https://github.com/TransEmbedBA/TREMBA

# Modified generator 

import sys 
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataloader import *
import numpy as np
from .FCN import *
from .utils import *
import torchvision.models as models
from .imagenet_model.Resnet import *

def train_tremba(state):

    device = torch.device(int(state['gpuid'][0]))
    print("Loaded device" , device)
    
    # Load Imagenet dataloaders
    train_loader, test_loader, nlabels, mean, std = imagenet(state)
    print(f"Obtained train and test loaders")
    
    nets = []
    
    model_dict = {

        "resnet152": models.resnet152,
        "densenet121": models.densenet121,
        # "densenet161": models.densenet161,
        "vgg16": models.vgg16,
        # "inceptionv3": models.inception_v3,
        "googlenet": models.googlenet,
        "squeezenet": models.squeezenet1_1,
        # "mnasnet": models.mnasnet1_0,
        "mobilenet_v2": models.mobilenet_v2,
        "resnet18": models.resnet18,
        # "vgg19": models.vgg19_bn,
        # "resnext101": models.resnext101_32x8d,
    }
    
    # Get the test set size and number of batches
    print("Obtaining test set size")

    total_length = len(test_loader.dataset)
    batch_length = len(test_loader)
    # for batch_idx, a in enumerate(test_loader):

    #     data = a[0]['data']
        
    #     if count < 1:
    #         batch_length += len(data)
            
    #     total_length += len(data)
    #     count = count+1
    
    print(f"Total test length: ", total_length)
    print(f"Total test batch length: ", batch_length)
    # test_loader.reset()
    
    # Instantiate each model in the model list (list of models to use in training ensemble)
    for model_name in state['model_list']:
        print("Loading ensemble model: ", model_name)
        
        if state["use_pretrained"]:
            pretrained_model = model_dict[model_name.lower()](pretrained=True)
        else:
            # pretrained_path = f"{state['retrained_cnn_path']}/imagenet_{model_name.lower()}_seed-{state['seed']}.ckpt"
            net = model_dict[model_name.lower()](pretrained=True)
            net = torch.nn.DataParallel(net)

            net = nn.Sequential(
                Normalize(mean, std),
                net
            )
#             net.cuda()
#             net.train()
    
        nets.append(net)

    # Send each model to GPU
    for i in range(len(nets)):
        nets[i] = torch.nn.DataParallel(nets[i]).cuda()#, state['gpuid'])
        nets[i].eval()
        nets[i].cuda()#.to(device)

    # Initialize Generator and send to CUDA device
    model = nn.Sequential(
        Imagenet_Encoder(),
        Imagenet_Decoder()
    )

    model = torch.nn.DataParallel(model).cuda()#, state['gpuid'])
    
    # If target is given, change save path
    # if state['target']:
    #     save_name = state['save_name']#"Imagenet_{}{}_target_{}.pytorch".format("_".join(state['model_list']), state['target_class'])
    # else:
    #     save_name = state['save_name']#"Imagenet_{}{}_untarget.pytorch".format("_".join(state['model_list']))

    save_name = state['save_name']

    if state['pretrained_generator']:
        try:
            # model.module.load_state_dict(torch.load(f"{state['generator_path']}/{save_name}"))
            model.module.load_state_dict(torch.load(f"{state['generator_path']}/{save_name}"))
            print("Loaded previous model weights successfully")
        except:
            print("No previous model or unable to load model.")
            print(sys.exc_info()[1])
        
        model.cuda()#.to(device)

    # print(model)

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'][0],
                                                gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])
    
    def train():

        model.train()
        train_len = len(train_loader.dataset)
        train_batches = len(train_loader)
        print(f"There are {train_batches} number of batches.")
        #for batch_idx, (data, label) in enumerate(train_loader):
        st = time.time()
        for batch_idx, a in enumerate(train_loader):

            # print(f"Batch {batch_idx} / {train_batches}")
            data = a[0]#['data']
            label = a[1]#['label']
            
            nat = data.cuda()#.to(device)
            
            # If target exists, we want to make it a targeted attack to match target class
            if state['target']:
                label = state['target']
            else:
                label = label.cuda()#.to(device)

            losses_g = []
            optimizer_G.zero_grad()
            for net in nets:
                noise = model(nat)
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()
            if (batch_idx + 1) % state['log_interval'] == 0:
                ed = time.time()
                print("batch {}, time {}, losses_g {}".format(batch_idx + 1, ed-st, dict(zip(state['model_list'], losses_g))))
                st = time.time()
#             print(batch_idx, end=" ")
            
            
    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]

        # for batch_idx, (data, label) in enumerate(test_loader):
        for batch_idx, a in enumerate(test_loader):
        
#             print(batch_idx, end=" ")
            data = a[0]#['data']
            label = a[1]#['label']
            
            nat = data.cuda()#to(device)
            if state['target']:
                label = state['target']
            else:
                label = label.cuda()#.to(device)
            noise = model(nat)
            adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
            
            for j in range(len(nets)):
                logits = nets[j](adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                if state['target']:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())

        # First one is batch size, second one is 1000
        state['test_loss'] = [loss_avg[i] / batch_length for i in range(len(loss_avg))]
        state['test_successes'] = [success[i] / total_length for i in range(len(success))]
        state['test_success'] = 0.0
        for i in range(len(state['test_successes'])):
            state['test_success'] += state['test_successes'][i]/len(state['test_successes'])

    best_success = 0.0
    for epoch in range(state['epochs']):
        print(f"Epoch {epoch}/{state['epochs']}")

        scheduler_G.step()
        state['epoch'] = epoch
        train()
        
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            with torch.no_grad():
                test()
        # train_loader.reset()
        # test_loader.reset()
        
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))
        torch.save(model.module.state_dict(), os.path.join(f"{state['generator_path']}/", save_name))
        with open(f"{state['generator_path']}/{save_name}_log.txt", 'w') as f:
            f.write(f"{epoch}\n")
            
        print("Saved at", os.path.join(f"{state['generator_path']}/", save_name))
