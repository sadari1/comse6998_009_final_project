#%%
import skvideo
skvideo.setFFmpegPath("C:\\Users\\suman\\ffmpeg\\ffmpeg-2022-05-04-git-0914e3a14a-full_build\\bin")
import skvideo.io 

import numpy as np
import PIL
# from Attack_Models.CNN_Attack import Attack
# import skvideo.io 
import json 
import torchvision.models as models
from TREMBA.FCN import *
from TREMBA.utils import *
from TREMBA.dataloader import *
import pandas as pd
from pretrainedmodels import utils as ptm_utils
from utils import * 

from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from video_caption_pytorch.process_features import process_batches
from video_caption_pytorch.process_features import create_batches as create_batches_eval
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils

#%%
def evaluate(config):

# cpath = 'configs/eval'
# cname = os.listdir(cpath)[0]
# cname = os.path.join(cpath, cname)
# with open(cname, 'r') as reader:
#     config = json.load(reader)


    video_name = config["video_name"]
    conv_model = config["source_model"]
    conv_model_name = conv_model
    target_class = config["target"]

    TREMBA_path = config["generator_path"]
    # TREMBA_config = f"attack_target_{target_class}.json"

    generator_name = config["generator_name"]

    lower_frame = config["lower_frame"]
    upper_frame = config["upper_frame"]

    eps = config["epsilon"]

    eval_model_path = config['eval_model_path']
    model_opt_name = config["eval_cnn"]

    video_path = config["video_path"]
    # model_opt_name = "resnet152"
    # model_opt_name = "vgg16"

    # modelname = 'nasnetalarge'

    #%%
    model_dict = {

        "resnet152": models.resnet152(pretrained=True),
        "vgg16": models.vgg16(pretrained=True),
    }

    model_name = conv_model
    #     modelname = 'resnet152'
    # modelname = 'vgg16'
    # modelname = 'densenet121'
    # modelname = 'googlenet'
    # modelname = 'squeezenet'
    # modelname = 'mobilenet'
    # modelname = 'mnasnet'
    # modelname = 'inceptionv3'

    # Append the extension to this depending on whether it is the npy arrays or if it is the video
    if not os.path.exists(config['results_dir']):
        os.makedirs(config['results_dir'])

    save_path = f"{config['results_dir']}/{conv_model}_{target_class}_{video_name}_{model_opt_name}"

    #     selection = 9

    #     target_class = 805

    opt_path = config["opt_path"]
    # opt_file_path = f"{opt_path}{model_opt_name}/opt_info.json"
    opt_file_path = config["opt_path"]#f"{opt_path}{model_opt_name}/opt_info.json"

    opt = json.load(open(opt_file_path))

    #%%
    opt['rnn-type'] = 'gru'
    opt['dump_json'] = 1
    opt['results_path'] = 'results/'
    opt['gpu'] = '0'
    opt['batch_size'] = 128
    opt['sample_max'] = 1
    opt['dump_path'] = 0
    opt['temperature'] = 1.0
    opt['beam_size'] = 1

    saved_model_path = f"{config['eval_model_path']}/msvd_{model_opt_name}/{model_opt_name}_model.pth"

    opt['saved_model'] = saved_model_path

    dataset = VideoDataset(opt, 'inference')
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len

    opt['bidirectional'] = bool(opt['bidirectional'])
    if opt['beam_size'] != 1:
        assert opt["batch_size"] == 1
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"], opt['dim_vid'],
                            n_layers=opt['num_layers'],
                            rnn_cell=opt['rnn_type'],
                            bidirectional=opt["bidirectional"],
                            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"],
                                n_layers=opt['num_layers'],
                                rnn_cell=opt['rnn_type'], bidirectional=opt["bidirectional"],
                                input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                                n_layers=opt['num_layers'],
                                rnn_cell=opt['rnn_type'], input_dropout_p=opt["input_dropout_p"],
                                rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    else:
        pass 
        # return

    #%%

    convnet = model_opt_name

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    conv_model = model_dict[convnet]#pretrainedmodels.__dict__[convnet](num_classes=1000, pretrained='imagenet')

    vocab = dataset.get_vocab()

    #%%

    full_decoder = ConvS2VT(convnet, model, opt)

    # adv_path = f"{config['save_dir']}/{conv_model_name}_{target_class}_{video_name}"
    adv_path = f"{config['adv_save_path']}/{conv_model_name}_{target_class}_{video_name}"

    original_video_path = f"{video_path}/{video_name}"

    adversarial_frames_np_path = f"{adv_path}_{config['epsilon']}_adv_images.npy"

    adversarial_video_path = f"{adv_path}_{config['epsilon']}_adv_video.avi"

    adv_frames = np.load(adversarial_frames_np_path)

    tf_img_fn = ptm_utils.TransformImage(full_decoder.conv)
    load_img_fn = PIL.Image.fromarray

    print(video_path)
    #%%
    #     new_classifier = nn.Sequential(*list(conv_model.classifier.children())[:-1])
    #     conv_model.classifier = new_classifier

    pd_array = []
    column_names = ["video_name", "source_model", "eval_model", "epsilon", "original_caption", "target", "adversarial_caption_np", "adversarial_caption_video"]

    with torch.no_grad():
        frames = skvideo.io.vread(original_video_path)[lower_frame:upper_frame]#[:len(adv_frames)]
        batches = create_batches_eval(frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)
        original_caption = sents[0]
        print("Original Caption: ", original_caption)
    #         conv_model.cuda()
    #         original_feats = conv_model.features(batches[0].cuda()).detach().cpu().numpy()
        
        print("Numpy CNN frames \nTotal frames: {}".format(len(adv_frames)))
        # print(frames[[0, 1, 2, 3, 4, 5]].shape)
    #         plt.imshow(torch.Tensor(adv_frames[0]/255.).permute(1,2,0))
    #         plt.show()

        # bp ---
        # batches = create_batches(adv_frames/255., load_img_fn, tf_img_fn)

        # batches = create_batches(torch.Tensor(adv_frames).permute(0,2,3,1).numpy().astype(np.uint8), load_img_fn, tf_img_fn)

        normed = Normalize(mean, std)(torch.Tensor(adv_frames/255.))

        batches = create_batches(torch.Tensor(adv_frames).permute(0, 2,3,1), DIM=224, batch_size=config['batch_size'])#/255.
        # batches = [batches]
        # Post normalization
    #         plt.imshow(normed[0].permute(1,2,0))
    #         plt.show()
        
        
        
    #         adv_feats = conv_model(normed.cuda()).detach().cpu().numpy()
        
        seq_prob, seq_preds = full_decoder([normed], mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)
        adv_numpy_caption = sents[0]
        print("Adversarial numpy: ", adv_numpy_caption)

        frames = skvideo.io.vread(adversarial_video_path)[lower_frame:upper_frame]
        batches = create_batches_eval(frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)
        adv_video_caption = sents[0]
        print("Adversarial Video Caption: ", adv_video_caption)
    #         conv_model.cuda()
        row_array = [video_name, model_name, model_opt_name, config['epsilon'], original_caption, target_class,
                        adv_numpy_caption,adv_video_caption]
        
        pd_array.append(row_array)
        
    eval_results_df = pd.DataFrame(pd_array, columns= column_names)
    csv_save_path = f"{save_path}_{config['epsilon']}_eval_summary.csv"
    eval_results_df.to_csv(csv_save_path, index=False)
    print("Pandas results table saved at: ", csv_save_path)

# %%

config_path = "configs/eval"
for c in os.listdir(config_path):
    config_name = os.path.join(config_path, c)
    with open(config_name, 'r') as reader:
        config = json.load(reader)

    evaluate(config)
    break 

#%%

if __name__ == '__main__':

    config_path = "configs/eval"
    # config_name = os.listdir(config_path)[2]
    config_name = os.path.join(config_path, config_name)
    with open(config_name, 'r') as reader:
        config = json.load(reader)
    
    main(config)