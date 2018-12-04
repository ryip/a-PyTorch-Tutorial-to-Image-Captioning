import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import glob
import time
from netdissect.broden import BrodenDataset
import pickle
import netdissect.dissection as dissection
import os
from random import shuffle



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)



    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # ( s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

def caption_image_beam_search_awes(encoder, decoder, image_path, word_map, beam_size=3,all_captions = False):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)



    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    #LAST LAYER HAS 2048 CHANNELS
    seqs_awe = torch.ones(k, 1, 2048).to(device)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_awe = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # ( s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

        awe = gate * awe
        # print(awe[0])
        # print(np.linalg.norm(awe[0].detach()))
        # print(np.linalg.norm(awe[0].detach(),1))
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        seqs_awe = torch.cat([seqs_awe[prev_word_inds], awe[prev_word_inds].unsqueeze(1)],dim=1)  # (s, step+1, enc_image_size, enc_image_size)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_awe.extend(seqs_awe[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    # print("scores:",complete_seqs_scores)
    # print(sorted(range(len(complete_seqs_scores)),key = lambda x: complete_seqs_scores[x]))
    # print(complete_seqs)
    # print(seq)
    # input("Press Enter to continue...")
    alphas = complete_seqs_alpha[i]
    awes = complete_seqs_awe[i]
    normalize = False
    if normalize:
        normalized_awes = [[x*len(awe) / sum(awe) for x in awe] for awe in awes]
    else:
        normalized_awes = awes
    if all_captions:
        top_captions = sorted(range(len(complete_seqs_scores)),key = lambda x: complete_seqs_scores[x],reverse=True)
        sorted_seqs = [complete_seqs[ind] for ind in top_captions]
        return seq, alphas, normalized_awes, sorted_seqs
    return seq, alphas, normalized_awes

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

# =========================================================
#                 RYIP DEFINED METHODS
# ==========================================================

def observe_single_channel(encoder,decoder,channel_num):
    dissection.replace_layers(encoder, [('resnet.7.2', 'output_layer'), ])
    replacement = torch.zeros(2048)
    oneval = [0, 1, 10, 100, 1000, 10000]
    for i in range(len(oneval)):
        replacement[channel_num] = oneval[i]
        encoder.replacement['output_layer'] = replacement.to(device).type(torch.cuda.FloatTensor)
        word_mat = gather_word_data(encoder, decoder, img_list[0], word_map, args.beam_size)
        # seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
        # alphas = torch.FloatTensor(alphas)
        # visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
        for word in word_mat:
            st = ""
            for w in word:
                st += w + " "

            print(st)


def gather_word_data(encoder, decoder, img_list, word_map, beam_size,include_awes=False,all_captions = False):
    if not isinstance(img_list,list):
        img_list = [img_list]
    sentences = []
    rev_word_map = {v: k for k, v in word_map.items()}
    for img in img_list:
        try:
            if all_captions:
                seq, alphas, awes, all_seqs = caption_image_beam_search_awes(encoder, decoder, img, word_map, beam_size,True)
                for s in all_seqs:
                    sentences.append([rev_word_map[i] for i in s])
                #print(sentences)
            else:
                seq, alphas, awes = caption_image_beam_search_awes(encoder, decoder, img, word_map, beam_size)
                sentences.append([rev_word_map[i] for i in seq])
        # alphas = torch.FloatTensor(alphas)
        # Visualize caption and attention of best sequence

        except ValueError:
            print("no caption")
            sentences.append([])
    if include_awes:
        return sentences,awes
    return sentences

def ablate_one_channel(encoder,decoder,img_to_test,channel_num):
    #channel = 106  # 23 = bathrooms
    channel = channel_num
    dissection.retain_layers(encoder, [('resnet.7.2', 'output_layer')])
    dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')])
    # dissection.replace_layers(encoder, [('resnet.7.2', 'output_layer')])
    # replacement = torch.zeros(2048)
    # replacement[channel] = 100
    ablation = torch.ones(2048)
    ablation[channel] = 0
    # print(ablation)
    encoder.ablation['output_layer'] = ablation.to(device).type(torch.cuda.FloatTensor)

    dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')], adding=True)
    addition = torch.zeros(2048)
    addition[channel] = 50
    encoder.ablation['output_layer'] = addition.to(device).type(torch.cuda.FloatTensor)

    viz = True
    if viz:
        seq, alphas = caption_image_beam_search(encoder, decoder, img_to_test, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)

        visualize_att(img_to_test, seq, alphas, rev_word_map, args.smooth)
    word_mat = gather_word_data(encoder, decoder, img_to_test, word_map, args.beam_size)
    encoder_output = encoder.retained['output_layer'][0]
    print(encoder_output[channel])

    for word in word_mat:
        print(word)

def save_channels_per_word(encoder,decoder,image_list,num_images = None,scale_by_attention=False,pkl_location = "word_to_tensors_default.p"):
    if num_images is None:
        num_images = len(image_list)
    truncated_img_list = image_list[:num_images]
    dissection.retain_layers(encoder, [('resnet.7.2', 'output_layer')])
    word_to_tensors = {}  # word -> [list of images, list of avg_channels]
    # Note that <start> and <end> will have all imgs and all avg channels

    # timing
    t = time.time()
    print("adding image captions to dictionary")
    #for img in truncated_img_list:
    for i in range(len(truncated_img_list)):
        if True:#i%100==0:
            print("image ",i,"/",len(truncated_img_list))
        img = truncated_img_list[i]
        # dissection.retain_layers(encoder, [('resnet.7.2', 'output_layer')])
        word_mat, awes = gather_word_data(encoder, decoder, img, word_map, args.beam_size, include_awes=True)
        if not scale_by_attention:
            awes = np.ones_like(awes)
        encoder_output = encoder.retained['output_layer'][0]
        # for word in word_mat:
        #    print(word)
        # print(encoder_output,encoder_output.size())
        avg_channels = encoder_output
        #print(avg_channels.type())
        #print(avg_channels.size())
        word_mat = word_mat[0]


        wds_to_sum_awes = {}
        for wd_index in range(len(word_mat)):
            word = word_mat[wd_index]
            awe = awes[wd_index]
            if word not in wds_to_sum_awes:
                wds_to_sum_awes[word] = awe
            else:
                wds_to_sum_awes[word] = np.add(awe,wds_to_sum_awes[word])

        for word in wds_to_sum_awes:
            word_awe = wds_to_sum_awes[word]
            #avg_channels = np.multiply(np.array(encoder_output).transpose(), word_awe).transpose()
            #avg_channels = np.array(encoder_output)

            #avg_channels = (encoder_output).type(torch.cuda.FloatTensor)
            #avg_channels = torch.ones_like(encoder_output).type(torch.cuda.FloatTensor)
            if word not in word_to_tensors:
                word_to_tensors[word] = [[img], [avg_channels],[word_awe]]
            else:
                word_to_tensors[word][0].append(img)
                word_to_tensors[word][1].append(avg_channels)
                word_to_tensors[word][2].append(word_awe)
    #word_freqs = sorted([[x, len(word_to_tensors[x][0])] for x in word_to_tensors], key=lambda x: x[1], reverse=True)
    toc = time.time() - t
    print("images: ", num_images, " time: ", toc, " time per img: ", toc / num_images)
    # for word in word_freqs:
    #    print(word)

    # print(encoder_output[0])
    # for word in word_to_tensors:
    #     print(word,np.shape(word_to_tensors[word][2]))

    #directory to pickle to:
    pkled_directory = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/" + pkl_location
    print("pickling to: ", pkled_directory)
    pickle.dump(word_to_tensors, open(pkled_directory , "wb"))
    # print("unpickling")
    # nd = pickle.load(open("word_to_tensors.p","rb"))
    # print(nd)

def search_for_channel_topics(pkl_location = "word_to_tensors_default.p"):
    # Dictionary mapping words to [[img],[avg_channels]] for each img that generates the word
    pkled_directory = os.path.dirname(__file__) + "/word_to_tensors_dicts/" + pkl_location
    print("loading pickled dict from: ", pkled_directory)

    nd = pickle.load(open(pkled_directory, "rb"))
    x = sorted(nd.keys(), key=lambda x: len(nd[x][0]), reverse=True)

    # Print out top 10 words.
    for i in range(10):
        print(x[i], len(nd[x[i]][0]))
    bestword = nd[x[0]][1]
    bestword2 = nd[x[9]][1]
    aa = []

    bb = []
    for i in range(len(bestword[0])):
        # print(i,len(bestword[0]))
        aa.append(sum([wd[i] for wd in bestword[:100]]) / len(bestword))
        bb.append(sum([wd[i] for wd in bestword2[:100]]) / len(bestword2))
    # print(aa)

    s = [nm.item() for nm in aa]
    q = [nm.item() for nm in bb]
    # plt.plot(s)
    # plt.plot(q)
    plt.plot([s[i] - q[i] for i in range(len(s))])
    plt.show()

def plot_word_activations(concept_to_word_list,word_to_tensor_pkl="word_to_tensors_default.p"):
    pkled_directory = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/" + word_to_tensor_pkl
    nd = pickle.load(open(pkled_directory, "rb"))

    sorted_word_frequencies = sorted(nd.keys(), key=lambda x: len(nd[x][0]), reverse=True)
    # Print out top 10 words.
    print("Words and their frequencies:")
    print("============")
    for i in range(50): print(sorted_word_frequencies[i], len(nd[sorted_word_frequencies[i]][0]))


    avg_all = nd["<start>"][1]
    avg_all_output = np.mean([np.array(c) for c in avg_all], axis=0)
    avg_per_channel = np.mean([np.array(c) for c in avg_all], axis=(2,3))
    avg_stdev = np.std(avg_per_channel,axis=0)
    print(avg_stdev[:10])

    concept_to_avgs = {}

    plt.figure(1)
    plt.xlabel("Rank of Channel Intensity")
    plt.ylabel("Relative intensity")
    plt.title("Intensity of channels for target word relative to baseline")
    plt.figure(2)
    plt.xlabel("Rank of Channel Intensity")
    plt.ylabel("Relative intensity")
    plt.title("Most Active Channels")
    plt.figure(3)
    plt.xlabel("Rank of Channel Intensity")
    plt.ylabel("Relative intensity")
    plt.title("Least Active Channels")

    for concept,word_list in concept_to_word_list.items():
        print("word list: ",word_list)
        avg_tgt_word_outputs = []
        for target_word in word_list:
            print("%s has %d examples in dataset"%(target_word,len(nd[target_word][0])))
            if target_word not in nd: continue

            avg_channels = nd[target_word][1]
            avg_awe_word = nd[target_word][2]

            for i in range(len(avg_channels)):
                avg_channels[i] = np.multiply(np.array(avg_channels[i]).transpose(), avg_awe_word[i]).transpose()
            avg_tgt_word_outputs.append(np.mean([np.array(c) for c in avg_channels], axis=0))


        #Weighted average of the responses of the words in the wordlist
        avg_tgt_word_output_final = np.average(avg_tgt_word_outputs,weights=[len(nd[x][0]) for x in word_list],axis=0)
        concept_to_avgs[concept] = avg_tgt_word_output_final

        def channel_significance_function(x):
            return np.sum(avg_tgt_word_output_final[x] - avg_all_output[x]) / avg_stdev[x]

        channels_by_significance = sorted(range(len(avg_all_output)),
                                          key=channel_significance_function, reverse=True)

        #Length of most and least active channel windows to look at
        hlength = 10
        tlength = 30

        plt.figure(1)
        plt.plot([channel_significance_function(x) for x in channels_by_significance],label = concept)
        plt.legend()

        plt.figure(2)
        most_activated = [channel_significance_function(x) for x in channels_by_significance][:hlength]
        plt.plot(most_activated,label = concept)
        print("most active channels: ",channels_by_significance[:10])
        print("Avg of top %s significant channels: %f"%(3,sum(most_activated[:3])/3.))
        for i in range(5):
            plt.annotate(channels_by_significance[i],xy = (i,most_activated[i]),xytext=(i,most_activated[i]))
        plt.legend()

        plt.figure(3)
        least_activated = [channel_significance_function(x) for x in channels_by_significance][-tlength:]
        plt.plot(least_activated,label = concept)
        print("least active channels: ", channels_by_significance[-10:])
        for j in range(10):
            i = j+1
            #plt.annotate(channels_by_significance[-i], xy=(tlength-i, least_activated[-i]), xytext=(tlength-i, least_activated[-i]))
        plt.legend()

    plt.show()
        # plt.plot([np.sum(avg_tgt_word_output_final[x] - avg_all_output[x]) for x in channels_by_significance])
        # plt.xlabel("Channel (sorted by decreasing relative intensity)")
        # plt.ylabel("Relative intensity")
        # plt.title("Intensity of channels (weighted by attention) for target word relative to baseline")
        # plt.show()
        #
        # plt.plot([np.sum(avg_tgt_word_output_final[x] - avg_all_output[x]) for x in channels_by_significance][:100])
        # plt.xlabel("Channel (sorted by decreasing relative intensity)")
        # plt.ylabel("Relative intensity")
        # plt.title("Front End")
        # plt.show()
        #
        # plt.plot([np.sum(avg_tgt_word_output_final[x] - avg_all_output[x]) for x in channels_by_significance][-100:])
        # plt.xlabel("Channel (sorted by decreasing relative intensity)")
        # plt.ylabel("Relative intensity")
        # plt.title("Tail End")
        # plt.show()

def insert_topic_into_caption(encoder,decoder,target_word,word_to_tensor_pkl="word_to_tensors_default.p",num_imgs_to_test=None,viz=False,randomize=False):
    pkled_directory = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/" + word_to_tensor_pkl
    nd = pickle.load(open(pkled_directory, "rb"))

    sorted_word_frequencies = sorted(nd.keys(), key=lambda x: len(nd[x][0]), reverse=True)
    # Print out top 10 words.
    for i in range(30): print(sorted_word_frequencies[i], len(nd[sorted_word_frequencies[i]][0]))

    print("========")
    print("target word ~ %s ~ appears %d times" % (target_word,len(nd[target_word][0])))

    # avg_channels = np.multiply(np.array(encoder_output).transpose(), word_awe).transpose()
    avg_channels = nd[target_word][1]
    avg_awe_word = nd[target_word][2]
    t = time.time()
    for i in range(len(avg_channels)):
        avg_channels[i] = np.multiply(np.array(avg_channels[i]).transpose(), avg_awe_word[i]).transpose()
    toc = time.time() - t

    #print("time: ",toc)


    avg_all = nd["<start>"][1]


    images_not_containing_target = [im for im in nd["<start>"][0] if im not in nd[target_word][0]]
    #####images_not_containing_target = nd[target_word][0]


    if randomize:
        shuffle(images_not_containing_target)

    avg_tgt_word_output = np.mean([np.array(c) for c in avg_channels], axis=0)
    avg_all_output = np.mean([np.array(c) for c in avg_all], axis=0)
    #print(np.linalg.norm(avg_tgt_word_output[0] - avg_all_output[0]))

    # Number of channels to change. Set to 0 to generate original image
    # ======================

    num_channels = 10

    #channels_by_significance = sorted(range(len(avg_all_output)), key=lambda x: np.linalg.norm(avg_tgt_word_output[x] - avg_all_output[x],1), reverse=True)
    channels_by_significance = sorted(range(len(avg_all_output)), key=lambda x: np.sum(avg_tgt_word_output[x] - avg_all_output[x]), reverse=True)
    top_n_channels = channels_by_significance[:num_channels] + channels_by_significance[-num_channels:]

    # plt.plot([np.sum(avg_tgt_word_output[x] - avg_all_output[x]) for x in channels_by_significance])
    # plt.xlabel("Channel (sorted by decreasing relative intensity)")
    # plt.ylabel("Relative intensity")
    # plt.title("Intensity of channels for target word relative to baseline")
    # plt.show()
    #
    # plt.plot([np.sum(avg_tgt_word_output[x] - avg_all_output[x]) for x in channels_by_significance][:100])
    # plt.xlabel("Channel (sorted by decreasing relative intensity)")
    # plt.ylabel("Relative intensity")
    # plt.title("Front End")
    # plt.show()
    #
    # plt.plot([np.sum(avg_tgt_word_output[x] - avg_all_output[x]) for x in channels_by_significance][-100:])
    # plt.xlabel("Channel (sorted by decreasing relative intensity)")
    # plt.ylabel("Relative intensity")
    # plt.title("Tail End")
    # plt.show()

    # print(top_n_channels)

    original_and_modified = []

    if num_imgs_to_test is None:
        num_imgs_to_test = len(images_not_containing_target)
    final_num_images_to_test = min(num_imgs_to_test,len(images_not_containing_target))
    for i in range(final_num_images_to_test):
        current_img = images_not_containing_target[i]
        word_mat = gather_word_data(encoder, decoder, current_img, word_map, args.beam_size,all_captions=True)
        original_and_modified.append([word_mat])


    # NETDISSECT STARTS HERE
    dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')])
    ablation = torch.ones(1,2048,8,8)

    ##ONLY FOR BLOCKING OUT "IN"

    for channel in top_n_channels:
        ablation[0][channel] = torch.zeros_like(ablation[0][channel])
    encoder.ablation['output_layer'] = ablation.to(device).type(torch.cuda.FloatTensor)

    dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')], adding=True)
    replacement = torch.zeros(1,2048,8,8)
    for channel in top_n_channels:

        #SCALING FACTOR
        scaling_factor = .2

        replacement[0][channel] = torch.tensor(avg_tgt_word_output[channel]/scaling_factor)

    encoder.ablation['output_layer'] = replacement.to(device).type(torch.cuda.FloatTensor)

    print("encoder updated!")
    for i in range(final_num_images_to_test):
        current_img = images_not_containing_target[i]
        if viz:
            seq, alphas, awes = caption_image_beam_search_awes(encoder, decoder, current_img, word_map, args.beam_size)
            #print(alphas)

            alphas = torch.FloatTensor(alphas)
            #print(alphas.shape)
            print("==ORIGINAL==")
            print(original_and_modified[i][0][0])
            visualize_att(current_img, seq, alphas, rev_word_map, args.smooth)
        word_mat = gather_word_data(encoder, decoder, current_img, word_map, args.beam_size)
        original_and_modified[i].append(word_mat[0])
        o_set = set()
        for capt in original_and_modified[i][0]:
            o_set.update([wd for wd in capt])


        m_set = set(original_and_modified[i][1]).difference(set([target_word]))
        score = len(o_set.intersection(m_set))/len(m_set)
        original_and_modified[i].append(score)
        #print(score)

    all_scores = [x[2] for x in original_and_modified if target_word in x[1]]
    avg_change = np.mean(all_scores)
    num_inserted = len(all_scores)
    print("target word: ", target_word, "num channels: ", num_channels)
    print("num inserted: ", num_inserted, "num_tested: ", num_imgs_to_test, "ratio: ", num_inserted / num_imgs_to_test, "avg word consistency: ",avg_change)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--method', '-md', default=None, help='method to run')

    args = parser.parse_args()

    #Richard-specific variables
    args.model = 'C:/Users/Richard/Documents/MIT stuff/2018 Fall/research/a-PyTorch-Tutorial-to-Image-Captioning/pretrained/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    args.word_map = 'C:/Users/Richard/Documents/MIT stuff/2018 Fall/research/a-PyTorch-Tutorial-to-Image-Captioning/pretrained/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    #args.image = 

    # Load model
    checkpoint = torch.load(args.model)
    #print(checkpoint)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']


    #print(encoder)

    encoder.eval()
    encoder = encoder.to(device)



    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    #print(len(word_map))
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    img_list = glob.glob(
        'C:/Users/Richard/Documents/MIT stuff/2018 Fall/research/a-PyTorch-Tutorial-to-Image-Captioning/img/flickr8k_dataset/*.jpg')
    one_img = img_list[201]

    do_original = False
    if do_original:
        seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)

        visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)


    method = "plot_word_activations"

    if args.method is not None: method = args.method

    print("Executing method: ",method)

    if method == "other":
        pass
        # pkl_dir = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/"
        # d1 = pickle.load(open(pkl_dir + "word_to_tensors_1000_nonnormalized.p", "rb"))
        # d2 = pickle.load(open(pkl_dir + "word_to_tensors_1000_nonnormalized_pt2.p", "rb"))
        # all_words = set(list(d1.keys()) + list(d2.keys()))
        # print(len(d1.keys()),len(d2.keys()))
        # print(len(all_words),"# of words")
        # master_dict = {}
        # for word in all_words:
        #     master_dict[word] = [[],[],[]]
        #     if word in d1:
        #         d1w = d1[word]
        #         master_dict[word][0]+=d1w[0]
        #         master_dict[word][1]+=d1w[1]
        #         master_dict[word][2] += d1w[2]
        #     if word in d2:
        #         d2w = d2[word]
        #         master_dict[word][0]+=d2w[0]
        #         master_dict[word][1]+=d2w[1]
        #         master_dict[word][2] += d2w[2]
        #
        # pkled_directory = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/" + "word_to_tensors_2000_nonnormalized.p"
        # print("pickling to: ", pkled_directory)
        #
        # pickle.dump(master_dict, open(pkled_directory, "wb"))


    elif method == "insert_topic":
        insert_topic_into_caption(encoder,decoder,"black",word_to_tensor_pkl="word_to_tensors_2000_nonnormalized.p",num_imgs_to_test=10,viz=True,randomize=True)

    elif method == "plot_word_activations":
        nouns = ["man","dog","woman","couple","frisbee"]
        verbs = ["standing","sitting","holding","riding","playing"]

        colors = ["black","white","brown","red"]
        humans = ["man","woman","boy","girl"]
        areas = ["water","snow","field","grass"]
        prepositions = ["on","of","to","with"]
        just_in = ["in"]
        #concept_dict = {"nouns":nouns,"verbs":verbs,"prepositions":prepositions}
        concept_dict = {"next":["next"],"in":["in"],"over":["over"],"under":["under"]}
        #concept_dict = {"man":["man"],"woman":["woman"],"dog":["dog"],"standing":["standing"],
        #                "sitting":["sitting"],"playing":["playing"],"in":["in"],"is":["is"],"with":["with"]}
        #concept_dict = {"standing": ["standing"],"sitting":["sitting"],"playing":["playing"],"riding":["riding"]}
        #concept_dict = {"colors":colors,"humans":humans,"areas":areas}
        plot_word_activations(concept_dict,word_to_tensor_pkl="word_to_tensors_2000_nonnormalized.p")
    elif method == "search_for_channel_topics":
        search_for_channel_topics(pkl_location="word_to_tensors_all.p")

    elif method=="save_channels_per_word":
        save_channels_per_word(encoder,decoder,img_list[1000:2000],num_images = 2000,scale_by_attention=True,pkl_location="word_to_tensors_1000_nonnormalized_pt2.p")

    elif method=="ablate_one_channel":
        ablate_one_channel(encoder,decoder,one_img,channel_num=23)

    elif method == "observe_single_channel":
        observe_single_channel(encoder,decoder,channel_num=1000)
    elif method == "test":
        d1 = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/" + "word_to_tensors_default.p"
        d2 = os.path.dirname(os.path.abspath(__file__)) + "/word_to_tensors_dicts/" + "word_to_tensors_1000.p"
        m1 = pickle.load(open(d1, "rb"))
        m2 = pickle.load(open(d2, "rb"))
        s1 = m1["<start>"]
        s2 = m2["<start>"]

        x1 = []
        x2 = []
        for i in range(10):
            x1.append(np.array(s1[1][i]))
            x2.append(np.array(s2[1][i]))

        x1 = np.array(x1)
        x2 = np.array(x2)
        n1 = np.array(s1[1][0])
        n2 = np.array(s2[1][0])