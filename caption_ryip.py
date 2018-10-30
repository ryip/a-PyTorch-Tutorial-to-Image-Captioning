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

def gather_word_data(encoder, decoder, img_list, word_map, beam_size):
    if not isinstance(img_list,list):
        img_list = [img_list]
    sentences = []
    rev_word_map = {v: k for k, v in word_map.items()}
    for img in img_list:
        try:
            seq, alphas = caption_image_beam_search(encoder, decoder, img, word_map, beam_size)
        # alphas = torch.FloatTensor(alphas)
        # Visualize caption and attention of best sequence
            sentences.append([rev_word_map[i] for i in seq])
        except ValueError:
            print("no caption")
            sentences.append([])
    return sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

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

    iter_over_list = True

    # Load word map (word2ix)
    print("!!!")

    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    #print(len(word_map))
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word


    # Choose whether to iterate over all the captions
    if iter_over_list:
        img_list = glob.glob('C:/Users/Richard/Documents/MIT stuff/2018 Fall/research/a-PyTorch-Tutorial-to-Image-Captioning/img/flickr8k_dataset/*.jpg')
        one_img = img_list[201]

        #specify method by which we search through captions
        method = "insert_topic"
        if method == "insert_topic":
            word_to_tensor_pkl = "word_to_tensors_4000.p"
            nd = pickle.load(open(word_to_tensor_pkl, "rb"))

            x = sorted(nd.keys(), key=lambda x: len(nd[x][0]), reverse=True)
            # Print out top 10 words.
            for i in range(30): print(x[i], len(nd[x[i]][0]))


            target_word = "man"
            avg_channels = nd[target_word][1]
            avg_all = nd["<start>"][1]

            images_not_containing_target = [im for im in nd["<start>"][0] if im not in nd[target_word][0]]
            #print(images_not_containing_target[0])

            #print(avg_channels)
            #num_avg_channels = [b.item() for b in [c for c in avg_channels]]
            #print(num_avg_channels)

            #avg_tgt_word_output = [sum(x)/len(x) for x in zip(*[np.array(c) for c in avg_channels])]
            avg_tgt_word_output = np.mean([np.array(c) for c in avg_channels],axis=0)
            avg_all_output = np.mean([np.array(c) for c in avg_all],axis=0)
            #print(len(avg_tgt_word_output))

            #Set to 0 to generate original image
            num_channels = 400
            top_n_channels = sorted(range(2048),key = lambda x: abs(avg_tgt_word_output[x]-avg_all_output[x]),reverse=True)[:num_channels]
            #print(top_n_channels)

            num_imgs_to_test = 100
            original_and_modified = []
            for i in range(num_imgs_to_test):

                current_img = images_not_containing_target[i]
                word_mat = gather_word_data(encoder, decoder, current_img, word_map, args.beam_size)
                original_and_modified.append(word_mat)
                #NETDISSECT STARTS HERE
            ne = encoder
            dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')])
            ablation = torch.ones(2048)
            for channel in top_n_channels:
                ablation[channel] = 0
            encoder.ablation['output_layer'] = ablation.to(device).type(torch.cuda.FloatTensor)

            dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')], adding=True)
            replacement = torch.zeros(2048)
            for channel in top_n_channels:
                replacement[channel] = avg_tgt_word_output[channel].item()
            encoder.ablation['output_layer'] = replacement.to(device).type(torch.cuda.FloatTensor)

            # dissection.replace_layers(encoder, [('resnet.7.2', 'output_layer')])
            # rep = torch.tensor(avg_tgt_word_output)
            # encoder.replacement['output_layer'] = rep.to(device).type(torch.cuda.FloatTensor)
            for i in range(num_imgs_to_test):
                current_img = images_not_containing_target[i]
                viz = False
                if viz:

                    seq, alphas = caption_image_beam_search(encoder, decoder, current_img, word_map, args.beam_size)
                    alphas = torch.FloatTensor(alphas)
                    print("==ORIGINAL==")
                    print(original_and_modified[i])
                    visualize_att(current_img, seq, alphas, rev_word_map, args.smooth)
                word_mat = gather_word_data(encoder, decoder, current_img, word_map, args.beam_size)
                original_and_modified[i].append(word_mat[0])
            # for img in original_and_modified:
            #     print(img[0])
            #     print(img[1])
            #     print("==")
            # for x in original_and_modified: print(x[1])

            num_inserted = sum([1 for x in original_and_modified if target_word in x[1]])
            print("target word: ",target_word, "num channels: ",num_channels)
            print("num inserted: ",num_inserted,"num_tested: ",num_imgs_to_test,"ratio: ",num_inserted/num_imgs_to_test)


        elif method == "search_for_channel_topics":
            word_to_tensor_pkl = "word_to_tensors_100.p"

            #Dictionary mapping words to [[img],[avg_channels]] for each img that generates the word
            nd = pickle.load(open(word_to_tensor_pkl, "rb"))
            x = sorted(nd.keys(),key = lambda x: len(nd[x][0]),reverse=True)

            #Print out top 10 words.
            for i in range(10):
                print(x[i],len(nd[x[i]][0]))
            bestword = nd[x[0]][1]
            bestword2 = nd[x[9]][1]
            aa = []

            bb = []
            for i in range(len(bestword[0])):
                #print(i,len(bestword[0]))
                aa.append(sum([wd[i] for wd in bestword[:100]])/len(bestword))
                bb.append(sum([wd[i] for wd in bestword2[:100]])/len(bestword2))
            #print(aa)

            s = [nm.item() for nm in aa]
            q = [nm.item() for nm in bb]
            #plt.plot(s)
            #plt.plot(q)
            plt.plot([s[i]-q[i] for i in range(len(s))])
            #plt.show()

        elif method=="search_for_channel_topics":
            num_imgs = 100
            truncated_img_list = img_list[:num_imgs]
            dissection.retain_layers(encoder, [('resnet.7.2', 'output_layer')])
            word_to_tensors = {} # word -> [list of images, list of avg_channels]
            # Note that <start> and <end> will have all imgs and all avg channels


            #timing
            t = time.time()

            for img in truncated_img_list:
                #dissection.retain_layers(encoder, [('resnet.7.2', 'output_layer')])
                word_mat = gather_word_data(encoder, decoder, img, word_map, args.beam_size)
                encoder_output = encoder.retained['output_layer'][0]
                #for word in word_mat:
                #    print(word)
                #print(encoder_output,encoder_output.size())
                avg_channels =encoder_output.mean(dim = 1).mean(dim=1)
                #print(avg_channels,avg_channels.size())
                for word in set(word_mat[0]):
                    if word not in word_to_tensors:
                        word_to_tensors[word] = [[img],[avg_channels]]
                    else:
                        word_to_tensors[word][0].append(img)
                        word_to_tensors[word][1].append(avg_channels)
            word_freqs = sorted([[x,len(word_to_tensors[x][0])] for x in word_to_tensors],key=lambda x: x[1],reverse=True)
            toc = time.time()-t
            print("images: ",num_imgs," time: ",toc, " time per img: ",toc/num_imgs)
            #for word in word_freqs:
            #    print(word)
            print("pickling")
            pickle.dump(word_to_tensors,open("word_to_tensors_100.p","wb"))
            #print("unpickling")
            #nd = pickle.load(open("word_to_tensors.p","rb"))
            #print(nd)

        elif method=="ablate_one_channel":
            channel = 106 #23 = bathrooms
            dissection.retain_layers(encoder, [('resnet.7.2', 'output_layer')])
            dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')])
            #dissection.replace_layers(encoder, [('resnet.7.2', 'output_layer')])
            #replacement = torch.zeros(2048)
            #replacement[channel] = 100
            ablation = torch.ones(2048)
            ablation[channel] = 0
            #print(ablation)
            encoder.ablation['output_layer'] = ablation.to(device).type(torch.cuda.FloatTensor)

            dissection.ablate_layers(encoder, [('resnet.7.2', 'output_layer')], adding=True)
            addition = torch.zeros(2048)
            addition[channel] = 50
            encoder.ablation['output_layer'] = addition.to(device).type(torch.cuda.FloatTensor)

            viz = True
            if viz:
                seq, alphas = caption_image_beam_search(encoder, decoder, one_img, word_map, args.beam_size)
                alphas = torch.FloatTensor(alphas)

                visualize_att(one_img, seq, alphas, rev_word_map, args.smooth)
            word_mat = gather_word_data(encoder, decoder, one_img, word_map, args.beam_size)
            encoder_output = encoder.retained['output_layer'][0]
            print(encoder_output[channel])


            for word in word_mat:
                print(word)



        elif method == "observe_single_channel":
            dissection.replace_layers(encoder, [('resnet.7.2', 'output_layer'), ])
            replacement = torch.zeros(2048)
            oneval = [0,1,10,100,1000,10000]
            for i in range(len(oneval)):
                replacement[1002] = oneval[i]
                encoder.replacement['output_layer'] = replacement.to(device).type(torch.cuda.FloatTensor)
                word_mat = gather_word_data(encoder, decoder, img_list[0], word_map, args.beam_size)
                #seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
                #alphas = torch.FloatTensor(alphas)
                #visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
                for word in word_mat:
                    st = ""
                    for w in word:
                        st +=w + " "

                    print(st)

    else:
        # Encode, decode with attention and beam search

        seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)

        visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
