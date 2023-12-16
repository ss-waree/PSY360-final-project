
import jsonlines
import logging
import torch
import csv

import copy

from dataset_iterators import *
from dataloading import *
from training import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


################################################################################################
# Category evaluations
################################################################################################

def marble(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None, num_black=10, num_white=10):
    
    # set up support set
    # set up training batch according to given number of marbles

    n_train = num_black + num_white
    
    # "1" being token that signifies marble
    train_inputs = [[1] for i in range(n_train)]

    # denotes marbles being black/white 
    train_labels = np.zeros(n_train)
    train_labels[num_black:] = 1

    # shuffle with seeds later
    random.shuffle(train_labels)

    # print(train_labels)

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels).unsqueeze(1),  
    "train_batch_size" : 1}

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    # training batch here = 1 bag of marbles
    temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # evaluate with "marble" token
    batch = {"input_ids" : torch.FloatTensor([[1]])}
    outp = temp_model(batch)["probs"].detach().numpy()

    return outp

def marble_n_runs(model, model_name, model_alpha, model_beta_0, model_beta_1, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None, num_black=10, num_white=10):
    probs_by_index = np.zeros((n_runs, 2))
    
    for i in range(n_runs):
        outputs = marble(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, max_n_features=max_n_features, num_black = num_black, num_white = num_white)
        probs_by_index[i] = outputs

    n_train = num_black + num_white
    true_theta = np.array([num_black/n_train, num_white/n_train])
    pred_theta = np.sum(probs_by_index, axis = 0)/n_runs
    err = np.abs(true_theta[0]-pred_theta[0]) 

    logging.info("True theta = " + str([true_theta]))
    logging.info("Average predicted theta = " + str(pred_theta))

    # Data to be written to the CSV file
    # data = [model_name, model_alpha, model_beta_0, model_beta_1, num_black, num_white, n_train, true_theta[0], true_theta[1], pred_theta[0], pred_theta[1], err, 1-err] 
    data = [model_name, num_black, num_white, n_train, true_theta[0], true_theta[1], pred_theta[0], pred_theta[1], err, 1-err] 

    # Open a file in write mode
    with open('results_c.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def table3(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None):

    train_pairs = [([0,0,0,1], 1), ([0,1,0,1], 1), ([0,1,0,0], 1), ([0,0,1,0], 1), ([1,0,0,0], 1), ([0,0,1,1], 0), ([1,0,0,1], 0), ([1,1,1,0], 0), ([1,1,1,1], 0)]
    random.shuffle(train_pairs)

    train_inputs = [x[0] for x in train_pairs]
    train_labels = [x[1] for x in train_pairs]

    train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1)}

    if train_batch_size is None:
        train_batch_size = len(train_inputs)

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    # training batch here = 1 bag of marbles
    temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # could try fixed ratios of 20 black 80 white etc 
    # Object, feature values, human, rr_dnf
    possible_inputs = [
            ("A1", [0,0,0,1], 0.77, 0.82),
            ("A2", [0,1,0,1], 0.78, 0.81),
            ("A3", [0,1,0,0], 0.83, 0.92),
            ("A4", [0,0,1,0], 0.64, 0.61),
            ("A5", [1,0,0,0], 0.61, 0.61),
            ("B1", [0,0,1,1], 0.39, 0.47),
            ("B2", [1,0,0,1], 0.41, 0.47),
            ("B3", [1,1,1,0], 0.21, 0.21),
            ("B4", [1,1,1,1], 0.15, 0.07),
            ("T1", [0,1,1,0], 0.56, 0.57),
            ("T2", [0,1,1,1], 0.41, 0.44),
            ("T3", [0,0,0,0], 0.82, 0.95),
            ("T4", [1,1,0,1], 0.40, 0.44),
            ("T5", [1,0,1,0], 0.32, 0.28),
            ("T6", [1,1,0,0], 0.53, 0.57),
            ("T7", [1,0,1,1], 0.20, 0.13)
            ]

    outputs = []
    for inp in possible_inputs:
        batch = {"input_ids" : torch.FloatTensor([expand_features(inp[1], max_length=max_n_features)])}
        outp = temp_model(batch)["probs"].item()
        outputs.append(outp)
        #print("\t".join([str(x) for x in inp]) + "\t" + str(outp))

    return outputs

def table3_n_runs(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None):

    probs_by_index = [[] for _ in range(16)]
    for _ in range(n_runs):
        outputs = table3(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, max_n_features=max_n_features)
        for index, output in enumerate(outputs):
            probs_by_index[index].append(output)
    
    # Object, feature values, human, rr_dnf
    possible_inputs = [
            ("A1", [0,0,0,1], 0.77, 0.82),
            ("A2", [0,1,0,1], 0.78, 0.81),
            ("A3", [0,1,0,0], 0.83, 0.92),
            ("A4", [0,0,1,0], 0.64, 0.61),
            ("A5", [1,0,0,0], 0.61, 0.61),
            ("B1", [0,0,1,1], 0.39, 0.47),
            ("B2", [1,0,0,1], 0.41, 0.47),
            ("B3", [1,1,1,0], 0.21, 0.21),
            ("B4", [1,1,1,1], 0.15, 0.07),
            ("T1", [0,1,1,0], 0.56, 0.57),
            ("T2", [0,1,1,1], 0.41, 0.44),
            ("T3", [0,0,0,0], 0.82, 0.95),
            ("T4", [1,1,0,1], 0.40, 0.44),
            ("T5", [1,0,1,0], 0.32, 0.28),
            ("T6", [1,1,0,0], 0.53, 0.57),
            ("T7", [1,0,1,1], 0.20, 0.13)
            ]

    human_probs = []
    rr_dnf_probs = []
    net_probs = []
    logging.info("Input name\tInput feature\tHumans\tRR_DNF\tMeta neural net")
    for index, inp in enumerate(possible_inputs):
        human_probs.append(inp[2])
        rr_dnf_probs.append(inp[3])
        net_probs.append(sum(probs_by_index[index])/n_runs)
        logging.info("\t".join([str(x) for x in inp]) + "\t" + str(sum(probs_by_index[index])/n_runs))

    corr_human_net = np.corrcoef(human_probs, net_probs)[0, 1]
    corr_rrdnf_net = np.corrcoef(rr_dnf_probs, net_probs)[0, 1]
    corr_human_rrdnf = np.corrcoef(human_probs, rr_dnf_probs)[0, 1]

    logging.info("human net correlation " + str(corr_human_net))
    logging.info("rr_dnf net correlation " + str(corr_rrdnf_net))
    logging.info("human rr_dnf corr " + str(corr_human_rrdnf))



