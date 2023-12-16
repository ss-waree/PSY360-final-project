import numpy as np
import torch


def sample_marbles(seed, num_bags, num_marbles, alpha, beta_0, beta_1):

    torch.manual_seed(seed)
    lambda_param = 1

    # we need alpha to be non-zero positive so we can sample from theta
    # num_colors = 2
    # alpha = 0.0
    # while alpha == 0: 
    #     alpha = torch.distributions.Exponential(lambda_param).sample()

    # beta = torch.distributions.Dirichlet(torch.ones(num_colors)).sample()

    # # theta_arr = np.zeros((num_bags, num_colors))

    # alpha = 10
    beta = torch.tensor([beta_0, beta_1])
    # training and testing need to be generated from the same theta
    marbles = np.zeros((num_bags,num_marbles))
    #theta = torch.distributions.Dirichlet(abs(alpha) * beta).sample().numpy()
    theta = torch.distributions.Dirichlet(alpha * beta).sample().numpy()
    for i in range(num_bags):
        num0 = round(num_marbles * theta[0])
        marbles[i,:num0] = 0
        marbles[i,num0:] = 1
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(num_marbles).numpy()
        # Shuffle the marbles using these indices
        marbles[i] = marbles[i][shuffled_indices]

    return marbles
    
    # all black or white marbles in a train-test set
    # rand_choice = np.random.randint(2)
    # if rand_choice == 0:
    #     return np.zeros((num_bags,num_marbles))
    # else:
    #     return np.ones((num_bags,num_marbles))

# Display the sampled values with 2 decimal points
# print(f"Sampled alpha: \n{alpha.item():.2f}")
# print(f"Sampled beta: \n{np.array2string(beta.numpy(), formatter={'float_kind':lambda x: f'{x:.2f}'})}")
# print(f"Sampled theta: \n{np.array2string(theta_arr, formatter={'float_kind':lambda x: f'{x:.2f}'})}")
# print(f"Generated marbles: \n{marbles}")

