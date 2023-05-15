import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(x):
    """
    Parameters: x: a vector. If the validation data is a matrix, then we should implement a loop in the main function. 
    Output: A value that represents log(sum(exp(xi))) for each xi in x.
    """
    m=np.max(x)
    new_x=x-m
    sum_exp=np.log(np.sum(np.exp(new_x)))+m
    return sum_exp

def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)


    # Initialize log_alpha and fill it in - feel free to use words_to_indices to index the specific word
    log_alpha=np.zeros((L,M))

    for t in range(L):
        for i in range(M):
            if t==0:
                log_alpha[t,i]=loginit[i]+logemit[i,words_to_indices[seq[t]]]
            else:
                log_alpha[t,i]=logsumexp(log_alpha[t-1,:]+logtrans[:,i])+logemit[i,words_to_indices[seq[t]]]

    # Initialize log_beta and fill it in - feel free to use words_to_indices to index the specific word
    log_beta=np.zeros((L,M))
    for t in range(L-1,-1,-1):
        for i in range(M):
            if t==L-1:
                log_beta[t,i]=0
            else:
                log_beta[t,i]=logsumexp(log_beta[t+1,:]+logtrans[i,:]+logemit[:,words_to_indices[seq[t+1]]])
    # Compute the predicted tags for the sequence - tags_to_indices can be used to index to the rwquired tag
    path=list()
    tags=list()
    for t in range(L):
        path.append(np.argmax(log_alpha[t]+log_beta[t]))
        for key,value in tags_to_indices.items():
            if value==path[t]:
                tags.append(key)

    
    
    # Compute the stable log-probability of the sequence
    log_prob=logsumexp(log_alpha[-1])
    # Return the predicted tags and the log-probability
    return tags, log_prob
    pass
    
def get_seq(validation_data):
    seq=[]
    for i in range(len(validation_data)):
        seq.append(validation_data[i][0])
    return seq
    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    loginit=np.log(hmminit)
    logemit=np.log(hmmemit)
    logtrans=np.log(hmmtrans)

    tags_list=[]
    log_prob_list=[]
    for j in range(len(validation_data)):
        seq=get_seq(validation_data[j])
        tags,log_prob=forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices)
        tags_list.append(tags)
        log_prob_list.append(log_prob)

    Accuracy=0
    words=0
    for m in range(len(validation_data)):
        val_data=validation_data[m]
        for n in range(len(val_data)):
            if val_data[n][1]==tags_list[m][n]:
                Accuracy+=1
            words+=1
    Accuracy=Accuracy/(words)
    avg_log_prob=np.mean(log_prob_list)
    # Write the predicted tags to the output file, (word, tag) per line, separated by a space. 
    # Each sequence should be separated by a newline.
    with open(predicted_file, "w") as f:
        for i in range(len(tags_list)):
            for j in range(len(tags_list[i])):
                f.write(validation_data[i][j][0]+"\t"+tags_list[i][j]+"\n")
            f.write("\n")
    # Write the average log-likelihood and accuracy to the metric file, each on a separate line.
    # Name: Value format
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: "+str(avg_log_prob)+"\n")
        f.write("Accuracy: "+str(Accuracy))
    pass