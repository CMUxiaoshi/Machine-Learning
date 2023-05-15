
import numpy as np
import sys

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset

def read_tsv(path):#this function read the tsv file
    
    data=open(path).readlines()
    dropped_data=[]
    for i in data:
        dropped_data.append(i.strip().split('\t'))
    return dropped_data
    
def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file) as f:
        read_file = read_tsv(file)
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def word_split(data):
    #this function split the data into words:
    #input:a vector contain many sentences, each sentence is in a single list. e.g. [[sentence1],[sentence2],[sentence3]]
    #output: a list contain many lists, each lists inside the big list contain the words in the sentence. e.g. [['word1','word2']]
    divid_sentence=[]
    for word in data:
        divid_sentence.append((word[0],word[1].split(' ')))
    return divid_sentence

def sent_value_cal(divid_sentence,dictionary):
    """
    Match the words in each sentence to the words in the dictionary. Calculate the value for each words and sum them up to get the sentence score.

    Parameters: 
    divid_sentence (list): a list contain many tuples, each tuple is (label,[word1,word2])
    dictionary(dictionary): a dictionary indexed by words, returning the corresponding glove
    
    Returns:
    sentence_value (list): a list contain tuples, each tuples are (word,value),value is an nparray

    """
    sentence_value=[]
    for sentence in divid_sentence:
        value=np.zeros(300)
        count_words=0
        for word in sentence[1]:
            if word in dictionary:
                value+=dictionary.get(word)
                count_words+=1
        value=value/count_words
        sentence_value.append((sentence[0],value))
    return sentence_value
    
def write_to_file(file_name, data):
    #write to tsv file
    with open(file_name, 'w') as f:
        for label,value in data:
            label=str(format(float(label), '.6f'))
            #keep 6 decimal places
            review_value=''
            for i in range(len(value)):
                review_value=review_value+'\t'+str(format(float(value[i]), '.6f'))
            result=label+review_value+'\n'
            f.write(result)

if __name__=='__main__':
    train_input=sys.argv[1]
    validation_input=sys.argv[2]
    test_input=sys.argv[3]
    feature_dictionary_input=sys.argv[4]
    formatted_train_out=sys.argv[5]
    formatted_validation_out=sys.argv[6]
    formatted_test_out=sys.argv[7]

    train_data=load_tsv_dataset(train_input)
    validation_data=load_tsv_dataset(validation_input)
    test_data=load_tsv_dataset(test_input)

    glove_map=load_feature_dictionary(feature_dictionary_input)
    
    divided_train_data=word_split(train_data)
    divided_validation_data=word_split(validation_data)
    divited_test_data=word_split(test_data)

    train_value=sent_value_cal(divided_train_data,glove_map)
    validation_value=sent_value_cal(divided_validation_data,glove_map)
    test_value=sent_value_cal(divited_test_data,glove_map)

    write_to_file(formatted_train_out,train_value)
    write_to_file(formatted_validation_out,validation_value)
    write_to_file(formatted_test_out,test_value)
    
    

    