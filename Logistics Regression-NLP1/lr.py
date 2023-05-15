
import numpy as np
import sys

def gradients_descent(x_array,y_array,num_epochs,init_theta,learning_rate):
    """
    SGD of negative log-likelihood function
    
    parameters:
    x_array: array of features, a metrix of shape (num_samples, num_features)
    y_array: array of labels, a vector of shape (num_samples)
    num_epochs: number of epochs, an integer, the threhold for the number of epochs
    init_theta: initial theta, a vector of shape (num_features)
    learning_rate: a float, the threhold for the learning rate

    output: 
    trained theta, a vector of shape (num_features)

    """
    epoch=0
    theta=init_theta

    while epoch<num_epochs:

        for i in range(x_array.shape[0]):
            p=1/(1+np.exp(-(theta@x_array[i])))
            gradient=-(y_array[i]-p)*x_array[i]
            theta=theta-learning_rate*gradient
        
        epoch=epoch+1
          
    return theta

def predict(theta,x_array):
    """
    Predict with theta we trained

    Parameters:
    theta: a array with shape (num_features), it is the trained theta
    x_array: array of features, a metrix of shape (num_samples, num_features)

    output:
    prediction label, a list of shape (num_samples)

    """
    y_predict=[]

    for i in range(x_array.shape[0]):
        p=1/(1+np.exp(-(theta@x_array[i])))
        if p>0.5:
            y_predict.append(1)
        else:
            y_predict.append(0)

    return y_predict

def error_rate(y_true,y_predict):
    """
    Calculate the error rate
    
    Parameter: 
    y_true: array of true labels, a vector of shape (num_samples)
    y_predict: array of predicted labels, a vector of shape (num_samples)

    output:
    error rate, a float

    """

    count=0
    rate=0
    for i in range(len(y_true)):
        if y_true[i]!=y_predict[i]:
            count+=1
    rate=count/len(y_true)
    return rate

def array_sp(data):
    """
    split the dataset into label and features,  add intercept into the x array

    Parameter:
    data: array of whole data, shape (num_samples, num_features+1)

    output:
    x_array: array of features(including intercept), a metrix of shape (num_samples, num_features+1)
    y_array: array of labels, a vector of shape (num_samples)
    

    """
    y_array=data[:,0]
    x_array=data[:,1:]
    intercept_array=np.ones(len(y_array))
    x_array=np.c_[intercept_array.T,x_array]
    
    return x_array,y_array

def output_matrix(error_rate_matrix,path):
    with open(path,'w') as f:
        f.write("error(train): "+str(format(float(error_rate_matrix[0]),'.6f'))+'\n')
        f.write("error(test): "+str(format(float(error_rate_matrix[1]),'.6f')))

def output_txt(content,path):
    with open(path,'w') as f:
        for i in content:
            f.write(str(i)+'\n')

if __name__=="__main__":
    formatted_train_input=sys.argv[1]
    formatted_validation_input=sys.argv[2]
    formatted_test_input=sys.argv[3]
    trained_out=sys.argv[4]
    test_out=sys.argv[5]
    metrics_out=sys.argv[6]
    num_epoch=int(sys.argv[7])
    learning_rate=float(sys.argv[8])

    train_data=np.loadtxt(formatted_train_input,delimiter="\t")
    validation_data=np.loadtxt(formatted_validation_input,delimiter="\t")
    test_data=np.loadtxt(formatted_test_input,delimiter="\t")

    divided_train_data_x,divided_train_data_y=array_sp(train_data)
    divided_validation_data_x,divided_validation_data_y=array_sp(validation_data)
    divided_test_data_x,divided_test_data_y=array_sp(test_data)
    
    init_theta=np.zeros(divided_train_data_x.shape[1])
    trained_theta=gradients_descent(divided_train_data_x,divided_train_data_y,num_epoch,init_theta,learning_rate)

    train_pre=predict(trained_theta,divided_train_data_x)
    vali_pre=predict(trained_theta,divided_validation_data_x)
    test_pre=predict(trained_theta,divided_test_data_x)

    error_rate_train=error_rate(divided_train_data_y,train_pre)
    error_rate_vali=error_rate(divided_validation_data_y,vali_pre)
    error_rate_test=error_rate(divided_test_data_y,test_pre)

    error_metrix=[error_rate_train,error_rate_test]

    output_matrix(error_metrix,metrics_out)
    output_txt(train_pre,trained_out)
    output_txt(test_pre,test_out)

