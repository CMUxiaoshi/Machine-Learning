import sys

class MajorityVoteClassifier:
    def __init__(self):
        self.mapping=None

    def training(self,attribute,label):
        type_key=list(set(label))
        y_type={}
        for i in type_key:
            y_type[i]=0
        for i in type_key:
            for j in label:
                if i == j:
                    y_type[i]+=1
        self.mapping=max(y_type,key=y_type.get)
        
        for i in type_key:
            if y_type[i] == y_type[self.mapping] and self.mapping!= i:
                self.mapping=1
        
    def predict(self,attribute):
        #if the first line is column name
        dim=len(attribute)-1 
        pre_label=[self.mapping for i in range(dim)]
        return pre_label

    def estimate_error(self,true_label,pre_label):
        error_rate=0
        for i in range(len(true_label)):
            if true_label[i]!=pre_label[i]:
                error_rate+=1
        error_rate=error_rate/len(true_label)
        return error_rate

def type_convert(data):
    int_convert=lambda x: int(x) if x.isdigit() else x
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=int_convert(data[i][j])
    return data

def read_tsv(path):
    data=open(path).readlines()
    dropped_data=[]
    for i in data:
        dropped_data.append(i.strip().split('\t'))
    raw_data=type_convert(dropped_data)
    return raw_data

def get_label(data):
    label=[i[-1] for i in data]
    del(label[0])
    return label

def output_txt(content,path):
    with open(path,'w') as f:
        for i in content:
            f.write(str(i)+'\n')

def output_matrix(error_rate_matrix,path):
    with open(path,'w') as f:
        f.write("error(train): "+str(error_rate_matrix[0])+'\n')
        f.write("error(test): "+str(error_rate_matrix[1]))

if __name__ == '__main__':
    train_input=sys.argv[1]
    test_input=sys.argv[2]
    train_output=sys.argv[3]
    test_output=sys.argv[4]
    matrics_output=sys.argv[5]

    train_input=read_tsv(train_input)
    test_input=read_tsv(test_input)
    
    MVC=MajorityVoteClassifier()
    classifier=MVC.training(None,get_label(train_input))
    predict_train=MVC.predict(train_input)
    predict_test=MVC.predict(test_input)
    error_train=MVC.estimate_error(get_label(train_input),predict_train)
    error_test=MVC.estimate_error(get_label(test_input),predict_test)
    error_matrix=[error_train, error_test]

    output_txt(predict_train,train_output)
    output_txt(predict_test,test_output)
    output_matrix(error_matrix,matrics_output)
