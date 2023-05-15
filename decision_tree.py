
import numpy as np
import sys
def type_convert(data):#this function turn all digit(str) in the original dataset into int
    
    int_convert=lambda x: int(x) if x.isdigit() else x
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=int_convert(data[i][j])
    return data

def read_tsv(path):#this function read the tsv file
    
    data=open(path).readlines()
    dropped_data=[]
    for i in data:
        dropped_data.append(i.strip().split('\t'))
    raw_data=type_convert(dropped_data)
    return raw_data

def output_txt(content,path):
    with open(path,'w') as f:
        for i in content:
            f.write(str(i)+'\n')

def output_matrix(error_rate_matrix,path):
    with open(path,'w') as f:
        f.write("error(train): "+str(error_rate_matrix[0])+'\n')
        f.write("error(test): "+str(error_rate_matrix[1]))

def set_divide(dataset):
    
    """
    input: all dataset[[name1,name2,labelname],[x1,x2,y1],...]
    output: attribute value, attribute name, label value
    This function divide the dataset into attribute value matrix and label value vector
    """
    y_value=[i[-1] for i in dataset]
    y_value.pop(0)
    x_value=[i[:-1] for i in dataset]
    attribute=x_value.pop(0)
    return x_value,y_value,attribute

def entropy(value):
    """
    input: a vector [y1 y2]
    output: the entropy of this vector(float)
    """
    type_key=list(set(value))
    type=[]
    for i in type_key:
        type.append(value.count(i))

    entropy=0
    for i in range(len(type)):
        entropy+=(type[i]/len(value))*np.log2(type[i]/len(value))
    entropy=-entropy
    return entropy

def max_mutual_inf(x_value,y_value,attribute):
    
    """
    input:attribute value(matrix[[row1][row2]]), attribute name[name1 name2], label value[y1 y2]
    output:max mutual information(int),the name of attribute that provide such a mutual information
    """

    gain_vector=[]
    for i in range(len(x_value[0])):
        x_left=0; x_right=0; y_left=[]; y_right=[]
        for j in range(len(x_value)):
            
            if x_value[j][i]==1:
                x_left+=1
                y_left.append(y_value[j])
                
            
            else:
                x_right+=1
                y_right.append(y_value[j])

                   
        Gain=entropy(y_value)-(((x_left/len(x_value))*entropy(y_left))+((x_right/len(x_value))*entropy(y_right)))

        gain_vector.append(Gain)

    mmi=max(gain_vector)
    name=attribute[gain_vector.index(mmi)]
    return mmi,name

class MajorityVoteClassifier:
    def __init__(self):
        self.prevalue=None

    def training(self,y_value):

        if len(y_value)==0:
            self.prevalue=1

        else:

            labelValueSet=list(set(y_value)) #'set' object is not subscriptable so tranfer as a list
            value_count=[y_value.count(i) for i in labelValueSet]
            self.prevalue=labelValueSet[value_count.index(max(value_count))]
            if value_count.count(value_count[0])==len(value_count):
                self.prevalue=1
            
            if y_value.count(y_value[0])==len(y_value):
                self.prevalue=y_value[0]
            
                        
    def predict(self,y_value):
        #if the first line is column name
        dim=len(y_value) 
        pre_y_value=[self.prevalue for i in range(dim)]
        return pre_y_value

    def estimate_error(self,true_y_value,pre_y_value):
        error_rate=0
        for i in range(len(true_y_value)):
            if true_y_value[i]!=pre_y_value[i]:
                error_rate+=1
        error_rate=error_rate/len(true_y_value)
        return error_rate

class Node:
    def __init__(self):
        self.type=None
        self.attr=None
        self.vote=None
        self.left=None
        self.right=None

def training_tree(x_value,y_value,attribute,max_depth,depth=0):
    node=Node()
    MVC=MajorityVoteClassifier()
    
    #basecase1: if the dataset is empty
    if len(y_value)==0 or len(x_value)==0 or len(x_value[0])==0 :

        node.type='leaf'
        
    #basecase2: if all rows of attribute values are the same
    count_of_row_equal=0

    for row in x_value:
        if len(row)>1:
            if row.count(row[0])==len(row):
                count_of_row_equal+=1
            else:
                break
        else:
            break

    if count_of_row_equal==len(x_value):
        node.type='leaf'
        
    #basecase3: if all label values are the same
    
    if y_value.count(y_value[0])==len(y_value):
        node.type='leaf'
        
    if node.type=='leaf':
        MVC.training(y_value)
        node.vote=MVC.prevalue
        return node

    #determine node
    mmi,node.attr=max_mutual_inf(x_value,y_value,attribute)

    #stop condition: mmi should be greater than 0
    if mmi <= 0:
        node.type='leaf'
        
    if node.type=='leaf':
        MVC.training(y_value)
        node.vote=MVC.prevalue
        return node

    else:
        node.type='internal'
        depth+=1

    #stop condition: depth is deeper than the max depth
    if depth > max_depth:
        node.type='leaf'
        

    else:
    #get the column number of the node attribute
        column_number=attribute.index(node.attr)
        
        left_branch_y=[y_value[i] for i in range(len(y_value)) if x_value[i][column_number]==1]
        right_branch_y=[y_value[i] for i in range(len(y_value)) if x_value[i][column_number]==0]
        
        #branch here should form new dataset
        #left branch,value=1
        left_branch=[i for i in x_value if i[column_number]==1]

        #branch here should form new dataset
        #right branch,value=0
        right_branch=[i for i in x_value if i[column_number]==0]

        #get rid of the attribute we used in the parent node
        new_attribute=attribute[:column_number]+attribute[column_number+1:]



        left_branch_x=[[i[j] for j in range(len(i)) if j!=column_number] for i in left_branch]
        right_branch_x=[[i[j] for j in range(len(i)) if j!=column_number] for i in right_branch]
        
        
        
        #recursion
        node.left=training_tree(left_branch_x,left_branch_y,new_attribute,max_depth,depth)
        node.right=training_tree(right_branch_x,right_branch_y,new_attribute,max_depth,depth)
        
    if node.type=='leaf':
        MVC.training(y_value)
        node.vote=MVC.prevalue

    return node

def one_row_pred(row,attribute,node):
#row: a single row of x matrix
#attribute: a vector of attributes names
#out put: a single predict value of one row
    if node.type=='leaf':
        return node.vote
        
    else:
        attr_loc=attribute.index(node.attr)

        if row[attr_loc]==1:
            return one_row_pred(row,attribute,node.left)

        else:
            return one_row_pred(row,attribute,node.right)

def predict_tree(x_value,attribute,node):

    #this function will predict the label
    #return is node.vote for each row, we need to creat a
    
    predict_value=[]
    for i in x_value:
        predict_value.append(one_row_pred(i,attribute,node))
    return predict_value

if __name__=='__main__':
    train_input=sys.argv[1]
    test_input=sys.argv[2]
    max_depth=int(sys.argv[3])
    train_output=sys.argv[4]
    test_output=sys.argv[5]
    metrics_output=sys.argv[6]
    
    MVC=MajorityVoteClassifier()
    dataset=read_tsv(train_input)
    x_value,y_value,attribute=set_divide(dataset)
    tree=training_tree(x_value,y_value,attribute,max_depth)
    train_pre_result=predict_tree(x_value,attribute,tree)
    
    dataset_predict=read_tsv(test_input)
    x_pre,y_pre,attribute_pre=set_divide(dataset_predict)
    pre_result=predict_tree(x_pre,attribute_pre,tree)

    err_tain=MVC.estimate_error(y_value,train_pre_result)
    err_test=MVC.estimate_error(y_pre,pre_result)

    err_metrics=[err_tain,err_test]
    output_txt(train_pre_result,train_output)
    output_txt(pre_result,test_output)
    output_matrix(err_metrics,metrics_output)

