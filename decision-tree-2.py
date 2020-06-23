#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pprint import pprint
import scipy.stats as sps


# In[2]:


#processing the mushroom dataset - works on mushroom dataset
dataset = pd.read_csv('data/mushrooms.csv',header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['target','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
             'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
             'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population',
             'habitat']
print(dataset.head()) 


# In[3]:


#processing the tic-tac-toe dataset

'''dataset = pd.read_csv('data/tic-tac-toe.csv',header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['target','top-left-square','top-middle-square','top-right-square','middle-left-square','middle-middle-square','middle-right-square',
                   'bottom-left-square','bottom-middle-square','bottom-right-square','Class']
print(dataset.head()) '''


# In[4]:


#calculates entropy by taking in the target column
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    #print ("entropy:",entropy)
    return entropy


# In[5]:


def InfoGain(data,split_attribute_name,target_name="target"):
    
    #entropy of whole dataset
    total_entropy = entropy(data[target_name])
    
    #vals and counts for split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain


# In[6]:


def ID3(data,originaldata,features,target_attribute_name="target",parent_node_class = None):
    
    #if all target values have same value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #if dataset empty
    elif len(data)==0:
        #return the mode target feature value in the original dataset
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    #if the feature space is empty
    elif len(features) ==0:
        return parent_node_class #most common target feature value of parent node
    
    #build the tree 
    else:
        #default val of node = mode target feature val of current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        features = np.random.choice(features,size=np.int(np.sqrt(len(features))),replace=False)
        
        #find best feature to split the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #initial creation of tree structure
        #root gets the name of the feature with the maximum info gain
        tree = {best_feature:{}}
        
        #remove feature with best info gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #growing branches
        for value in np.unique(data[best_feature]):
            value = value
            #split dataset along feature with largest info gain 
            sub_data = data.where(data[best_feature] == value).dropna()
            #call the ID3 algorithm for each of those sub_datasets with the new parameters
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)     


# In[7]:


def predict(query,tree,default = 'p'):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result


# In[8]:


def train_test_split(dataset):
    training_data = dataset.iloc[:round(0.75*len(dataset))].reset_index(drop=True)
    testing_data = dataset.iloc[round(0.75*len(dataset)):].reset_index(drop=True)
    return training_data,testing_data


# In[9]:


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1] 
#print(training_data)


# In[10]:


#create the ID3 tree to demo on 

id3_tree = (training_data,training_data,training_data.drop(labels=['target'],axis=1).columns)
#print(id3_tree)


# In[11]:


query = testing_data.iloc[0,:].drop('target').to_dict()
query_target = testing_data.iloc[0,0]
print('target: ',query_target)
prediction = 'p' #default prediction
print('prediction for id3 tree: ',prediction)


# In[12]:


#test model on id3
def ID3_Test(data,id3_tree):
    data['predictions'] = None
    for i in range(len(data)):
        query = data.iloc[i,:].drop('target').to_dict()
        data.loc[i,'predictions'] = 'p'
    accuracy = sum(data['predictions'] == data['target'])/len(data)*100
    return accuracy
        


# In[13]:


print("Accuracy of ID3 with testing data: ")
ID3_Test(testing_data,id3_tree)


# In[14]:


print("Accuracy of ID3 with training data: ")
ID3_Test(training_data,id3_tree)


# In[15]:


def RandomForest_Train(dataset,number_of_Trees):
    #create list for single forest
    random_forest_sub_tree = []
    
    #create number of n models 
    for i in range(number_of_Trees):
        #bootstrap sampled datasets from the original dataset 
        bootstrap_sample = dataset.sample(frac=1,replace=True)
        
        #training and testing subset of bootstrap sampled dataset 
        bootstrap_training_data = train_test_split(bootstrap_sample)[0]
        bootstrap_testing_data = train_test_split(bootstrap_sample)[1] 
        
        
        #grow tree model using recursion
        random_forest_sub_tree.append(ID3(bootstrap_training_data,bootstrap_training_data,bootstrap_training_data.drop(labels=['target'],axis=1).columns))
        
    return random_forest_sub_tree


# In[16]:


random_forest = RandomForest_Train(dataset,50)


# In[23]:


def RandomForest_Predict(query,random_forest,default='p'):
    predictions = []
    for tree in random_forest:
        predictions.append(predict(query,tree,default))
    return sps.mode(predictions)[0][0]


# In[24]:


query2 = training_data.iloc[0,:].drop('target').to_dict()
query2_target = training_data.iloc[0,0]
print('target: ',query2_target)
prediction = RandomForest_Predict(query2,random_forest)
print('prediction for random forest: ',prediction)


# In[25]:


query = testing_data.iloc[0,:].drop('target').to_dict()
query_target = testing_data.iloc[0,0]
print('target: ',query_target)
prediction = RandomForest_Predict(query,random_forest)
print('prediction for random forest: ',prediction)


# In[26]:


#test model on testing data and return the accuracy
def RandomForest_Test(data,random_forest):
    data['predictions'] = None
    for i in range(len(data)):
        query = data.iloc[i,:].drop('target').to_dict()
        data.loc[i,'predictions'] = RandomForest_Predict(query,random_forest,default='p')
    accuracy = sum(data['predictions'] == data['target'])/len(data)*100
    return accuracy
        


# In[21]:


print('The prediction accuracy with random forest on testing data is:')
RandomForest_Test(testing_data,random_forest)


# In[22]:


print('The prediction accuracy with random forest on training data is:')
RandomForest_Test(training_data,random_forest)


# In[ ]:




