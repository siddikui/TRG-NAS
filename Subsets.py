# An experimental class to extract data subsets with correct labels no matter which classes
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,labels):
        self.dataset = dataset
        self.indices = indices
        labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = labels 
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)


def get_data_subsets(dataset, class_labels):
    
    """
    returns: data subset with modified labels to avoid error at training.
    moreover, unlike v0.2, can take any classes of choice.
    """
    
    #Get all targets tuples
    targets = np.array(dataset.targets )
    print("Total samples: ",targets.shape,"Samples dtype: ", targets.dtype)
    # Classes we want to extract
    class_labels = class_labels
    inds = np.squeeze(np.array(np.where(np.in1d(targets,class_labels))))
    print('Extracted samples: ',inds.shape)
    
    #print('Actual Target Indices', targets[inds][0:20])
    new_targets = targets[inds]
        
    # Because the originally selected class labels maybe random, therefore renaming them.    
    mylabels = np.arange(len(class_labels))
    myDict = defaultdict(int)

    i=0
    for c in class_labels:
        myDict[c] = i
        i+=1

    for i in range(len(new_targets)):  
        new_targets[i] = myDict[new_targets[i]]
        
    #print('Modifi Target Indices',new_targets[0:20])
    print('New Target Shape: ', new_targets.shape)
    
    dataset = my_subset(dataset, inds, torch.from_numpy(new_targets))
    
    return dataset