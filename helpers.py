import random
import numpy as np
import scipy.stats

parameter_dict = {
            "ideology":0,
            "economy":0,
            "social":0,
            "foreign":0,
            "environment":0,
            "authlib":0,
            "party":0,
            "fiscal":0,
            "civil":0,
            "nation":0,
            "immigration":0,
            "government":0,
            "religion":0,
            "healthcare":0
        }

def generate_disjoint_sets(data, sample_size):
    shuffled = data
    random.shuffle(shuffled)

    return [shuffled[i:i + sample_size] for i in range(0, len(shuffled), sample_size)]


def assign_elements_to_bins(elements, num_bins):
    # Step 1: Initialize empty bins
    bins = [[] for _ in range(num_bins)]
    
    # Step 2: Iterate over each element and randomly assign it to a bin
    random.shuffle(elements)
    for element in elements:
        random_bin = random.randint(0, num_bins - 1)
        bins[random_bin].append(element)
    
    return bins


def euclidean_distance(p,q):
    p = np.asarray(list(p.ideology_dict.values()))
    q = np.asarray(list(q.ideology_dict.values()))
    return np.linalg.norm(p-q)

def initialize_ideology(ideol_dict,mode='uniform'):
    
    mode_actions = {
        "normal": scipy.stats.norm.rvs(size=len(ideol_dict.values())),
        "uniform": scipy.stats.uniform.rvs(loc=0,scale=1,size=len(ideol_dict.values()))
    }
    
    new_values = mode_actions[mode]
    new_dict = dict(zip(ideol_dict.keys(),new_values))
    return new_dict


def apply_threshold(self):
    rows, cols = np.indices(self.model.align_mat.shape)
    original_indicies = np.stack((rows,cols),axis=-1).reshape(-1,2)
    flat_array = self.model.align_mat.flatten()
    filtered_mask = flat_array >= self.threshold
    return {
        'array':flat_array[filtered_mask],
        'initial index':original_indicies[filtered_mask]
    }
    
def mean_absolute_difference(array1,array2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    return np.abs(array1-array2).mean()