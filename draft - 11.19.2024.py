import helpers as hlp
import mesa
import numpy as np
import pandas
# import seaborn
import random
import matplotlib.pyplot as plt
# import bokeh
import math
# import pyvis
import uuid
import itertools
import scipy


class LegisAgent(mesa.Agent):
    def __init__(self,model):
        super().__init__(model)
        
        self.parameter_dict = hlp.parameter_dict
        self.voters = None
        self.rand_seed = random.getstate()
        #norm_dist = scipy.stats.truncnorm(a=0,b=float('inf'))
        ##self.threshold = random.uniform(0,(np.sqrt(len(self.parameter_dict.values()))))
        self.threshold = scipy.stats.truncnorm(a=0,b=float('inf'),loc=np.sqrt(len(self.parameter_dict.values()))).rvs(size=1)[0]
        self.invitations = set()
        

        
        
    # def initialize_ideology(self,ideol_dict):
        
    #     #new_values = [random.uniform(0,1) for x in ideol_dict.values()]
    #     new_values = scipy.stats.norm().rvs(size=len(ideol_dict.values()))
    #     new_dict = dict(zip(ideol_dict.keys(),new_values))
    #     return new_dict
    
    def step(self):
        #def alliance_phase(self):    
        def apply_threshold(self):
            rows, cols = np.indices(self.model.align_mat.shape)
            original_indicies = np.stack((rows,cols),axis=-1).reshape(-1,2)
            flat_array = self.model.align_mat.flatten()
            filtered_mask = flat_array >= self.threshold
            return {
                'array':flat_array[filtered_mask],
                'initial index':original_indicies[filtered_mask]
            }
        #print(f'Hi, I am agent {self.unique_id} and the most important thing to me is {max(self.ideology_dict, key=self.ideology_dict.get)}.')
        self.elig_members = apply_threshold(self)
        
class LegisParty:
    def __init__(self,unique_id):
        self.unique_id = unique_id

# def generate_disjoint_sets(data, sample_size):
#     import random
#     shuffled = data
#     random.shuffle(shuffled)

#     return [shuffled[i:i + sample_size] for i in range(0, len(shuffled), sample_size)]

# def assign_elements_to_bins(elements, num_bins):
#     # Step 1: Initialize empty bins
#     bins = [[] for _ in range(num_bins)]
    
#     # Step 2: Iterate over each element and randomly assign it to a bin
#     random.shuffle(elements)
#     for element in elements:
#         random_bin = random.randint(0, num_bins - 1)
#         bins[random_bin].append(element)
    
#     return bins

# def euclidean_distance(p,q):
#     p = np.asarray(list(p.ideology_dict.values()))
#     q = np.asarray(list(q.ideology_dict.values()))
#     return np.linalg.norm(p-q)

class LegisModel(mesa.Model):
    
    def __init__(self,n,seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        
        # Total voter population, initalized at 0 and will increase along with each LegisAgent
        self.electorate_size = 0
        self.step_number = 0
        self.voter_groups = []
        self.parties = []
        self.bills = []
        
        # Create agents
        for _ in range(self.num_agents):
            a = LegisAgent(self)
            a.ideology_dict = hlp.initialize_ideology(a.parameter_dict)
            
            # voter_pop_modifier = lambda x: x**2
            # a.voters = scipy.stats.binom.rvs(voter_pop_modifier(self.num_agents),0.5,size=1)[0]
            
            tmp_voters = Voters(self,a)
            self.voter_groups.append(tmp_voters)
            
            self.electorate_size+= tmp_voters.population
        
        # def euclidean_distance(p,q):
        #     p = np.asarray(list(p.ideology_dict.values()))
        #     q = np.asarray(list(q.ideology_dict.values()))
        #     return np.linalg.norm(p-q)
        
        self.align_mat = np.eye(self.num_agents)
        itr = np.nditer(self.align_mat,order='K',flags=['multi_index'])
        for n in itr:
            i,j = itr.multi_index[0],itr.multi_index[1]
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            self.align_mat[i,j] = hlp.euclidean_distance(agent_i,agent_j)
            
    def alliance_phase(self,initialize_type='random',manual_override=None,method='equal'):
        
        def rand_initialize_parties(self,initialize_type,manual_override,method):
            
            rand_seed = random.getstate()
            if initialize_type == 'random':
                # Creates a random number of parties based on some bounding conditions
                num_parties = random.randrange(1,(self.num_agents//10)+1)
            elif initialize_type == 'manual':
                num_parties = manual_override
            for x in range(num_parties):
                self.parties.append(LegisParty(unique_id = uuid.uuid4()))
            match method:
                case 'equal':
                    # Randomly groups agents into (approximately) equal groups
                    sample_size = self.num_agents//num_parties
                    grouped_agents = hlp.generate_disjoint_sets(list(self.agents),sample_size)
                case 'random':
                    grouped_agents = hlp.assign_elements_to_bins(list(self.agents),num_parties)
            for i in range(len(self.parties)):
                self.parties[i].members = grouped_agents[i]
        if not self.parties:
            rand_initialize_parties(self,initialize_type,manual_override,method)
        
    def housekeeping_phase(self):
        
        def calculate_party_metrics(party):
    
            ideology_array = np.array([list(member.ideology_dict.values()) for member in party.members])
            avg_array = np.mean(ideology_array,axis=0)
            max_array = np.max(ideology_array,axis=0)
            min_array = np.min(ideology_array,axis=0)
            return dict(zip(hlp.parameter_dict.keys(),np.column_stack((avg_array,max_array,min_array))))
        for i in self.parties:
            i.party_ideology_metrics = calculate_party_metrics(i)
            
            
        def get_party_distmat(parties,base_dim=3):
    
            base_mat = np.array([list(i.party_ideology_metrics.values()) for i in parties])
            party_size = len(parties)
            id_dict_size = len(hlp.parameter_dict)
            idnt = np.zeros((party_size,party_size,id_dict_size,base_dim))
            for i,j,k in np.ndindex((party_size,party_size,id_dict_size)):
                idnt[:,:,k] = base_mat[:,k]
            for i,j,k,z in np.ndindex(idnt.shape):
                idnt[i,j,k,z] = np.linalg.norm(
                    idnt[i,j,k,z] - idnt[j,i,k,z]
                )
            return idnt
        self.party_distmat = get_party_distmat(self.parties)
        party_pairs = itertools.combinations(self.parties,2)
        
    
    def step(self):
        self.agents.shuffle_do('step')
        
        
class Bill(LegisModel):
    def __init__(self,unique_id,model):
        
        self.unique_id = unique_id
        self.model = model
        model.bills.append(self)
        self.num_issues = random.randrange(1,len(hlp.parameter_dict))
        issues = random.sample(list(hlp.parameter_dict.keys()),self.num_issues)
        issue_values = scipy.stats.norm().rvs(size=self.num_issues)
        self.contents = dict(zip(issues,issue_values))
        
        
    def get_agent_alignments(self):
        
        alignment_list = []
        for i in self.model.agents:
            ideology_subset = {key: i.ideology_dict[key] for key in self.contents.keys()}
            agent_values = np.array(list(ideology_subset.values()))
            bill_values = np.array(list(self.contents.values()))
            alignment_list.append(np.linalg.norm(agent_values - bill_values))
        self.alignment_list = alignment_list
        
        
class Voters():
    def __init__(self,model,agent):
        voter_pop_modifier = lambda x: x**2
        self.legislator = agent
        self.population = scipy.stats.binom.rvs(voter_pop_modifier(model.num_agents),0.5,size=1)[0]
        self.ideology_dict = hlp.initialize_ideology(hlp.parameter_dict)