import numpy as np 
import random

class A1:
    def generate_markov_chain(self, potential_states, sequence_of_states):
       self.potential_states = potential_states
       n = len(potential_states)
       
       index_of = {state: i for i, state in enumerate(potential_states)}
       
       counts = np.zeros((n,n), dtype=float)
       
       for i in range(len(sequence_of_states)-1):
           current_state = sequence_of_states[i]
           next_state = sequence_of_states[i +1]
           row = index_of[current_state]
           col = index_of[next_state]
           counts[row, col] += 1
        
       row_sums = counts.sum(axis=1, keepdims=True)
       row_sums[row_sums == 0] = 1   
           
       self.transition_matrix = counts / row_sums
       return self.transition_matrix
       
    def generate_samples(self, first_state, random_seed, length):
        random.seed(random_seed)
        sequence = [first_state]
        current_state = first_state 
        
        for _ in range(length):
            row_index = self.potential_states.index(current_state)
            probabilities = self.transition_matrix[row_index]
            
            r = random.random()
            cumulative = 0.0
            next_state = self.potential_states[-1]
            for state_index, prob in enumerate(probabilities):
                cumulative += prob
                if r < cumulative:
                    next_state = self.potential_states[state_index]
                    break 
            sequence.append(next_state)
            current_state = next_state         
        return sequence
     
    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues -1))
        vector = np.real(eigenvectors[:, idx])
        vector = vector / vector.sum()
        return vector