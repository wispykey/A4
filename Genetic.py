import numpy
import random

class Genetic():
    def __init__(self, fitness_fn) -> None:
        self.fitness_fn = fitness_fn # The given fitness function
        self.generation_number = 0

    # Creates n generations given an initial population 
    def create_n_generations(self, population, n) -> None:
        print("\nGeneration:", self.generation_number, "\n")

        next_generation = []
        weights = tuple(self.calculate_probabilities(population))
        
        for i in range(len(population) // 2):
            print("SELECTING...")

            parents = self.select_parents(population, weights)
            
            print("Parent 1: ", parents[0])
            print("Parent 2: ", parents[1])

            children = self.reproduce_children(parents) 
            mutated_children = self.mutate_children(children)

            next_generation.extend(mutated_children)

        if (n - 1 >= 0):
            self.generation_number += 1
            self.create_n_generations(next_generation, n - 1)
        
        return None

    # Select parents based on weighted probabilities
    def select_parents(self, population, weights) -> list:
        parent1 = random.choices(population, weights, k=1)
        parent2 = random.choices(population, weights, k=1) 

        while (parent2 == parent1):
            # Reroll until parents are different
            parent2 = random.choices(population, weights, k=1)

        return numpy.concatenate((parent1, parent2))
    
    # Creates new children based on parents
    def reproduce_children(self, parents) -> list:
        num_variables = len(parents[0])
         # The crossover point is interpreted as inclusive to first child
        crossover_point = random.randrange(1, num_variables)

        print("REPRODUCING...")
        print("Crossover point:", chr(crossover_point + 65 - 1))

        child1 = self.produce_child(parents, crossover_point, num_variables)
        child2 = self.produce_child(parents[::-1], crossover_point, num_variables) 

        return [child1, child2]
    
    # Creates a new child by splicing the parents together
    def produce_child(self, parents, crossover_point, num_variables) -> object:
        child = {}

        for i in range(0, crossover_point):
            letter = chr(i + 65)
            child[letter] = parents[0][letter]
        for j in range(crossover_point, num_variables):
            letter = chr(j + 65)
            child[letter] = parents[1][letter]

        return child
    
    # Attempts to mutate a single variable from each child
    def mutate_children(self, children) -> list:
        for i in range(len(children)):
            if (random.randrange(100) <= 30): # Mimic a 30% probability of mutation
                mutate_variable = chr(random.randrange(len(children[0])) + 65)
                mutate_value = random.choice([1,2,3,4]) # Hard-coded domain
                
                while (mutate_value == children[i][mutate_variable]):          
                    mutate_value = random.choice([1,2,3,4]) # Reroll until different                  

                print("Mutation in Child " + str(i + 1) + "!", end=" ") 
                print("Changed", mutate_variable, "from", 
                      children[i][mutate_variable], "to", mutate_value)

                children[i][mutate_variable] = mutate_value              
            
            print("Child  " + str(i + 1) + ": ", children[i])
        print("\n")

        return children

    # Computes the probability of each state being picked as a parent
    def calculate_probabilities(self, population) -> list:
        print("Fitness: ")

        weights = []
        fitness_score_sum = self.sum_fitness_scores(population)

        for i in range(len(population)):
            fitness_score = self.fitness_fn(population[i])
            probability = fitness_score / fitness_score_sum
            weights.append(probability)

            print(population[i], "| Score:", str(fitness_score) + 
                  " (" + f"{probability * 100:.2f}" + "%)")
        print("\n")

        return weights
    
    # Computes the sum of all the fitness scores of each state
    def sum_fitness_scores(self, population) -> int:
        fitness_score_sum = 0
        for p in population:
            fitness_score_sum += self.fitness_fn(p)
        return fitness_score_sum


"""
Outside of class definition
"""

# Sums the number of satisfied constraints for a given state
def num_satisfied_constraints(state) -> int:
    return ( 
        (state["A"] > state["G"]) + 
        (state["A"] <= state["H"]) + 
        (abs(state["F"] - state["B"]) == 1) + 
        (state["G"] < state["H"]) + 
        (abs(state["G"] - state["C"]) == 1) + 
        (abs(state["H"] - state["C"]) % 2 == 0) + 
        (state["H"] != state["D"]) + 
        (state["D"] >= state["G"]) + 
        (state["D"] != state["C"]) + 
        (state["E"] != state["C"])+ 
        (state["E"] < (state["D"] - 1)) + 
        (state["E"] != (state["H"] - 2)) + 
        (state["G"] != state["F"]) + 
        (state["H"] != state["F"]) + 
        (state["C"] != state["F"]) + 
        (state["D"] != (state["F"] - 1)) + 
        (abs(state["E"] - state["F"]) % 2 == 1))

state1 = {"A":1, "B":1, "C":1, "D":1, "E":1, "F":1, "G":1, "H":1}
state2 = {"A":2, "B":2, "C":2, "D":2, "E":2, "F":2, "G":2, "H":2}
state3 = {"A":3, "B":3, "C":3, "D":3, "E":3, "F":3, "G":3, "H":3}
state4 = {"A":4, "B":4, "C":4, "D":4, "E":4, "F":4, "G":4, "H":4}
state5 = {"A":1, "B":2, "C":3, "D":4, "E":1, "F":2, "G":3, "H":4}
state6 = {"A":4, "B":3, "C":2, "D":1, "E":4, "F":3, "G":2, "H":1}
state7 = {"A":1, "B":2, "C":1, "D":2, "E":1, "F":2, "G":1, "H":2}
state8 = {"A":3, "B":4, "C":3, "D":4, "E":3, "F":4, "G":3, "H":4}
initial_population = [state1, state2, state3, state4, state5, state6, state7, state8]

algorithm = Genetic(num_satisfied_constraints) # Initialize fitness function
algorithm.create_n_generations(initial_population, 5)

