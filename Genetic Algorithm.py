import random
import sys

import numpy as np


class Item:

    def __init__(self, element_id, weight, value) -> None:
        self.ID = element_id
        self.weight = weight
        self.value = value


class Node:
    def __init__(self, element_id, x, y) -> None:
        self.ID = element_id
        self.x = x
        self.y = y


items = [
    Item(0, 3, 266),
    Item(1, 13, 442),
    Item(2, 10, 671),
    Item(3, 9, 526),
    Item(4, 7, 388),
    Item(5, 1, 245),
    Item(6, 8, 210),
    Item(7, 8, 145),
    Item(8, 2, 126),
    Item(9, 9, 322)
]

nodes = [
    Node(0, 119, 38),
    Node(1, 37, 38),
    Node(2, 197, 55),
    Node(3, 85, 165),
    Node(4, 12, 50),
    Node(5, 100, 53),
    Node(6, 81, 142),
    Node(7, 121, 137),
    Node(8, 85, 145),
    Node(9, 80, 197),
    Node(10, 91, 176),
    Node(11, 106, 55),
    Node(12, 123, 57),
    Node(13, 40, 81),
    Node(14, 78, 125),
    Node(15, 190, 46),
    Node(16, 187, 40),
    Node(17, 37, 107),
    Node(18, 17, 11),
    Node(19, 67, 56),
    Node(20, 78, 133),
    Node(21, 87, 23),
    Node(22, 184, 197),
    Node(23, 111, 12),
    Node(24, 66, 178)
]


class Chromosome:

    def __init__(self, num_of_genes, possibilities: [], can_repeat=True):
        if can_repeat:
            self.genes = [random.choice(possibilities) for _ in range(num_of_genes)]
        else:
            self.genes = random.sample(possibilities, num_of_genes)
        self.fit = 0

    def __str__(self):
        if any(getattr(node, 'ID', None) is None for node in self.genes):
            return f'{self.fit} | {" ".join(str(int(x)) for x in self.genes)}'
        else:
            return f'{self.fit * -1} | {" ".join(str(node.ID) for node in self.genes)}'


def mutation_swap(selected, prob):
    if random.random() < prob:
        id_1 = random.randint(0, len(selected.genes) - 1)
        id_2 = random.randint(0, len(selected.genes) - 1)
        selected.genes[id_1], selected.genes[id_2] = selected.genes[id_2], selected.genes[id_1]
    return selected


def mutation(selected, prob):
    if random.random() < prob:
        id_1 = random.randint(0, len(selected.genes) - 1)
        selected.genes[id_1] = 0 if selected.genes[id_1] == 1 else 1
    return selected


class Genetic:

    def __init__(self, possibilities, fitness_function):

        self.possibilities = possibilities
        self.population = []
        self.fitness_func = fitness_function

    def print_pop(self):

        print("==============================")
        print("BEST:", self.population[0])
        print("==============================\n")

    def generate_pop(self, num_of_individual, num_of_genes, can_repeat=True):

        self.population = [Chromosome(num_of_genes, self.possibilities, can_repeat) for _ in range(num_of_individual)]
        self.fitness()

    def fitness(self):
        for chromosome in self.population:
            chromosome.fit = self.fitness_func(chromosome)

        self.population = sorted(self.population, key=lambda x: x.fit, reverse=True)



    def rulet(self, num_of_output):
        selection_prob = [chromosome.fit / sum(ch.fit for ch in self.population) for chromosome in self.population]
        return random.choices(self.population, k=num_of_output, weights=selection_prob)

    # def rulet_3(self, num_of_output):
    #     list = []
    #     max = sum(ch.fit for ch in self.population)
    #     for i in range( num_of_output):
    #         number = random.uniform(0, max)
    #         temp = 0
    #         for i in self.population:
    #             temp += i.fit
    #             if temp > number:
    #                 list.append(i)
    #                 break
    #     return  list

    @staticmethod
    def crossover_one_point(parents):
        a_parent = parents[0]
        b_parent = parents[1]
        pivot = random.randrange(0, len(a_parent.genes) - 1)

        child_a = Chromosome(len(a_parent.genes), [0])
        child_b = Chromosome(len(b_parent.genes), [0])

        child_a.genes = np.append(a_parent.genes[:pivot], b_parent.genes[pivot:])
        child_b.genes = np.append(b_parent.genes[:pivot], a_parent.genes[pivot:])

        return child_a, child_b

    @staticmethod
    def order_cross_over(parents):
        parent_1 = parents[0]
        parent_2 = parents[1]

        length = len(parent_1.genes)

        if length < 2:
            return parent_1, parent_2

        start_point = random.randint(1, length - 1)
        end_point = random.randint(1, length - 1)

        start_index = min(start_point, end_point)
        end_index = max(start_point, end_point)

        temp = parent_1.genes[start_index:end_index]
        temp2 = [gene for gene in parent_2.genes if gene not in temp]

        child_genes = temp + temp2
        child = Chromosome(length, [0])
        child.genes = child_genes
        return child

    def evolve(self, iterations, max_fit, mut_prob, elit_percent, order=False , rulet_type= 0):

        stay_alive = int(len(self.population) * elit_percent)

        for i in range(iterations):
            self.fitness()

            if self.population[0].fit == max_fit:
                break

            print(i, ":", self.population[0])

            next_gen = self.population[:stay_alive]

            while len(next_gen) < len(self.population):

                parents = self.rulet(2)

                if order is False:
                    offspring_0, offspring_1 = self.crossover_one_point(parents)
                    offspring_0 = mutation_swap(offspring_0, mut_prob)
                    offspring_1 = mutation_swap(offspring_1, mut_prob)
                    next_gen.append(offspring_0)
                    next_gen.append(offspring_1)

                else:
                    offspring_0 = self.order_cross_over(parents)
                    offspring_0 = mutation_swap(offspring_0, mut_prob)
                    next_gen.append(offspring_0)

            self.population = next_gen
        self.print_pop()


# 1
def calculate_fitness_1(chromosome):
    return np.sum(chromosome.genes)


# 2
def decode(chromosome):
    a_value = chromosome.genes[0] + chromosome.genes[1] ** 2 + chromosome.genes[2] ** 3 + chromosome.genes[3] ** 4
    b_value = chromosome.genes[4] + chromosome.genes[5] ** 2 + chromosome.genes[6] ** 3 + chromosome.genes[7] ** 4

    return a_value, b_value


# 2
def calculate_fitness_2(chromosome):
    a_value, b_value = decode(chromosome)
    value = 2 * pow(a_value, 2) + b_value
    return 1 / (1 + abs(value - 33))


# 3
def calculate_fitness_backpack(chromosome):
    weight, value = 0, 0

    for i, item in enumerate(items):

        if chromosome.genes[i] == 1:
            weight += item.weight
            value += item.value

            if weight > 35:
                return -1000

    return value


# 4
def distance(point_a, point_b):
    x = abs(point_a.x - point_b.x)
    y = abs(point_a.y - point_b.y)
    return int(np.sqrt(pow(x, 2) + pow(y, 2)))


# 4
def calculate_fitness_tsp(chromosome):
    value = 0

    for i in range(1, len(chromosome.genes)):
        value += distance(chromosome.genes[i - 1], chromosome.genes[i])

    value += distance(chromosome.genes[-1], chromosome.genes[0])

    return -value


if __name__ == '__main__':
    runAll = False
    task_number = 0

    if len(sys.argv) != 2:
        runAll = True
    else:
        task_number = int(sys.argv[1])

    runAll = False
    task_number = 2

    if task_number == 1 or runAll is True:
        task_number = 1
        print(f'zad{task_number}\n')

        possible_genes = [0, 1]
        point = Genetic(possible_genes, fitness_function=calculate_fitness_1)
        point.generate_pop(num_of_individual=10, num_of_genes=10)
        point.evolve(100, max_fit=10, mut_prob=0.6, elit_percent=0)

    if task_number == 2 or runAll is True:
        task_number = 2
        print(f'zad{task_number}\n')

        possible_genes = [0, 1]
        point = Genetic(possible_genes, fitness_function=calculate_fitness_2)
        point.generate_pop(num_of_individual=10, num_of_genes=8)
        point.evolve(100, max_fit=0, mut_prob=0.5, elit_percent=0.2 , rulet_type= 1)
        a, b = decode(point.population[0])

        print("a ->", a, "b ->", b, "value->", 2 * a ** 2 + b)
        print("\n==============================\n")

    if task_number == 3 or runAll is True:
        task_number = 3
        print(f'zad{task_number}\n')

        possible_genes = [0, 1]
        point = Genetic(possible_genes, fitness_function=calculate_fitness_backpack)
        point.generate_pop(num_of_individual=8, num_of_genes=len(items))
        point.evolve(100, max_fit=2222, mut_prob=0.05, elit_percent=0.25)

    if task_number == 4 or runAll is True:
        task_number = 4
        print(f'zad{task_number}\n')

        possible_genes = nodes
        point = Genetic(possible_genes, fitness_function=calculate_fitness_tsp)
        point.generate_pop(num_of_individual=100, num_of_genes=len(nodes), can_repeat=False)
        point.evolve(1000000, max_fit=0, mut_prob=0.01, elit_percent=0.2, order=True)
