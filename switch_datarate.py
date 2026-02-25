import numpy as np
import json
graph_swrate = {}
def custom_distribution(target_sum, num_numbers, distribution='normal'):
    while True:
        if distribution == 'normal':
            mean = target_sum / num_numbers
            std_dev = mean / 2  # Adjust standard deviation as needed
            numbers = np.random.normal(mean, std_dev, num_numbers)
        elif distribution == 'uniform':
            numbers = np.random.uniform(1, target_sum, num_numbers)
        else:
            raise ValueError("Invalid distribution")

        # Check for negative numbers
        if np.any(numbers < 0):
            continue  # Re-generate if negative numbers found

        # Adjust numbers to ensure they sum to target_sum
        total = sum(numbers)
        factor = target_sum / total
        numbers = [int(num * factor) for num in numbers]
        return numbers

target_sum = 22600
num_numbers = 113
for x in range(1000):
	result = custom_distribution(target_sum, num_numbers, 'normal')
	graph_swrate[x] = result
#print(result)
#print(sum(result))
print (graph_swrate)
with open("graph_swrate.json", 'w') as file:
    json.dump(graph_swrate, file, indent=4) 