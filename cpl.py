import lpsolverfactory as lpf

# Load the data for Solomon instance R101.
data = lpf.load_solomon_data('R101')

# Create the optimization model.
model = lpf.LpProblem('TWVRP', lpf.LpMinimize)

# Create variables.
x = [lpf.LpVariable(i, cat='Binary') for i in range(data.num_nodes)]
y = [lpf.LpVariable(i, cat='Binary') for i in range(data.num_nodes)]

# Create constraints.
for i in range(data.num_nodes):
    model += x[i] + y[i] == 1

for i in range(data.num_nodes):
    for j in range(data.num_nodes):
        if data.distances[i][j] > 0:
            model += x[i] * y[j] <= data.distances[i][j]

# Set the objective function.
model += sum(data.demands[i] * x[i] for i in range(data.num_nodes))

# Solve the model.
solver = lpf.LpSolver('CPLEX')
solver.solve(model)

# Print the solution.
print(model.objective.value())
for i in range(data.num_nodes):
    print(x[i].value(), y[i].value())
