import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# Load data
drivers = pd.read_csv('drivers.csv')
orders = pd.read_csv('delivery_orders.csv').head(10)  # Use first 10 orders
distance_traffic = pd.read_csv('distance_traffic_matrix.csv')

# Create location list with depot
locations = ['DEPOT'] + orders['delivery_location_id'].unique().tolist()

# Create distance matrix with traffic adjustment
distance_matrix = []
for from_loc in locations:
    row = []
    for to_loc in locations:
        if from_loc == to_loc:
            row.append(0)
        else:
            traffic_data = distance_traffic[(distance_traffic['from_location_id'] == from_loc) &
                                            (distance_traffic['to_location_id'] == to_loc)]
            row.append(int(traffic_data['distance_km'].values[0] *
                           traffic_data['traffic_multiplier'].values[0]))
    distance_matrix.append(row)

# Calculate time matrix (assuming average speed of 30 km/h)
time_matrix = []
for row in distance_matrix:
    time_row = [(d / 30) * 60 for d in row]
    time_matrix.append(time_row)


# VRP setup
def create_data_model():
    data = {}
    data['distance_matrix'] = distance_matrix
    data['time_matrix'] = time_matrix
    data['num_vehicles'] = len(drivers)
    data['depot'] = 0

    # Each order counts as 1 unit demand
    data['demands'] = [0] + [1] * len(orders)

    # Vehicle capacities
    data['vehicle_capacities'] = drivers['max_daily_deliveries'].tolist()

    # Time windows [start_time, end_time] in minutes from 8:00 AM
    # Depot is open from 8:00 AM to 6:00 PM (0 to 600 minutes)
    np.random.seed(42)  # For reproducibility
    data['time_windows'] = [[0, 600]]  # Depot time window
    for _ in range(len(orders)):
        start = np.random.randint(0, 480)  # Random start time between 8:00 AM and 4:00 PM
        data['time_windows'].append([start, start + 120])  # 2-hour window

    # Service times (minutes spent at each location)
    data['service_times'] = [0]  # No service time at depot
    for _ in range(len(orders)):
        data['service_times'].append(10)  # 10 minutes for each delivery

    # Priority orders (higher priority = lower number)
    data['priorities'] = [0]  # No priority for depot
    for _ in range(len(orders)):
        data['priorities'].append(np.random.randint(1, 4))  # Priority levels 1-3

    return data


data = create_data_model()

# Create routing index manager
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])

# Create routing model
routing = pywrapcp.RoutingModel(manager)


# Register distance callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]


dist_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)


# Add capacity constraint
def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]


demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # null capacity slack
    data['vehicle_capacities'],  # vehicle maximum capacities
    True,  # start cumul to zero
    'Capacity'
)


# Add time window constraints
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    # Travel time + service time at from_node
    return int(data['time_matrix'][from_node][to_node] + data['service_times'][from_node])


time_callback_index = routing.RegisterTransitCallback(time_callback)

routing.AddDimension(
    time_callback_index,
    120,  # Allow waiting time (slack) of up to 2 hours
    600,  # Maximum time per vehicle (10 hours = 600 minutes)
    False,  # Don't force start cumul to zero
    'Time'
)

time_dimension = routing.GetDimensionOrDie('Time')

# Add time window constraints for each location
for location_idx, time_window in enumerate(data['time_windows']):
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

# Add priority by penalizing late delivery through global span cost
for node_idx in range(1, len(data['priorities'])):  # Skip depot
    if data['priorities'][node_idx] == 1:  # High priority orders
        index = manager.NodeToIndex(node_idx)
        # Adding a penalty for visiting high priority nodes late
        routing.AddDisjunction([index], 1000, 1)  # High penalty for not visiting

# Set global span cost coefficient to balance route lengths
time_dimension.SetGlobalSpanCostCoefficient(100)

# Setup search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30  # 30 seconds time limit

# Solve
solution = routing.SolveWithParameters(search_parameters)

# Print solution
if solution:
    print("Routes:")
    total_distance = 0

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = ["DEPOT"]
        route_distance = 0

        previous_index = index
        index = solution.Value(routing.NextVar(index))

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            location_name = locations[node]
            route.append(location_name)

            # Add distance
            route_distance += distance_callback(previous_index, index)

            # Time window information
            time_var = time_dimension.CumulVar(index)
            time_min = solution.Min(time_var)
            time_max = solution.Max(time_var)
            arrival_time = time_min

            # Check if this is a priority order
            priority = "Standard"
            if node < len(data['priorities']) and data['priorities'][node] == 1:
                priority = "HIGH"
            elif node < len(data['priorities']) and data['priorities'][node] == 2:
                priority = "Medium"

            print(f"  - {location_name} (Arrival: {arrival_time} min, Priority: {priority})")

            previous_index = index
            index = solution.Value(routing.NextVar(index))

        # Add distance back to depot
        route_distance += distance_callback(previous_index, routing.End(vehicle_id))
        route.append("DEPOT")

        print(f"\nDriver {drivers['driver_id'][vehicle_id]}")
        print(f"  Route: {' -> '.join(route)}")
        print(f"  Total distance: {route_distance} km")
        print(f"  Orders delivered: {len(route) - 2}")  # Excluding depot visits
        print()

        total_distance += route_distance

    print(f"Total distance: {total_distance} km")
    print(f"Total orders delivered: {len(orders)}")
else:
    print("No solution found!")