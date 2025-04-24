import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# Load data
drivers = pd.read_csv("data/drivers.csv")
all_orders = pd.read_csv("data/delivery_orders_with_days.csv")

# Identify failed orders from yesterday
failed_order_ids = ['LOC013', 'LOC007']
failed_orders_mask = all_orders.head(9)['delivery_location_id'].isin(failed_order_ids)
failed_orders = all_orders.head(9)[failed_orders_mask].copy()

print("✅ Failed orders found:")
print(failed_orders[['order_id', 'delivery_location_id']])

# Prepare next day's orders
# Grab next 9 in sequence (to ensure total stays within typical routing constraints)
next_day_orders = pd.concat([
    all_orders.iloc[9:18].copy(),  # 9 normal orders
    failed_orders                  # Add up to 2 failed ones
]).reset_index(drop=True)

print("\n✅ Orders for the next day (including failed ones):")
print(next_day_orders[['order_id', 'delivery_location_id']])

# Create efficient distance lookup matrix with NaN handling
distance_traffic = pd.read_csv("data/distance_traffic_matrix.csv")

# Create location list with depot
locations = ['DEPOT'] + next_day_orders['delivery_location_id'].unique().tolist()

# Ensure all failed delivery locations are present
for loc in failed_order_ids:
    if loc not in locations:
        print(f"❌ WARNING: Failed location {loc} not included in routing locations!")

print("\n✅ All locations including depot:")
print(locations)

# Create distance matrix with traffic adjustment
distance_matrix = []
for from_loc in locations:
    row = []
    for to_loc in locations:
        if from_loc == to_loc:
            row.append(0)
        else:
            traffic_data = distance_traffic[
                (distance_traffic['from_location_id'] == from_loc) &
                (distance_traffic['to_location_id'] == to_loc)
            ]
            if not traffic_data.empty:
                distance = int(traffic_data['distance_km'].values[0] *
                               traffic_data['traffic_multiplier'].values[0])
                row.append(distance)
            else:
                print(f"⚠️ Missing distance from {from_loc} to {to_loc}, setting to large value")
                row.append(9999)
    distance_matrix.append(row)

# Time matrix (assuming 30 km/h)
time_matrix = [[(d / 30) * 60 for d in row] for row in distance_matrix]

# VRP setup
def create_data_model():
    data = {}
    data['distance_matrix'] = distance_matrix
    data['time_matrix'] = time_matrix
    data['num_vehicles'] = len(drivers)
    data['depot'] = 0

    # Demand and capacity
    data['demands'] = [0] + [1] * len(next_day_orders)
    data['vehicle_capacities'] = drivers['max_daily_deliveries'].tolist()

    # Time windows
    np.random.seed(42)
    data['time_windows'] = [[0, 600]]
    for _ in range(len(next_day_orders)):
        start = np.random.randint(0, 480)
        data['time_windows'].append([start, start + 120])

    # Service times
    data['service_times'] = [0] + [10] * len(next_day_orders)

    # Priorities
    data['priorities'] = [0] + [np.random.randint(1, 4) for _ in range(len(next_day_orders))]

    return data

data = create_data_model()

manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])

routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

dist_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)

def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    data['vehicle_capacities'],
    True,
    'Capacity'
)

def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(data['time_matrix'][from_node][to_node] + data['service_times'][from_node])

time_callback_index = routing.RegisterTransitCallback(time_callback)
routing.AddDimension(
    time_callback_index,
    120,
    600,
    False,
    'Time'
)

time_dimension = routing.GetDimensionOrDie('Time')
for location_idx, time_window in enumerate(data['time_windows']):
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

# Penalize not serving high-priority nodes
for node_idx in range(1, len(data['priorities'])):
    if data['priorities'][node_idx] == 1:
        index = manager.NodeToIndex(node_idx)
        routing.AddDisjunction([index], 1000, 1)

time_dimension.SetGlobalSpanCostCoefficient(100)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.seconds = 30

solution = routing.SolveWithParameters(search_parameters)

# Print solution
if solution:
    print("\n✅ ROUTING SOLUTION FOUND:")
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

            route_distance += distance_callback(previous_index, index)

            time_var = time_dimension.CumulVar(index)
            arrival_time = solution.Min(time_var)
            priority = data['priorities'][node]
            priority_label = {1: "HIGH", 2: "Medium", 3: "Standard"}.get(priority, "Standard")

            print(f"  - {location_name} (Arrival: {arrival_time} min, Priority: {priority_label})")

            previous_index = index
            index = solution.Value(routing.NextVar(index))
        route_distance += distance_callback(previous_index, routing.End(vehicle_id))
        route.append("DEPOT")

        print(f"\nDriver {drivers['driver_id'][vehicle_id]}")
        print(f"  Route: {' -> '.join(route)}")
        print(f"  Total distance: {route_distance} km")
        print(f"  Orders delivered: {len(route) - 2}")
        print()

        total_distance += route_distance

    print(f"✅ Total distance: {total_distance} km")
    print(f"✅ Total orders delivered: {len(next_day_orders)}")
else:
    print("❌ No solution found!")
