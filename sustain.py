import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# Load data
drivers = pd.read_csv("data/drivers.csv")
orders = pd.read_csv("data/delivery_orders_with_days.csv").head(9)
distance_traffic = pd.read_csv("data/distance_traffic_matrix.csv")

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
            if not traffic_data.empty:
                adjusted_distance = int(traffic_data['distance_km'].values[0] *
                                     traffic_data['traffic_multiplier'].values[0])
                row.append(adjusted_distance)
            else:
                row.append(9999)
    distance_matrix.append(row)

# Calculate time matrix (30 km/h average speed)
time_matrix = [[(d / 30) * 60 for d in row] for row in distance_matrix]

# Emission parameters with Diesel support
vehicle_efficiency = {
    'Van': 8,
    'Truck': 10,
    'Electric': 12,
    'Diesel': 10
}

emission_factors = {
    'Van': 2.31,
    'Truck': 2.95,
    'Electric': 0.0,
    'Diesel': 2.95
}

def get_vehicle_params(vehicle_type):
    """Safe parameter retrieval with defaults"""
    return {
        'efficiency': vehicle_efficiency.get(vehicle_type, 6),  # default 6 km/L
        'factor': emission_factors.get(vehicle_type, 2.5)       # default 2.5 kg/L
    }

def create_data_model():
    data = {
        'distance_matrix': distance_matrix,
        'time_matrix': time_matrix,
        'num_vehicles': len(drivers),
        'depot': 0,
        'demands': [0] + [1] * len(orders),
        'vehicle_capacities': drivers['max_daily_deliveries'].tolist(),
        'vehicle_types': drivers['vehicle_type'].tolist(),
        'service_times': [0] + [10] * len(orders),
        'priorities': [0] + np.random.randint(1, 4, len(orders)).tolist()
    }
    
    np.random.seed(42)
    data['time_windows'] = [[0, 600]]  # Depot
    for _ in range(len(orders)):
        start = np.random.randint(0, 480)
        data['time_windows'].append([start, start + 120])
    
    return data

data = create_data_model()

# Routing setup
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)

# Distance constraint
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]
dist_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)

# Capacity constraint
def demand_callback(from_index):
    return data['demands'][manager.IndexToNode(from_index)]
demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, 
                                      data['vehicle_capacities'], True, 'Capacity')

# Time window constraint
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(data['time_matrix'][from_node][to_node] + data['service_times'][from_node])
time_callback_index = routing.RegisterTransitCallback(time_callback)
routing.AddDimension(time_callback_index, 120, 600, False, 'Time')
time_dimension = routing.GetDimensionOrDie('Time')

# Priority constraints
for node in range(1, len(data['priorities'])):
    if data['priorities'][node] == 1:
        routing.AddDisjunction([manager.NodeToIndex(node)], 1000, 1)

# Solve
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30
solution = routing.SolveWithParameters(search_parameters)

# Results processing
if solution:
    total_stats = {'distance': 0, 'emissions': 0, 'orders': len(orders)}
    driver_stats = []
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route, alerts = ["DEPOT"], []
        route_distance = 0
        priority_count = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            
            if node != 0:
                route.append(locations[node])
                route_distance += data['distance_matrix'][node][manager.IndexToNode(next_index)]
                
                # Time window check
                arrival = solution.Min(time_dimension.CumulVar(index))
                tw_end = data['time_windows'][node][1]
                if arrival > tw_end:
                    alerts.append(f"Late arrival at {locations[node]} ({arrival} > {tw_end})")
                
                # Priority check
                if data['priorities'][node] == 1:
                    priority_count += 1
                    if arrival > (tw_end - 30):
                        alerts.append(f"Close call for HIGH priority at {locations[node]}")
            
            index = next_index

        # Finalize route and emissions
        route.append("DEPOT")
        vehicle_type = data['vehicle_types'][vehicle_id]
        params = get_vehicle_params(vehicle_type)
        emissions = (route_distance / params['efficiency']) * params['factor']
        
        driver_stats.append({
            'id': drivers.iloc[vehicle_id]['driver_id'],
            'type': vehicle_type,
            'distance': route_distance,
            'emissions': emissions,
            'orders': len(route)-2,
            'alerts': alerts,
            'priority': priority_count
        })
        total_stats['distance'] += route_distance
        total_stats['emissions'] += emissions

   # Sustainability scoring with full error protection
emissions_data = []
for s in driver_stats:
    if s['orders'] > 0:
        try:
            emissions_data.append(s['emissions']/s['orders'])
        except ZeroDivisionError:
            emissions_data.append(0)
    else:
        emissions_data.append(0)

if emissions_data:
    min_emis = min(emissions_data)
    max_emis = max(emissions_data)
else:
    min_emis = max_emis = 0

print("=== Delivery Plan ===")
for stats in driver_stats:
    # Base emission score calculation
    try:
        if max_emis != min_emis:
            e_score = 100 - ((stats['emissions']/stats['orders'] - min_emis)/(max_emis - min_emis) * 100)
        else:
            e_score = 100  # All scores equal when no variation
    except ZeroDivisionError:
        e_score = 100  # Fallback for identical emissions
    
    # Final score calculation with bounds
    try:
        final_score = max(0, min(100, 
            e_score - (len(stats['alerts'])*5 + (stats['priority']*2))
        ))
    except:
        final_score = max(0, min(100, e_score))
    
    print(f"\nDriver {stats['id']} ({stats['type']} - {stats['orders']} orders)")
    print(f"  Sustainability Score: {final_score:.1f}/100")
    print(f"  Distance: {stats['distance']}km | Emissions: {stats['emissions']:.2f}kg CO2")
    if stats['alerts']:
        print("  🚨 Alerts:")
        for alert in stats['alerts']:
            print(f"    - {alert}")

print("\n=== Summary ===")
print(f"Total distance: {total_stats['distance']}km")
print(f"Total emissions: {total_stats['emissions']:.2f}kg CO2")
print(f"Orders delivered: {total_stats['orders']}/{len(orders)}")
    