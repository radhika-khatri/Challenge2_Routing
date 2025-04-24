import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# --- Configuration Parameters ---
AVG_SPEED_KPH = 30
WORKDAY_DURATION_MIN = 600  # 10 hours
MAX_WAIT_TIME_MIN = 120
SERVICE_TIME_PER_STOP = 10
PRIORITY_LEVELS = 3

def main():
    # --- Data Loading ---
    print("\n=== Loading Data ===")
    try:
        drivers = pd.read_csv("data/drivers.csv")
        orders = pd.read_csv("data/delivery_orders_with_days.csv").head(9)
        distance_traffic = pd.read_csv("data/distance_traffic_matrix.csv")
        print(f"✅ Loaded {len(drivers)} drivers, {len(orders)} orders, and traffic data")
    except FileNotFoundError as e:
        print(f"❌ Critical error loading data: {e}")
        return

    # --- Matrix Preparation ---
    print("\n=== Preparing Matrices ===")
    locations = ['DEPOT'] + orders['delivery_location_id'].unique().tolist()
    location_count = len(locations)
    
    # Distance Matrix with Traffic
    distance_matrix = []
    missing_pairs = []
    for from_loc in locations:
        row = []
        for to_loc in locations:
            if from_loc == to_loc:
                row.append(0)
                continue
                
            traffic_data = distance_traffic[
                (distance_traffic['from_location_id'] == from_loc) &
                (distance_traffic['to_location_id'] == to_loc)
            ]
            
            if not traffic_data.empty:
                adjusted = int(traffic_data['distance_km'].values[0] * 
                            traffic_data['traffic_multiplier'].values[0])
                row.append(adjusted)
            else:
                row.append(9999)
                if from_loc != 'DEPOT' and to_loc != 'DEPOT':
                    missing_pairs.append(f"{from_loc}→{to_loc}")
        distance_matrix.append(row)
    
    if missing_pairs:
        print(f"⚠️ Missing {len(missing_pairs)} location pairs (set to 9999): {', '.join(missing_pairs[:3])}...")

    # Time Matrix Calculation
    time_matrix = [[(d / AVG_SPEED_KPH) * 60 for d in row] for row in distance_matrix]
    print(f"✅ Created {location_count}x{location_count} distance/time matrices")

    # --- Vehicle Parameters ---
    vehicle_params = {
        'Van': {'efficiency': 8, 'emission': 2.31},
        'Truck': {'efficiency': 10, 'emission': 2.95},
        'Electric': {'efficiency': 12, 'emission': 0.0},
        'Diesel': {'efficiency': 10, 'emission': 2.95}
    }

    # --- OR-Tools Data Model ---
    print("\n=== Initializing VRP Model ===")
    data = {
        'distance_matrix': distance_matrix,
        'time_matrix': time_matrix,
        'num_vehicles': len(drivers),
        'depot': 0,
        'demands': [0] + [1]*len(orders),
        'vehicle_capacities': drivers['max_daily_deliveries'].tolist(),
        'vehicle_types': drivers['vehicle_type'].tolist(),
        'service_times': [0] + [SERVICE_TIME_PER_STOP]*len(orders),
    }

    # Time Windows Generation
    np.random.seed(42)
    data['time_windows'] = [[0, WORKDAY_DURATION_MIN]]  # Depot
    for _ in range(len(orders)):
        start = np.random.randint(0, WORKDAY_DURATION_MIN-MAX_WAIT_TIME_MIN)
        data['time_windows'].append([start, start + MAX_WAIT_TIME_MIN])

    # Priority Orders (1 = highest)
    data['priorities'] = [0] + np.random.randint(1, PRIORITY_LEVELS+1, len(orders)).tolist()
    print("✅ Model data prepared with time windows and priorities")

    # --- OR-Tools Setup ---
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot']
    )
    routing = pywrapcp.RoutingModel(manager)

    # Distance Constraint
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    dist_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)

    # Capacity Constraint
    def demand_callback(from_index):
        return data['demands'][manager.IndexToNode(from_index)]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity'
    )

    # Time Constraint
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['time_matrix'][from_node][to_node] + 
                 data['service_times'][from_node])
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index, MAX_WAIT_TIME_MIN, WORKDAY_DURATION_MIN, False, 'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Apply Time Windows
    for location_idx, (start, end) in enumerate(data['time_windows']):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(start, end)

    # Priority Handling
    for node in range(1, len(data['priorities'])):
        if data['priorities'][node] == 1:
            routing.AddDisjunction([manager.NodeToIndex(node)], 100000, 1)

    # --- Solver Configuration ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30
    print("✅ Solver configured with guided local search")

    # --- Execute Solution ---
    print("\n=== Solving VRP ===")
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        print("❌ No solution found!")
        return

    # --- Solution Analysis ---
    print("\n=== Solution Details ===")
    total_stats = {
        'distance': 0,
        'emissions': 0,
        'orders': 0,
        'priority_orders': 0,
        'late_deliveries': 0
    }
    
    driver_stats = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = {"stops": ["DEPOT"], "distance": 0, "emissions": 0, 
                "priority_orders": 0, "alerts": []}
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            
            if node != 0:
                # Track basic route info
                route['stops'].append(locations[node])
                leg_distance = data['distance_matrix'][node][manager.IndexToNode(next_index)]
                route['distance'] += leg_distance
                
                # Track priorities
                if data['priorities'][node] == 1:
                    route['priority_orders'] += 1
                    total_stats['priority_orders'] += 1
                
                # Time window checks
                arrival = solution.Min(time_dimension.CumulVar(index))
                tw_end = data['time_windows'][node][1]
                if arrival > tw_end:
                    route['alerts'].append(
                        f"Late arrival at {locations[node]} ({arrival} > {tw_end})"
                    )
                    total_stats['late_deliveries'] += 1
            
            index = next_index

        # Finalize route metrics
        route['stops'].append("DEPOT")
        v_type = data['vehicle_types'][vehicle_id]
        params = vehicle_params.get(v_type, {'efficiency': 10, 'emission': 2.5})
        route['emissions'] = (route['distance'] / params['efficiency']) * params['emission']
        
        # Update totals
        total_stats['distance'] += route['distance']
        total_stats['emissions'] += route['emissions']
        total_stats['orders'] += len(route['stops']) - 2  # Exclude depot
        
        # Driver-specific stats
        driver_stats.append({
            'id': drivers.iloc[vehicle_id]['driver_id'],
            'type': v_type,
            **route
        })

    # --- Sustainability Scoring ---
    emissions_per_order = [
        s['emissions']/(len(s['stops'])-2) if (len(s['stops'])-2) > 0 else 0 
        for s in driver_stats
    ]
    min_emis, max_emis = min(emissions_per_order), max(emissions_per_order)
    
    for stats in driver_stats:
        # Emission score calculation
        try:
            emissions = stats['emissions']/(len(stats['stops'])-2)
            e_score = 100 - ((emissions - min_emis)/(max_emis - min_emis) * 100)
        except ZeroDivisionError:
            e_score = 100
            
        # Penalty calculation
        penalties = len(stats['alerts'])*5 + stats['priority_orders']*2
        final_score = max(0, min(100, e_score - penalties))
        
        # Display results
        print(f"\nDriver {stats['id']} ({stats['type']})")
        print(f"  Route: {' → '.join(stats['stops'])}")
        print(f"  Distance: {stats['distance']}km | Emissions: {stats['emissions']:.1f}kg")
        print(f"  Score: {final_score:.1f}/100 (Base: {e_score:.1f}, Penalties: -{penalties})")
        if stats['alerts']:
            print("  🚨 Alerts:")
            for alert in stats['alerts']:
                print(f"    - {alert}")

    # --- Final Summary ---
    print("\n=== Operational Summary ===")
    print(f"Total Distance: {total_stats['distance']}km")
    print(f"Total Emissions: {total_stats['emissions']:.1f}kg CO2")
    print(f"Orders Delivered: {total_stats['orders']}/{len(orders)}")
    print(f"Late Deliveries: {total_stats['late_deliveries']}")

if __name__ == "__main__":
    main()