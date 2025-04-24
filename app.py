from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
import os
import json
from datetime import datetime, timedelta

# --- Configuration Parameters ---
AVG_SPEED_KPH = 30
WORKDAY_DURATION_MIN = 600  # 10 hours
MAX_WAIT_TIME_MIN = 120
SERVICE_TIME_PER_STOP = 10
PRIORITY_LEVELS = 3

app = Flask(__name__, template_folder='frontend', static_folder='static')

# Secret key for sessions
app.secret_key = 'delivery_optimization_secret_key'

# --- Vehicle Parameters ---
vehicle_params = {
    'Van': {'efficiency': 8, 'emission': 2.31},
    'Truck': {'efficiency': 10, 'emission': 2.95},
    'Electric': {'efficiency': 12, 'emission': 0.0},
    'Diesel': {'efficiency': 10, 'emission': 2.95}
}

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Create session file for failed orders if it doesn't exist
FAILED_ORDERS_FILE = 'data/failed_orders.json'
if not os.path.exists(FAILED_ORDERS_FILE):
    with open(FAILED_ORDERS_FILE, 'w') as f:
        json.dump([], f)

def get_failed_orders():
    """Read failed orders from JSON file"""
    try:
        with open(FAILED_ORDERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_failed_orders(failed_orders):
    """Save failed orders to JSON file"""
    with open(FAILED_ORDERS_FILE, 'w') as f:
        json.dump(failed_orders, f)

def load_data():
    """Load data from CSV files"""
    try:
        drivers = pd.read_csv("data/drivers.csv")
        all_orders = pd.read_csv("data/delivery_orders_with_days.csv")
        distance_traffic = pd.read_csv("data/distance_traffic_matrix.csv")
        return drivers, all_orders, distance_traffic
    except FileNotFoundError as e:
        flash(f"Error loading data: {e}", "danger")
        return None, None, None

def prepare_orders_for_today(all_orders, failed_order_ids=None):
    """Prepare today's orders, including any failed orders from previous day"""
    # For demo purposes, we're using the first 9 orders as "today's orders"
    today_orders = all_orders.head(9).copy()
    
    # If we have failed orders, mark them in the dataframe
    if failed_order_ids:
        today_orders['is_failed_order'] = today_orders['delivery_location_id'].isin(failed_order_ids)
    else:
        today_orders['is_failed_order'] = False
        
    return today_orders

def prepare_orders_for_tomorrow(all_orders, failed_order_ids=None):
    """Prepare tomorrow's orders, including any failed orders from today"""
    if not failed_order_ids:
        failed_order_ids = []
        
    # Get failed orders from today
    failed_orders = all_orders.head(9)[all_orders.head(9)['delivery_location_id'].isin(failed_order_ids)].copy()
    failed_orders['is_rescheduled'] = True
    
    # Get next 9 regular orders for tomorrow
    regular_orders = all_orders.iloc[9:18].copy()
    regular_orders['is_rescheduled'] = False
    
    # Combine for tomorrow's schedule
    tomorrow_orders = pd.concat([regular_orders, failed_orders]).reset_index(drop=True)
    
    return tomorrow_orders

def solve_vrp(orders, drivers, distance_traffic):
    """Solve the Vehicle Routing Problem for the given orders and drivers"""
    # --- Matrix Preparation ---
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

    # Time Matrix Calculation
    time_matrix = [[(d / AVG_SPEED_KPH) * 60 for d in row] for row in distance_matrix]

    # --- OR-Tools Data Model ---
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
    
    # Mark rescheduled orders as high priority if possible
    if 'is_rescheduled' in orders.columns:
        for i, is_rescheduled in enumerate(orders['is_rescheduled']):
            if is_rescheduled:
                data['priorities'][i+1] = 1  # +1 because index 0 is the depot

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

    # --- Execute Solution ---
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return None

    # --- Solution Analysis ---
    total_stats = {
        'distance': 0,
        'emissions': 0,
        'orders': 0,
        'priority_orders': 0,
        'late_deliveries': 0
    }
    
    driver_stats = []
    driver_routes = []
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = {"stops": ["DEPOT"], "distance": 0, "emissions": 0, 
                "priority_orders": 0, "alerts": [], "delivery_stops": []}
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            
            if node != 0:  # Not depot
                location_id = locations[node]
                route['stops'].append(location_id)
                
                # Get order details for this stop
                order_info = orders[orders['delivery_location_id'] == location_id].iloc[0]
                
                # Create delivery stop info
                delivery_stop = {
                    'location_id': location_id,
                    'order_id': order_info['order_id'] if 'order_id' in order_info else f"Order-{location_id}",
                    'arrival_time': solution.Min(time_dimension.CumulVar(index)),
                    'due_time': data['time_windows'][node][1],
                    'is_late': solution.Min(time_dimension.CumulVar(index)) > data['time_windows'][node][1],
                    'is_rescheduled': order_info.get('is_rescheduled', False)
                }
                
                route['delivery_stops'].append(delivery_stop)
                
                # Track basic route info
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
        if len(route['stops']) > 2:  # Only include drivers with actual stops
            driver_id = drivers.iloc[vehicle_id]['driver_id']
            driver_stats.append({
                'id': driver_id,
                'name': f"Driver {driver_id}",
                'type': v_type,
                **route
            })
            driver_routes.append(route)

    # --- Sustainability Scoring ---
    emissions_per_order = [
        s['emissions']/(len(s['stops'])-2) if (len(s['stops'])-2) > 0 else 0 
        for s in driver_stats
    ]
    
    if emissions_per_order:
        min_emis, max_emis = min(emissions_per_order), max(emissions_per_order)
        
        for stats in driver_stats:
            # Emission score calculation
            try:
                emissions = stats['emissions']/(len(stats['stops'])-2)
                e_score = 100 - ((emissions - min_emis)/(max_emis - min_emis) * 100) if max_emis > min_emis else 100
            except ZeroDivisionError:
                e_score = 100
                
            # Penalty calculation
            penalties = len(stats['alerts'])*5 + stats['priority_orders']*2
            stats['final_score'] = max(0, min(100, e_score - penalties))
            stats['e_score'] = e_score
            stats['penalties'] = penalties

    return {
        'driver_stats': driver_stats,
        'total_stats': total_stats,
        'locations': locations
    }

def format_time(minutes):
    """Convert minutes to HH:MM format"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

@app.template_filter('format_time')
def format_time_filter(minutes):
    return format_time(minutes)

@app.route('/')
def index():
    """Dashboard view"""
    # Default empty stats dict to prevent 'stats is undefined' error
    stats = {
        'total_drivers': 0,
        'total_orders': 0,
        'today_orders': 0,
        'tomorrow_orders': 0,
        'failed_orders': 0,
        'today_date': datetime.now().strftime("%A, %B %d"),
        'tomorrow_date': (datetime.now() + timedelta(days=1)).strftime("%A, %B %d")
    }
    
    # Error flag
    error = None
    
    drivers, all_orders, _ = load_data()
    if drivers is None:
        error = "Error loading data"
    else:
        # Get stats for display
        failed_orders = get_failed_orders()
        
        # Format dates for display
        today = datetime.now().strftime("%A, %B %d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A, %B %d")
        
        stats = {
            'total_drivers': len(drivers),
            'total_orders': len(all_orders),
            'today_orders': 9,  # First 9 orders
            'tomorrow_orders': 9 + len(failed_orders),  # Next 9 orders + failed ones
            'failed_orders': len(failed_orders),
            'today_date': today,
            'tomorrow_date': tomorrow
        }
    
    return render_template('index.html', stats=stats, error=error)

@app.route('/routes')
def routes():
    """View optimized routes"""
    # Load the data
    drivers, all_orders, distance_traffic = load_data()
    if any(x is None for x in [drivers, all_orders, distance_traffic]):
        return redirect(url_for('index'))
    
    # Get failed orders from previous runs
    failed_order_ids = [order['location_id'] for order in get_failed_orders()]
    
    # Prepare today's orders (including any previously failed ones that are being retried)
    today_orders = prepare_orders_for_today(all_orders, failed_order_ids)
    
    # Solve the VRP
    solution = solve_vrp(today_orders, drivers, distance_traffic)
    if solution is None:
        flash("Could not find a solution for the routing problem", "danger")
        return redirect(url_for('index'))

    # Set time window display format
    for driver in solution['driver_stats']:
        for stop in driver['delivery_stops']:
            stop['arrival_time_fmt'] = format_time(stop['arrival_time'])
            stop['due_time_fmt'] = format_time(stop['due_time'])
    
    return render_template('routes.html', 
                          drivers=solution['driver_stats'],
                          stats=solution['total_stats'],
                          today_date=datetime.now().strftime("%A, %B %d"))

@app.route('/report_failed', methods=['GET', 'POST'])
def report_failed():
    """Form to report failed deliveries"""
    if request.method == 'POST':
        # Get the selected failed deliveries
        failed_locations = request.form.getlist('failed_locations')
        
        # Get existing failed orders
        failed_orders = get_failed_orders()
        
        # Process each failed location
        for location_id in failed_locations:
            # Check if already in failed orders
            if not any(order['location_id'] == location_id for order in failed_orders):
                # Add to failed orders with timestamp
                failed_orders.append({
                    'location_id': location_id,
                    'reported_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rescheduled_for': (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                })
        
        # Save the updated list
        save_failed_orders(failed_orders)
        
        flash(f"Successfully reported {len(failed_locations)} failed deliveries", "success")
        return redirect(url_for('failed_orders'))
    
    # For GET request, show the form
    # Load the data
    drivers, all_orders, distance_traffic = load_data()
    if any(x is None for x in [drivers, all_orders, distance_traffic]):
        return redirect(url_for('index'))
    
    # Get failed orders from previous runs
    failed_order_ids = [order['location_id'] for order in get_failed_orders()]
    
    # Prepare today's orders (including any previously failed ones that are being retried)
    today_orders = prepare_orders_for_today(all_orders, failed_order_ids)
    
    # Solve the VRP to get the routes
    solution = solve_vrp(today_orders, drivers, distance_traffic)
    if solution is None:
        flash("Could not find a solution for the routing problem", "danger")
        return redirect(url_for('index'))
    
    # Prepare data for template
    all_stops = []
    for driver in solution['driver_stats']:
        driver_id = driver['id']
        for stop in driver['delivery_stops']:
            # Add driver info to each stop
            stop['driver_id'] = driver_id
            stop['driver_name'] = driver['name']
            all_stops.append(stop)
    
    return render_template('report_failed.html', stops=all_stops)

@app.route('/failed_orders')
def failed_orders():
    """View rescheduled orders"""
    # Load the data
    drivers, all_orders, distance_traffic = load_data()
    if any(x is None for x in [drivers, all_orders, distance_traffic]):
        return redirect(url_for('index'))
    
    # Get failed orders from previous runs
    failed_orders_list = get_failed_orders()
    failed_order_ids = [order['location_id'] for order in failed_orders_list]
    
    # Prepare tomorrow's orders with the rescheduled failed ones
    tomorrow_orders = prepare_orders_for_tomorrow(all_orders, failed_order_ids)
    
    # Solve the VRP for tomorrow
    tomorrow_solution = solve_vrp(tomorrow_orders, drivers, distance_traffic)
    if tomorrow_solution is None:
        flash("Could not find a solution for tomorrow's routing problem", "danger")
        return redirect(url_for('index'))
    
    # Format dates for display
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%A, %B %d")
    
    # Set time window display format
    for driver in tomorrow_solution['driver_stats']:
        for stop in driver['delivery_stops']:
            stop['arrival_time_fmt'] = format_time(stop['arrival_time'])
            stop['due_time_fmt'] = format_time(stop['due_time'])
    
    # Count rescheduled orders
    rescheduled_count = sum(1 for driver in tomorrow_solution['driver_stats'] 
                          for stop in driver['delivery_stops'] 
                          if stop.get('is_rescheduled', False))
    
    return render_template('failed_orders.html', 
                          drivers=tomorrow_solution['driver_stats'],
                          stats=tomorrow_solution['total_stats'],
                          failed_orders=failed_orders_list,
                          tomorrow_date=tomorrow_date,
                          rescheduled_count=rescheduled_count)

@app.route('/reset_failed', methods=['POST'])
def reset_failed():
    """Reset the failed orders list"""
    save_failed_orders([])
    flash("Successfully reset failed orders list", "success")
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting Flask application on port 5000...")
    app.run(debug=True)