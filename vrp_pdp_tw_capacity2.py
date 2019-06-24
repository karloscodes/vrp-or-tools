"""Simple Pickup Delivery Problem (PDP)."""
# # https://gist.github.com/daoducminh/1dfa78d2885cadefa267c63c1de8c8a3

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np
from scipy.spatial import distance_matrix


def create_data_model():
    """Stores the data for the problem."""
    data = {}

    locations = [[0, 0], [8, 1], [2, 7], [3, 3],
                 [4, 4], [2, 6], [7, 3], [2, 6], 
                 [7, 3], [2, 6], [5, 6], [9, 3], 
                 [9, 1], [6, 5], [1, 8], [7, 2], [10, 0]]

    # using to ensure that distances are correct
    times = distance_matrix(locations, locations).astype(int)

    print("times shape: {}".format(times.shape))

    data["time_matrix"] = times

    print("time_matrix:")
    print(times)

    data['demands'] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]

    data["pickups_deliveries"] = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    data["time_windows"] = [[1, 19],
                            [1, 19],
                            [10, 30],
                            [15, 35],
                            [15, 30],
                            [15, 40],
                            [25, 45],
                            [30, 50]
                            ]

    assert(len(data["pickups_deliveries"]) == len(data["time_windows"]))

    data['vehicle_capacities'] = [50, 50]
    data['num_vehicles'] = 2
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            index = assignment.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(
            manager.IndexToNode(index), assignment.Min(time_var),
            assignment.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            assignment.Min(time_var))
        print(plan_output)
        total_time += assignment.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))


def main():
    """Solve the VRP with time windows."""
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Add time dimension
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        300,  # allow waiting time
        1000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add time window constraints for each location except depot.
    # also add pickups/dropoffs
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    
    for pd, tw in zip(data['pickups_deliveries'], data["time_windows"]):
        pickup, dropoff = pd
        routing.AddPickupAndDelivery(pickup, dropoff)
        routing.solver().Add(routing.VehicleVar(pickup) == routing.VehicleVar(dropoff))

        time_dimension.CumulVar(dropoff).SetRange(tw[0],
                                                  tw[1])

    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        print_solution(data, manager, routing, assignment)


if __name__ == '__main__':
    main()
