import numpy as np
import random
import copy
import math
import vrplib


class Node:
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        demand: float,
        ready_time: float,
        due_time: float,
        service_time: float,
    ):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


class Route:
    def __init__(self, id, data, travel_path):
        self.id = id
        self.travel_path = travel_path
        self.data = data
        self.dist_matrix = self.data[2]
        self.vehicle_capacity = self.data[4]
        self.nodes = self.data[1]
        self.load = self.calculate_route_load()

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    def calculate_route_dist(self):
        distance = 0
        current_ind = self.travel_path[0]
        for next_ind in self.travel_path[1:]:
            distance += self.dist_matrix[current_ind][next_ind]
            current_ind = next_ind
        self.route_distance = distance
        return distance

    def calculate_route_load(self):
        load = 0
        for node in self.travel_path:
            load += self.nodes[node].demand
        self.load = load
        return load

    def check_capacity_constrain(self):
        self.calculate_route_load()
        if self.load > self.vehicle_capacity:
            return False
        else:
            return True

    def get_current_time(self):
        current_time = 0
        for i, node in enumerate(self.travel_path):
            if i >= len(self.travel_path) - 1:
                break
            current_node = node
            next_node = self.travel_path[i + 1]
            dist = self.dist_matrix[current_node][next_node]
            wait_time = max(self.nodes[next_node].ready_time - (current_time + dist), 0)
            arrival_time = current_time + dist + wait_time
            current_time = arrival_time + self.nodes[next_node].service_time

        return current_time

    def check_time_constrain(self):
        # i need to calculate current time correctly
        # i need to calculate arrival time correctly
        # i need to calculate wait time

        # time meet along the road
        current_time = 0
        for i, node in enumerate(self.travel_path):
            if i >= len(self.travel_path) - 1:
                break
            current_node = node
            next_node = self.travel_path[i + 1]
            dist = self.dist_matrix[current_node][next_node]

            wait_time = max(self.nodes[next_node].ready_time - (current_time + dist), 0)
            arrival_time = current_time + dist + wait_time
            if arrival_time > self.nodes[next_node].due_time:
                return False

            current_time = arrival_time + self.nodes[next_node].service_time
            # ##print(f"arrival_time = {arrival_time}, current_time ={current_time}, wait_time = {wait_time},dist:{dist},next node due date:{self.nodes[next_node].due_time}, next_node ready_time = {self.nodes[next_node].ready_time}")

            # still want to check if the vehicle can return to the depot at the due time
        return True

    def delete_from_route(self, nodes_removed):
        # k = random.randint(0, int(len(self.travel_path)/2))
        # nodes_removed = random.sample(self.travel_path[1:-1],k=k)
        # ##print(k)
        self.travel_path = [x for x in self.travel_path if x not in nodes_removed]
        return self.travel_path

    def insert_into_route(self, index, node):
        self.travel_path.insert(index, node)

    def check_adding_node(self, index, node):
        temp_route = Route(self.id, self.data, self.travel_path)
        temp_route.insert_into_route(index, node)
        time_constrain = temp_route.check_time_constrain()
        capacity_constrain = temp_route.check_capacity_constrain()
        combined = time_constrain and capacity_constrain
        del temp_route
        return combined

    def cal_nearest_next_index(self, available_nodes):
        """
        Find the nearest reachable next index.
        :param index_to_visit:
        :return:
        """

        current_index = self.travel_path[-2]
        current_load = self.calculate_route_load()
        current_time = self.get_current_time()
        index_to_visit = [xy.id for xy in available_nodes]
        nearest_ind = None
        nearest_distance = None

        for next_index in index_to_visit:
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue

            dist = self.dist_matrix[current_index][next_index]
            wait_time = max(self.nodes[next_index].ready_time - current_time - dist, 0)
            service_time = self.nodes[next_index].service_time
            # Check whether it is possible to return to the service station after visiting a customer.
            if (
                current_time
                + dist
                + wait_time
                + service_time
                + self.dist_matrix[next_index][0]
                > self.nodes[0].due_time
            ):
                continue

            # Do not serve customers beyond their due time.
            if current_time + dist > self.nodes[next_index].due_time:
                continue

            if (
                nearest_distance is None
                or self.dist_matrix[current_index][next_index] < nearest_distance
            ):
                nearest_distance = self.dist_matrix[current_index][next_index]
                nearest_ind = next_index

        return nearest_ind

    def get_node_by_index(self, index):
        node = self.travel_path[index]
        return node

    def optimize_route(self):
        # Steps
        # 1. loop on every node on route
        # 2. change its location on the route and see if its better
        # 3. if route distance is minmized then its updated
        last_idx = len(self.travel_path) - 1
        for idx, node in enumerate(self.travel_path):
            print(f"starting with node:{node}")
            for new_idx in range(last_idx):
                if node == 0:
                    continue
                if idx == last_idx:
                    continue
                if new_idx == idx:
                    continue
                if new_idx == 0:
                    continue
                temp_route = Route(self.id, self.data, self.travel_path)
                # node deleted
                temp_route.delete_from_route([node])
                # insert into next idx
                constrains_status = temp_route.check_adding_node(new_idx, node)
                print(constrains_status)
                print(f"new travel path:{temp_route.travel_path}")
                if constrains_status:
                    print("success" * 10)
                    new_distance = temp_route.calculate_route_dist()
                    current_distance = self.calculate_route_dist()
                    if new_distance < current_distance:
                        print("Even Better Solution" * 50)
                        self.travel_path = temp_route.travel_path
                        self.load = self.calculate_route_load()


class Solution:
    def __init__(self, file_path, travel_path):
        if type(travel_path) == type(np.array([])):
            self.travel_path = list(travel_path)
        else:
            self.travel_path = travel_path

        self.file_path = file_path
        self.data = self.create_from_file(file_path)
        self.dist_matrix = self.data[2]
        self.vehicle_capacity = self.data[4]
        self.nodes = self.data[1]
        self.num_of_customers = self.data[0]
        self.lst_of_routes = []
        # total distance done
        self.create_routes()

    def create_from_file(self, file_path):
        # Read the positions of the depot and customers from the input file
        node_list = []
        with open(file_path, "rt") as f:
            count = 1
            for line in f:
                # Extract the number of vehicles and the capacity of each vehicle from line 5
                if count == 5:
                    vehicle_num, vehicle_capacity = line.split()
                    vehicle_num = int(vehicle_num)
                    vehicle_capacity = int(vehicle_capacity)

                # Extract the number of vehicles and the capacity of each vehicle from line 5
                elif count >= 10:
                    node_list.append(line.split())
                count += 1

        # Create a list of Node objects from the extracted information
        node_num = len(node_list)
        nodes = list(
            Node(
                int(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
                float(item[6]),
            )
            for item in node_list
        )

        # Create a distance matrix between all nodes
        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            # A node has zero distance to itself

            node_dist_mat[i][i] = 1e-8
            for j in range(i + 1, node_num):
                node_b = nodes[j]
                # Calculate the Euclidean distance between nodes a and b and store it in the distance matrix
                node_dist_mat[i][j] = Solution.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]

        instance = vrplib.read_instance(file_path, instance_format="solomon")
        node_dist_mat = instance["edge_weight"]
        return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    def get_route_by_node(self, node):
        for x in self.lst_of_routes:
            if node in x.travel_path:
                return x

    def get_candidates_with_similar_characteristics(self, node, d_range, t_range):
        # search for customers within the same time windows and similar demand
        # demand relaxation of + or - 10
        demand_min = node.demand - d_range
        demand_max = node.demand + d_range

        ready_time = node.ready_time - t_range
        due_time = node.due_time + t_range
        # print(f"time range: ready_time:{ready_time}, due_time:{due_time}")
        candidates = []
        for p_node in self.nodes:
            if p_node.id == node.id:
                continue
            # print(f"p_node ready time:{p_node.ready_time}, due time: {p_node.due_time}")
            if (
                p_node.demand < demand_max
                and p_node.ready_time > ready_time
                and p_node.due_time < due_time
            ):
                candidates.append(p_node)
        return candidates

    def calculate_total_distance(self):
        distance = 0
        current_ind = self.travel_path[0]
        for next_ind in self.travel_path[1:]:
            distance += self.dist_matrix[current_ind][next_ind]
            current_ind = next_ind
        return distance

    def update_routes(self):
        self.lst_of_routes = []
        self.create_routes()
        self.num_of_customers = len([x for x in self.travel_path if x != 0])

    def update_travel_path_from_routes(self):
        new_path = []
        for route in self.lst_of_routes:
            new_path += route.travel_path[:-1]

        new_path.append(0)
        self.travel_path = new_path
        self.num_of_customers = len([x for x in self.travel_path if x != 0])

    def create_routes(self):
        result = []
        temp = []
        for i in self.travel_path:
            if i == 0:
                result.append(temp)
                temp = []
            else:
                temp.append(i)
        result.append(temp)
        result = result[1:-1]

        for lst in result:
            lst.insert(0, 0)
            lst.append(0)

        c = 0
        for path in result:
            route = Route(c, self.data, path)
            self.lst_of_routes.append(route)
            c += 1
        return result

    def create_alternative_routes(self, travel_path):
        if type(travel_path) == type(np.array([])):
            travel_path = list(travel_path)
        new_routes = []
        result = []
        temp = []
        for i in travel_path:
            if i == 0:
                result.append(temp)
                temp = []
            else:
                temp.append(i)
        result.append(temp)
        result = result[1:-1]

        for lst in result:
            lst.insert(0, 0)
            lst.append(0)

        c = 0
        for path in result:
            route = Route(c, self.data, path)
            new_routes.append(route)
            c += 1
        return new_routes

    def check_constrains(self):
        for route in self.lst_of_routes:
            capacity_constrain = route.check_capacity_constrain()
            time_constrain = route.check_time_constrain()
            if not capacity_constrain:
                return False
            if not time_constrain:
                return False
        return True

    def show_path(self):
        return np.array(self.travel_path)

    def update_travel_path_from_new_travel_path(self, new_travel_path):
        self.travel_path = copy.deepcopy(list(new_travel_path))
        self.update_routes()

    def get_path_weights(self):
        travel_path_array = np.array(self.travel_path)

    def calculate_weights(self):
        saving_matrix = []
        for route in self.lst_of_routes:
            for node in route.travel_path:
                temp_sol = Solution(self.file_path, self.travel_path)
                if node != 0:
                    temp_sol.lst_of_routes[route.id].delete_from_route([node])
                    temp_sol.update_travel_path_from_routes()
                    new_distance = temp_sol.calculate_total_distance()
                    # print(new_distance)
                    saving = self.calculate_total_distance() - new_distance
                    saving_matrix.append([node, saving, new_distance])
        saving_matrix.sort(key=lambda x: x[1], reverse=True)
        return saving_matrix

    def destory_operator(self, n_size):
        removed_nodes = []
        customers = np.array([x for x in range(0, len(self.nodes))])

        removed_nodes = self.calculate_weights()[:n_size]
        # for _ in range(n_size):
        #     removed_node = random.randint(1, 100)
        #     removed_nodes.append(removed_node)
        new_travel_path = np.array(self.travel_path)
        new_travel_path = new_travel_path[
            ~np.isin(new_travel_path, np.array(removed_nodes))
        ]
        self.update_travel_path_from_new_travel_path(new_travel_path)
        return removed_nodes

    def get_customers_not_allocated(self):
        # assuming that its always 100 customers
        customers = np.array([x for x in range(0, len(self.nodes))])
        travel_path = np.array(copy.deepcopy(self.travel_path))
        customers_not_allocated = customers[~np.isin(customers, travel_path)]

        return customers_not_allocated

    def copy_from_solution(self, solution):
        new_travel_path = solution.travel_path
        self.update_travel_path_from_new_travel_path(new_travel_path)

    def apply_lns(self, n_size):
        self.distances_found = []
        # 0. get the current solution distance
        current_solution_distance = self.calculate_total_distance()
        current_solution = Solution(self.file_path, self.travel_path)
        # 1. destroy customers
        self.destory_operator(n_size)
        # print(f"customers to be allocated before starting lns:{self.get_customers_not_allocated()}")
        best_solution_found = False
        counter = 0
        while len(self.get_customers_not_allocated()) != 0 and not best_solution_found:
            # 2. get customers not allocated
            nodes_not_allocated = self.get_customers_not_allocated()
            random.shuffle(nodes_not_allocated)
            # print(f"Nodes that are not allocated:{nodes_not_allocated}")

            # loop on nodes not allocated
            for node in nodes_not_allocated.copy():
                # loop on routes
                break_loop = False
                # print("*"*50)
                best_position = []

                for route in self.lst_of_routes:
                    # print("-"*50)
                    # print(f"------------ Route ID:{route.id} -----------")
                    current_route_distance = current_solution.lst_of_routes[
                        route.id
                    ].calculate_route_dist()
                    # loop on every position in routes
                    for i in range(1, len(route.travel_path)):
                        new_route = copy.deepcopy(route)
                        # add node in position i
                        new_route.insert_into_route(i, node)
                        # print(f"node:{node}, new route path:{new_route.travel_path}")

                        # check if new route constrains are True
                        if (
                            new_route.check_time_constrain()
                            and new_route.check_capacity_constrain()
                        ):
                            self.update_travel_path_from_routes()

                            # print(f"time constrain:{new_route.check_time_constrain()} capacity constrain:{new_route.check_capacity_constrain()}")
                            # check if new route distance is less than the current route distance
                            new_route_distance = new_route.calculate_route_dist()
                            # print(f"new route distance:{new_route_distance}, current route distance:{current_route_distance}")
                            # if int(new_route_distance)<= int(current_route_distance): # must check if the overall distance is increased or decreased
                            # print(f"updated new solution distance:{self.calculate_total_distance()}")

                            distance_saved = current_route_distance - new_route_distance
                            best_position.append([distance_saved, route.id, i])

                            # route.insert_into_route(i,node)
                            # # node allocated i need to break its loop
                            # self.update_travel_path_from_routes()
                            # ##print(f"Still nodes:{self.get_customers_not_allocated()} to be allocated")
                            # break_loop = True
                            # break
                    # if break_loop == True:
                    #     break

                best_saved = -50000
                best_position_value = None
                best_route_id = None
                for answer in best_position:
                    distance_saved = answer[0]
                    route_id = answer[1]
                    idx = answer[2]

                    # print(f"Distance saved:{distance_saved},route id:{route_id}, position:{idx}")
                    # print(f"distance saved:{distance_saved} compared to best saved {best_saved}, = {distance_saved>best_saved}")
                    if distance_saved > best_saved:
                        best_saved = distance_saved
                        best_position_value = idx
                        best_route_id = route_id
                    else:
                        # print("some error!")
                        pass

                if best_route_id:
                    self.lst_of_routes[best_route_id].insert_into_route(
                        best_position_value, node
                    )
                    # print(f"Node:{node} has been allocated in route:{best_route_id} it saved:{best_saved}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()
            # print(f"new solution distance = {new_solution_distance}")
            # print(f"new travel path{self.show_path()}")
            # print(f"routes:")
            # for route in self.lst_of_routes:
            #     ##print(f"{route.travel_path}")
            #     pass
            # print(f"new solution distance before tabu search = {new_solution_distance}")
            self.apply_tabu_search(100, 2, 10)
            # print(f"new solution distance after tabu search = {self.calculate_total_distance()}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()

            if (
                int(new_solution_distance) >= int(current_solution_distance)
                or len(self.get_customers_not_allocated()) > 0
            ):
                self.copy_from_solution(current_solution)
                print("No better solution found")
            else:
                best_solution_found = True
                # print(f"Better Solution has been found(new_distance:{new_solution_distance}, old solution distance:{current_solution_distance})")
                # print(f"customers not allocated:{self.get_customers_not_allocated()}")
                self.distances_found.append(new_solution_distance)

            # print(f"nodes are not allocated increased to:{self.get_customers_not_allocated()}")
            # print("there's customers that are not allocated")

            counter += 1
            if len(self.get_customers_not_allocated()) > 0:
                if counter % 10 == 0:
                    n_size = int(n_size / counter) * 3
                    # print(n_size)
                    self.destory_operator(n_size)

                if n_size <= 2:
                    if (
                        int(new_solution_distance) >= int(current_solution_distance)
                        or len(self.get_customers_not_allocated()) > 0
                    ):
                        self.copy_from_solution(current_solution)
                        print("No better solution found")
                    return self.distances_found
                    # print("breaking for n_size is smaller than or equal 2")
                    break
        return self.distances_found

    def apply_tabu_search(self, iterations, tabu_iterations, tabu_limit):
        lst = np.array(self.travel_path)
        counter = 0
        tabu_lst = [0]

        while True:
            counter += 1
            if counter >= iterations:
                break
            current_lst = copy.deepcopy(lst)
            for i in range(tabu_iterations):
                # print(f"tabu list:{tabu_lst}")
                c1 = random.choice(current_lst)
                c2 = random.choice(current_lst)
                # print(f"C1:{c1}, C2:{c2}")
                if i % tabu_limit == 0:
                    tabu_lst = [0]

                if c1 in tabu_lst:
                    continue

                if c2 in tabu_lst:
                    continue

                if c1 == c2:
                    continue

                tabu_lst.append(c1)
                tabu_lst.append(c2)

                current_lst[np.where(lst == c1)] = c2
                current_lst[np.where(lst == c2)] = c1

            current_solution = Solution(self.file_path, self.travel_path)
            new_solution = Solution(self.file_path, current_lst)
            new_distance = new_solution.calculate_total_distance()
            if (
                new_distance < current_solution.calculate_total_distance()
                and new_solution.check_constrains()
            ):
                self.copy_from_solution(new_solution)
                return True
            else:
                pass

        return False

    def apply_branch_and_bound(self):
        possible_solution = []
        path = []
        choosed_nodes = [self.nodes[0]]
        route_path = [0]
        new_route = Route(len(path) + 1, self.data, [0, 0])
        for i in range(len(self.nodes)):
            space = [node for node in self.nodes if node not in choosed_nodes]
            if len(space) > 0:
                node_id = new_route.cal_nearest_next_index(space)
                temp_route = copy.deepcopy(new_route)
                # print(f"node id :{node_id}")
                if node_id is not None:
                    if temp_route.check_adding_node(-1, node_id):
                        # print(f"node:{node_id}")
                        # print(f"route path before adding it {new_route.travel_path}")
                        new_route.insert_into_route(-1, node_id)
                        # print(f"route path after adding it {new_route.travel_path}")

                        choosed_nodes.append(self.nodes[node_id])
                    else:
                        path.append(new_route)
                        new_route = Route(len(path) + 1, self.data, [0, 0])
                else:
                    path.append(new_route)
                    new_route = Route(len(path) + 1, self.data, [0, 0])

        customers = []
        for route in path:
            # print(f"type route:{type(route)}")
            customers += route.travel_path[:-1]
        customers_fullfilled = np.array(customers + [0])
        return customers_fullfilled

    def fulfill_travel_path(self, n_size):
        self.distances_found = []
        # 0. get the current solution distance
        current_solution_distance = self.calculate_total_distance()
        current_solution = Solution(self.file_path, self.travel_path)
        # 1. destroy customers
        self.destory_operator(n_size)
        # print(f"customers to be allocated before starting lns:{self.get_customers_not_allocated()}")
        best_solution_found = False
        counter = 0
        while len(self.get_customers_not_allocated()) != 0 and not best_solution_found:
            # 2. get customers not allocated
            self.update_travel_path_from_routes()
            # print(f"num of customers:{self.num_of_customers}")
            nodes_not_allocated = self.get_customers_not_allocated()
            # print(f"len nodes not allocated:{len(nodes_not_allocated)}")

            # loop on nodes not allocated
            for node in nodes_not_allocated.copy():
                # loop on routes
                break_loop = False
                # print("*"*50)
                best_position = []

                for route in self.lst_of_routes:
                    # print("-"*50)
                    # print(f"------------ Route ID:{route.id} -----------")
                    current_route_distance = current_solution.lst_of_routes[
                        route.id
                    ].calculate_route_dist()
                    # loop on every position in routes
                    for i in range(1, len(route.travel_path)):
                        new_route = copy.deepcopy(route)
                        # add node in position i
                        new_route.insert_into_route(i, node)
                        # print(f"node:{node}, new route path:{new_route.travel_path}")

                        # check if new route constrains are True
                        if (
                            new_route.check_time_constrain()
                            and new_route.check_capacity_constrain()
                        ):
                            self.update_travel_path_from_routes()

                            # print(f"time constrain:{new_route.check_time_constrain()} capacity constrain:{new_route.check_capacity_constrain()}")
                            # check if new route distance is less than the current route distance
                            new_route_distance = new_route.calculate_route_dist()
                            # print(f"new route distance:{new_route_distance}, current route distance:{current_route_distance}")
                            # if int(new_route_distance)<= int(current_route_distance): # must check if the overall distance is increased or decreased
                            # print(f"updated new solution distance:{self.calculate_total_distance()}")

                            distance_saved = current_route_distance - new_route_distance
                            best_position.append([distance_saved, route.id, i])

                            # route.insert_into_route(i,node)
                            # # node allocated i need to break its loop
                            # self.update_travel_path_from_routes()
                            # ##print(f"Still nodes:{self.get_customers_not_allocated()} to be allocated")
                            # break_loop = True
                            # break
                    # if break_loop == True:
                    #     break

                best_saved = -50000
                best_position_value = None
                best_route_id = None
                for answer in best_position:
                    distance_saved = answer[0]
                    route_id = answer[1]
                    idx = answer[2]

                    # print(f"Distance saved:{distance_saved},route id:{route_id}, position:{idx}")
                    # print(f"distance saved:{distance_saved} compared to best saved {best_saved}, = {distance_saved>best_saved}")
                    if distance_saved > best_saved:
                        best_saved = distance_saved
                        best_position_value = idx
                        best_route_id = route_id
                    else:
                        # print("some error!")
                        pass

                if best_route_id:
                    self.lst_of_routes[best_route_id].insert_into_route(
                        best_position_value, node
                    )
                    # print(f"Node:{node} has been allocated in route:{best_route_id} it saved:{best_saved}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()
            # print(f"new solution distance = {new_solution_distance}")
            # print(f"new travel path{self.show_path()}")
            # print(f"routes:")
            # for route in self.lst_of_routes:
            #     ##print(f"{route.travel_path}")
            #     pass
            # print(f"new solution distance before tabu search = {new_solution_distance}")
            self.apply_tabu_search(100, 10, 10)
            # print(f"new solution distance after tabu search = {self.calculate_total_distance()}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()

            if (
                int(new_solution_distance) >= int(current_solution_distance)
                or len(self.get_customers_not_allocated()) > 0
            ):
                self.copy_from_solution(current_solution)
                # print("No better solution found")
            else:
                best_solution_found = True
                # print(f"Better Solution has been found(new_distance:{new_solution_distance}, old solution distance:{current_solution_distance})")
                # print(f"customers not allocated:{self.get_customers_not_allocated()}")
                self.distances_found.append(new_solution_distance)

            # print(f"nodes are not allocated increased to:{self.get_customers_not_allocated()}")
            # print("there's customers that are not allocated")

            counter += 1
            if len(self.get_customers_not_allocated()) > 0:
                if counter % 10 == 0:
                    n_size = int(n_size / counter) * 3
                    # print(n_size)
                    self.destory_operator(n_size)

                if counter > 10:
                    if (
                        int(new_solution_distance) >= int(current_solution_distance)
                        or len(self.get_customers_not_allocated()) > 0
                    ):
                        self.copy_from_solution(current_solution)
                        # print("No better solution found")
                    return self.distances_found
                    # print("breaking for n_size is smaller than or equal 2")
                    break
        return self.distances_found

    def modified_lns(self, n_size):
        self.distances_found = []
        # 0. get the current solution distance
        current_solution_distance = self.calculate_total_distance()
        current_solution = Solution(self.file_path, self.travel_path)
        # 1. destroy customers
        self.destory_operator(n_size)
        # print(f"customers to be allocated before starting lns:{self.get_customers_not_allocated()}")
        best_solution_found = False
        counter = 0
        while len(self.get_customers_not_allocated()) != 0 and not best_solution_found:
            # 2. get customers not allocated
            nodes_not_allocated = self.get_customers_not_allocated()
            random.shuffle(nodes_not_allocated)
            print(f"Nodes that are not allocated:{nodes_not_allocated}")

            # loop on nodes not allocated
            for node in nodes_not_allocated.copy():
                # loop on routes
                break_loop = False
                # print("*"*50)
                best_position = []

                for route in self.lst_of_routes:
                    # print("-"*50)
                    # print(f"------------ Route ID:{route.id} -----------")
                    current_route_distance = current_solution.lst_of_routes[
                        route.id
                    ].calculate_route_dist()
                    # loop on every position in routes
                    for i in range(1, len(route.travel_path)):
                        new_route = copy.deepcopy(route)
                        # add node in position i
                        current_node_at_i = self.lst_of_routes[
                            new_route.id
                        ].get_node_by_index(i)

                        new_route.delete_from_route([current_node_at_i])
                        new_route.insert_into_route(i, node)
                        # print(f"node:{node}, new route path:{new_route.travel_path}")

                        # check if new route constrains are True
                        if (
                            new_route.check_time_constrain()
                            and new_route.check_capacity_constrain()
                        ):
                            self.update_travel_path_from_routes()

                            # print(f"time constrain:{new_route.check_time_constrain()} capacity constrain:{new_route.check_capacity_constrain()}")
                            # check if new route distance is less than the current route distance
                            new_route_distance = new_route.calculate_route_dist()
                            # print(f"new route distance:{new_route_distance}, current route distance:{current_route_distance}")
                            # if int(new_route_distance)<= int(current_route_distance): # must check if the overall distance is increased or decreased
                            # print(f"updated new solution distance:{self.calculate_total_distance()}")

                            distance_saved = current_route_distance - new_route_distance
                            best_position.append([distance_saved, route.id, i])

                            # route.insert_into_route(i,node)
                            # # node allocated i need to break its loop
                            # self.update_travel_path_from_routes()
                            # ##print(f"Still nodes:{self.get_customers_not_allocated()} to be allocated")
                            # break_loop = True
                            # break
                    # if break_loop == True:
                    #     break

                best_saved = -50000
                best_position_value = None
                best_route_id = None
                for answer in best_position:
                    distance_saved = answer[0]
                    route_id = answer[1]
                    idx = answer[2]

                    # print(f"Distance saved:{distance_saved},route id:{route_id}, position:{idx}")
                    # print(f"distance saved:{distance_saved} compared to best saved {best_saved}, = {distance_saved>best_saved}")
                    if distance_saved > best_saved:
                        best_saved = distance_saved
                        best_position_value = idx
                        best_route_id = route_id
                    else:
                        # print("some error!")
                        pass

                if best_route_id:
                    self.lst_of_routes[best_route_id].delete_from_route(
                        [best_position_value]
                    )

                    self.lst_of_routes[best_route_id].insert_into_route(
                        best_position_value, node
                    )
                    print(
                        f"Node:{node} has been allocated in route:{best_route_id} it saved:{best_saved}"
                    )

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()
            # self.apply_lns(0)
            print(f"new solution distance = {new_solution_distance}")
            # print(f"new travel path{self.show_path()}")
            # print(f"routes:")
            # for route in self.lst_of_routes:
            #     ##print(f"{route.travel_path}")
            #     pass
            # print(f"new solution distance before tabu search = {new_solution_distance}")
            # self.apply_tabu_search(100, 2, 10)
            # print(f"new solution distance after tabu search = {self.calculate_total_distance()}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()

            if (
                int(new_solution_distance) >= int(current_solution_distance)
                or len(self.get_customers_not_allocated()) > 0
            ):
                self.copy_from_solution(current_solution)
                # print("No better solution found")
            else:
                best_solution_found = True
                # print(f"Better Solution has been found(new_distance:{new_solution_distance}, old solution distance:{current_solution_distance})")
                # print(f"customers not allocated:{self.get_customers_not_allocated()}")
                self.distances_found.append(new_solution_distance)

            # print(f"nodes are not allocated increased to:{self.get_customers_not_allocated()}")
            # print("there's customers that are not allocated")

            counter += 1
            if len(self.get_customers_not_allocated()) > 0:
                if counter % 10 == 0:
                    n_size = int(n_size / counter) * 3
                    # print(n_size)
                    self.destory_operator(n_size)

                if n_size <= 2:
                    if (
                        int(new_solution_distance) >= int(current_solution_distance)
                        or len(self.get_customers_not_allocated()) > 0
                    ):
                        self.copy_from_solution(current_solution)
                        # print("No better solution found")
                    return self.distances_found
                    # print("breaking for n_size is smaller than or equal 2")
                    break
        return self.distances_found

    def fit_un_allocated_customers(self):
        n_size = 0
        self.distances_found = []
        # 0. get the current solution distance
        current_solution_distance = self.calculate_total_distance()
        current_solution = Solution(self.file_path, self.travel_path)
        # 1. destroy customers
        self.destory_operator(n_size)
        # print(f"customers to be allocated before starting lns:{self.get_customers_not_allocated()}")
        best_solution_found = False
        counter = 0
        while len(self.get_customers_not_allocated()) != 0 and not best_solution_found:
            # 2. get customers not allocated
            nodes_not_allocated = self.get_customers_not_allocated()
            random.shuffle(nodes_not_allocated)
            # print(f"Nodes that are not allocated:{nodes_not_allocated}")

            # loop on nodes not allocated
            for node in nodes_not_allocated.copy():
                # loop on routes
                break_loop = False
                # print("*"*50)
                best_position = []

                for route in self.lst_of_routes:
                    # print("-"*50)
                    # print(f"------------ Route ID:{route.id} -----------")
                    current_route_distance = current_solution.lst_of_routes[
                        route.id
                    ].calculate_route_dist()
                    # loop on every position in routes
                    for i in range(1, len(route.travel_path)):
                        new_route = copy.deepcopy(route)
                        # add node in position i
                        new_route.insert_into_route(i, node)
                        # print(f"node:{node}, new route path:{new_route.travel_path}")

                        # check if new route constrains are True
                        if (
                            new_route.check_time_constrain()
                            and new_route.check_capacity_constrain()
                        ):
                            self.update_travel_path_from_routes()

                            # print(f"time constrain:{new_route.check_time_constrain()} capacity constrain:{new_route.check_capacity_constrain()}")
                            # check if new route distance is less than the current route distance
                            new_route_distance = new_route.calculate_route_dist()
                            # print(f"new route distance:{new_route_distance}, current route distance:{current_route_distance}")
                            # if int(new_route_distance)<= int(current_route_distance): # must check if the overall distance is increased or decreased
                            # print(f"updated new solution distance:{self.calculate_total_distance()}")

                            distance_saved = current_route_distance - new_route_distance
                            best_position.append([distance_saved, route.id, i])

                            # route.insert_into_route(i,node)
                            # # node allocated i need to break its loop
                            # self.update_travel_path_from_routes()
                            # ##print(f"Still nodes:{self.get_customers_not_allocated()} to be allocated")
                            # break_loop = True
                            # break
                    # if break_loop == True:
                    #     break

                best_saved = -50000
                best_position_value = None
                best_route_id = None
                for answer in best_position:
                    distance_saved = answer[0]
                    route_id = answer[1]
                    idx = answer[2]

                    # print(f"Distance saved:{distance_saved},route id:{route_id}, position:{idx}")
                    # print(f"distance saved:{distance_saved} compared to best saved {best_saved}, = {distance_saved>best_saved}")
                    if distance_saved > best_saved:
                        best_saved = distance_saved
                        best_position_value = idx
                        best_route_id = route_id
                    else:
                        # print("some error!")
                        pass

                if best_route_id:
                    self.lst_of_routes[best_route_id].insert_into_route(
                        best_position_value, node
                    )
                    # print(f"Node:{node} has been allocated in route:{best_route_id} it saved:{best_saved}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()
            # print(f"new solution distance = {new_solution_distance}")
            # print(f"new travel path{self.show_path()}")
            # print(f"routes:")
            # for route in self.lst_of_routes:
            #     ##print(f"{route.travel_path}")
            #     pass
            # print(f"new solution distance before tabu search = {new_solution_distance}")
            self.apply_tabu_search(100, 2, 10)
            # print(f"new solution distance after tabu search = {self.calculate_total_distance()}")

            self.update_travel_path_from_routes()
            new_solution_distance = self.calculate_total_distance()

            if len(self.get_customers_not_allocated()) > 0:
                self.copy_from_solution(current_solution)
                # print("No better solution found")
            else:
                best_solution_found = True
                # print(f"Better Solution has been found(new_distance:{new_solution_distance}, old solution distance:{current_solution_distance})")
                # print(f"customers not allocated:{self.get_customers_not_allocated()}")
                self.distances_found.append(new_solution_distance)

            # print(f"nodes are not allocated increased to:{self.get_customers_not_allocated()}")
            # print("there's customers that are not allocated")

            counter += 1
            if len(self.get_customers_not_allocated()) > 0:
                if counter % 10 == 0:
                    n_size = int(n_size / counter) * 3
                    # print(n_size)
                    self.destory_operator(n_size)

                if n_size <= 2:
                    if len(self.get_customers_not_allocated()) > 0:
                        self.copy_from_solution(current_solution)
                        # print("No better solution found")
                    return self.distances_found
                    # print("breaking for n_size is smaller than or equal 2")
                    break
        return self.distances_found
