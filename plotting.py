import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue

import random


class plot_solution:
    def __init__(self, solution):
        self.nodes = solution.nodes
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self._depot_color = "k"
        self._customer_color = "steelblue"
        self._line_color = "darksalmon"
        self.lines = []
        self.solution = solution

    def _draw_point(self):
        # 画出depot
        self.figure_ax.scatter(
            [self.nodes[0].x],
            [self.nodes[0].y],
            c=self._depot_color,
            label="depot",
            s=40,
        )

        # 画出customer
        self.figure_ax.scatter(
            list(node.x for node in self.nodes[1:]),
            list(node.y for node in self.nodes[1:]),
            c=self._customer_color,
            label="customer",
            s=20,
        )
        plt.pause(0.05)

    def run(self):
        self._draw_point()
        self.figure.show()
        
        path = self.solution.travel_path
        distance= self.solution.calculate_total_distance()
        used_vehicle_num = len(self.solution.lst_of_routes)

        self.distance = distance
        self.used_vehicle_num = used_vehicle_num
        self.figure_ax.clear()
        self._draw_point()
        self.figure_ax.set_title(
            "travel distance: %0.2f, number of vehicles: %d "
                    % (distance, used_vehicle_num)
                )
        self._draw_line(path)
        plt.pause(0.05)

    def generate_random_colors(self, num_colors):
        colors = []
        for i in range(num_colors):
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(color)
        return colors

    def _draw_line(self, path):
        line_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]

            # p1 = (self.nodes[path[i]].x,self.nodes[path[i]].y)
            p2 = (self.nodes[path[i - 1]].x, self.nodes[path[i - 1]].y)

            if p2[0] == self.nodes[0].x and p2[1] == self.nodes[0].y:
                # if color_idx== len(colors)-1:
                #     color_idx =0
                line_color = (
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                )
            line_drawn = self.figure_ax.plot(
                x_list, y_list, color=line_color, linewidth=1, label="line"
            )
            self.lines.append(line_drawn)
            plt.pause(0.01)

        plot_name = "travel distance- %0.2f - , number of vehicles- %d " % (
            self.distance,
            self.used_vehicle_num,
        )
        plt.savefig(f"solutions/{plot_name}.png")

    def close(self):
        plt.close(fig=None)