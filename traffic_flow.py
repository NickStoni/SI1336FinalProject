import constants
import matplotlib.pyplot as plt
import numpy.random as rng
import numpy as np
import random
import seaborn as sns

from scipy.stats import sem

class Observables:
    """ Class for storing observables """

    def __init__(self):
        self.time = []  # list to store time
        self.flowrate = []  # list to store the flow rate

class Road:
    """ Class for the state of a number of cars """

    def __init__(self, density, road_length, mu, sigma=1, num_lanes=1):
        self.road_length = road_length
        self.num_lanes = num_lanes
        self.num_cars = np.round(density * road_length * num_lanes)
        self.t = 0
        self.road = [[[-1]*2] * road_length for _ in range(num_lanes)]
        counter = 0
        while counter < self.num_cars:
            lane = random.randrange(0, num_lanes)
            x = random.randrange(0, road_length)
            if self.road[lane][x][0] == -1:
                v_max = np.random.randint(2, 15) if mu == -1 else np.random.normal(mu, sigma)
                v_max = int(np.round(v_max))
                self.road[lane][x] = [0, v_max if v_max > 0 else 1]
                counter += 1

    def distance(self, direction, car_x, lane_to_check, limit):
        offset = 0
        while offset < limit and self.road[lane_to_check][(car_x + direction * offset) % self.road_length][0] == -1:
            offset += 1
        return offset


class Propagator:
    def __init__(self, lane_change_model):
        self.lane_change_model = lane_change_model
        self.p_s = constants.p_s

    def is_lane_available(self, cars, lane, x, v, v_max):
        d_ahead = cars.distance(1, x, lane, v_max + 2)
        d_behind = cars.distance(-1, x, lane, v_max + 2)
        return (d_ahead >= v + 2) and (d_behind >= v_max + 2)

    def update_lanes(self, cars):
        if cars.num_lanes == 1:
            return

        road_updated = [[[-1]*2] * cars.road_length for _ in range(cars.num_lanes)]
        for lane in range(cars.num_lanes):
            for x in range(cars.road_length):
                if cars.road[lane][x][0] != -1:
                    d = cars.distance(1, x + 1, lane, cars.road[lane][x][1] + 2) + 1
                    left_lane = lane + 1
                    right_lane = lane - 1

                    if self.lane_change_model == "EU":
                        if right_lane >= 0 and self.is_lane_available(cars, right_lane, x, cars.road[lane][x][0], cars.road[lane][x][1]):
                            road_updated[right_lane][x] = cars.road[lane][x]
                        elif d < cars.road[lane][x][0] + 2 and left_lane < cars.num_lanes \
                                and self.is_lane_available(cars, left_lane, x, cars.road[lane][x][0], cars.road[lane][x][1]):
                            road_updated[left_lane][x] = cars.road[lane][x]
                        else:
                            road_updated[lane][x] = cars.road[lane][x]

                    elif self.lane_change_model == "NA":
                        if d < cars.road[lane][x][0] + 2:
                            random_lanes = [right_lane, left_lane]
                            random.shuffle(random_lanes)
                            if 0 <= random_lanes[0] < cars.num_lanes \
                                    and self.is_lane_available(cars, random_lanes[0], x, cars.road[lane][x][0], cars.road[lane][x][1]):
                                road_updated[random_lanes[0]][x] = cars.road[lane][x]
                            elif 0 <= random_lanes[1] < cars.num_lanes \
                                    and self.is_lane_available(cars, random_lanes[1], x, cars.road[lane][x][0], cars.road[lane][x][1]):
                                road_updated[random_lanes[1]][x] = cars.road[lane][x]
                            else:
                                road_updated[lane][x] = cars.road[lane][x]
                        else:
                            road_updated[lane][x] = cars.road[lane][x]
                    else:
                        raise "The lane change model chosen is incorrect. Should be one of: NA, EU."

        cars.road = road_updated

    def update_speed(self, cars):
        for lane in range(cars.num_lanes):
            for x in range(cars.road_length):
                v_max = cars.road[lane][x][1]
                if cars.road[lane][x][0] != -1:
                    d = cars.distance(1, x + 1, lane, v_max + 2) + 1

                    # Rule 1: If v < v_max, increase v by 1 unit
                    if cars.road[lane][x][0] < v_max:
                        cars.road[lane][x][0] += 1

                    # Rule 2: If dist, d, to next car is <= v_i, reduce v_i to d - 1
                    if d <= cars.road[lane][x][0]:
                        cars.road[lane][x][0] = d - 1

                    # Rule 3: Reduce v_i by 1 unit, with probability p_s, given v_i>0
                    if cars.road[lane][x][0] > 0 and np.random.rand() < self.p_s:
                        cars.road[lane][x][0] -= 1

    def update_position(self, cars):
        sum_velocities = 0
        road_updated = [[[-1]*2] * cars.road_length for _ in range(cars.num_lanes)]
        for lane in range(cars.num_lanes):
            for x in range(cars.road_length):
                if cars.road[lane][x][0] != -1:
                    x_next = (x + cars.road[lane][x][0]) % cars.road_length
                    road_updated[lane][x_next] = cars.road[lane][x]
                    sum_velocities += cars.road[lane][x][0]
        cars.road = road_updated
        return sum_velocities

    def timestep(self, cars):
        self.update_lanes(cars)
        self.update_speed(cars)
        sum_velocities = self.update_position(cars)

        cars.t += 1

        return sum_velocities / (cars.road_length * cars.num_lanes)  # return the flow for the entire highway

    def propagate(self, cars, obs):
        """ Perform a single integration step """

        fr = self.timestep(cars)

        # Append observables to their lists
        obs.time.append(cars.t)
        obs.flowrate.append(fr)

############################################################################################
class Simulation:

    def reset(self, cars):
        self.cars = cars
        self.obs = Observables()

    def __init__(self, cars):
        self.reset(cars)

    def plot_observables(self, title="simulation"):
        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.flowrate)
        plt.xlabel('time')
        plt.ylabel('flow rate')
        plt.savefig(title + ".pdf")
        plt.show()

    def display(self, cars):
        plt.figure(figsize=(10, 2.5 + (2.55 * cars.num_lanes)))
        sns.heatmap(np.where(cars.road == -1, np.nan, cars.road),
                    square=True, vmin=0, vmax=5,
                    cbar=True, xticklabels=False, yticklabels=False,
                    alpha=.75, linewidths=0.75, linecolor="black")
        plt.show()
    # Run without displaying any animation (fast)
    def run(self,
            propagator,
            numsteps=200,  # final time
            title="simulation",  # Name of output file and title shown at the top
            ):
        cars_pos_history = []

        for it in range(numsteps):
            propagator.propagate(self.cars, self.obs)

        # self.plot_observables(title)
        return cars_pos_history

    # Run while displaying the animation of bunch of cars going in circe (slow-ish)
    def run_animate(self,
                    propagator,
                    numsteps=200,  # Final time
                    stepsperframe=1,  # How many integration steps between visualising frames
                    title="simulation",  # Name of output file and title shown at the top
                    ):
        self.plot_observables(title)


def std_vs_n():
    density = 0.3
    road_length = 100
    v_max = [5, 10]
    for v in v_max:
        print(v)
        temp_flow_rate = []
        std = []
        for n in range(1, 40):
            print(n)
            road = Road(density=density, road_length=road_length, num_lanes=3, mu=v, sigma=1)
            simulation = Simulation(road)
            simulation.run(propagator=Propagator(lane_change_model="EU"), numsteps=500)
            temp_flow_rate.append(np.mean(simulation.obs.flowrate[-100:]))
            if len(temp_flow_rate) > 1:
                std.append(sem(temp_flow_rate))
        plt.plot(range(2, 40), std, label=f"EU model, Normal dist: $\mu$ = {v}, $\sigma$ = {1}")

    for v in v_max:
        print(v)
        temp_flow_rate = []
        std = []
        for n in range(1, 40):
            print(n)
            road = Road(density=density, road_length=road_length, num_lanes=3, mu=v, sigma=1)
            simulation = Simulation(road)
            simulation.run(propagator=Propagator(lane_change_model="NA"), numsteps=500)
            temp_flow_rate.append(np.mean(simulation.obs.flowrate[-100:]))
            if len(temp_flow_rate) > 1:
                std.append(sem(temp_flow_rate))
        plt.plot(range(2, 40), std, label=f"NA model, Normal dist: $\mu$ = {v}, $\sigma$ = {1}")

    temp_flow_rate = []
    std = []
    for n in range(1, 40):
        print(n)
        road = Road(density=density, road_length=road_length, num_lanes=3, mu=-1, sigma=1)
        simulation = Simulation(road)
        simulation.run(propagator=Propagator(lane_change_model="EU"), numsteps=500)
        temp_flow_rate.append(np.mean(simulation.obs.flowrate[-100:]))
        if len(temp_flow_rate) > 1:
            std.append(sem(temp_flow_rate))
    plt.plot(range(2, 40), std, label=f"EU model, Uniform dist: [2, 15]")

    temp_flow_rate = []
    std = []
    for n in range(1, 40):
        print(n)
        road = Road(density=density, road_length=road_length, num_lanes=3, mu=-1, sigma=1)
        simulation = Simulation(road)
        simulation.run(propagator=Propagator(lane_change_model="NA"), numsteps=500)
        temp_flow_rate.append(np.mean(simulation.obs.flowrate[-100:]))
        if len(temp_flow_rate) > 1:
            std.append(sem(temp_flow_rate))
    plt.plot(range(2, 40), std, label=f"NA model, Uniform dist: [2, 15]")

    plt.xlabel("Number of runs, N")
    plt.ylabel("Standard error in flow rate")
    plt.legend()
    plt.show()


def flow_rate_vs_t():
    density = 0.3
    road_length = 100
    v_max = [2, 5, 10]
    for v in v_max:
        road = Road(density=density, road_length=road_length, num_lanes=3, mu=v, sigma=1)
        simulation = Simulation(road)
        simulation.run(propagator=Propagator(lane_change_model="EU"), numsteps=500)
        plt.plot(range(500), simulation.obs.flowrate, label=f"vmax = {v}")
    plt.xlabel("Time, ticks")
    plt.ylabel("Flow rate, sum velocities/total road length")
    plt.legend()
    plt.show()

def fundamental_diag_vs_road_length():
    road_length = np.linspace(10, 200, 5).astype(int)
    density = np.linspace(0, 1, 10)

    for l in road_length:
        print(l)
        flow_rate = []

        for d in density:
            print(d)
            flow_rate_temp = []
            for _ in range(5):
                road = Road(density=d, road_length=l, num_lanes=3, mu=10, sigma=4)
                simulation = Simulation(road)
                simulation.run(propagator=Propagator(lane_change_model="EU"), numsteps=200)
                flow_rate_temp.append(np.mean(simulation.obs.flowrate[-100:]))
            final = (np.mean(flow_rate_temp))
            flow_rate.append(final)
        plt.plot(density, flow_rate, label=f"road length = {l}")
    plt.xlabel("Density, cars / road length")
    plt.ylabel("Flow rate, sum velocities/total road length")
    plt.legend()
    plt.show()


def fundamental_vs_lanes():
    road_length = 100
    n = 10
    density = np.linspace(0, 1, 20)
    for l in [1, 2, 3]:
        print(l)
        if l == 1:
            traffic = ["-"]
        else:
            traffic = ["EU", "NA"]

        for m in traffic:
            flow_rate = []
            for d in density:
                print(d)
                flow_rate_temp = []
                for _ in range(n):
                    road = Road(density=d, road_length=road_length, num_lanes=l, mu=-1, sigma=4)
                    simulation = Simulation(road)
                    simulation.run(propagator=Propagator(lane_change_model=m), numsteps=500)
                    flow_rate_temp.append(np.mean(simulation.obs.flowrate[-100:]))
                final = (np.mean(flow_rate_temp))
                flow_rate.append(final)
            plt.plot(density, flow_rate, label=f"lanes = {l}, traffic = {m}")
    plt.xlabel("Density, cars / road length")
    plt.ylabel("Flow rate, sum velocities/total road length")
    plt.legend()
    plt.show()

def road_heatmap():
    road_length = 100
    lanes = 3
    d = 0.7
    road = Road(density=d, road_length=road_length, num_lanes=lanes, mu=5, sigma=0)
    propagator = Propagator(lane_change_model="NA")
    numsteps = 500

    data = [[0] * road_length for _ in range(lanes)]
    for it in range(numsteps):
        propagator.propagate(road, Observables())
        for l in range(lanes):
            for x in range(road.road_length):
                if road.road[l][x][0] != -1:
                    data[l][x] += 1
    ax = sns.heatmap(data, linewidth=0, cmap="YlGnBu")
    plt.xlabel("x coordinate")
    plt.ylabel("lane")
    plt.show()


def main():
    #std_vs_n()
    #flow_rate_vs_t()
    #fundamental_diag_vs_road_length()
    fundamental_vs_lanes()
    #road_heatmap()


if __name__ == "__main__":
    main()
