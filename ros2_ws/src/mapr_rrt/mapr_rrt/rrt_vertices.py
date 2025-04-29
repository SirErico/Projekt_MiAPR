import rclpy
import time
from mapr_rrt.grid_map import GridMap
import numpy as np

np.random.seed(44)


class RRT(GridMap):
    def __init__(self):
        super(RRT, self).__init__()
        self.step = 0.1

    def check_if_valid(self, a, b):
        """
        Checks if the segment connecting a and b lies in the free space.
        :param a: point in 2D
        :param b: point in 2D
        :return: boolean
        """
        num_samples = 100
        a = np.array([a[0],a[1]])  # Convert tuple to numpy array
        b = np.array([b[0],b[1]])  # Convert tuple to numpy array
        height, width = self.map.shape
        for t in np.linspace(b, a, num_samples, endpoint=False):
            # Sample point on the line segment
            t *= 1.0 / self.resolution
            # Check if the sample point is within bounds
            if not (0 <= t[0] < width and 0 <= t[1] < height):
                return False
            # check if no collision
            if self.map[int(t[1]), int(t[0])] == 100:
                return False
        return True

    def random_point(self):
        """
        Draws random point in 2D
        :return: point in 2D
        """
        # x = y = 0.
        # bounds are [0; self.width] and [0; self.height]
        x = np.random.random(1) * self.width
        y = np.random.random(1) * self.height
        return (x[0], y[0])

    def find_closest(self, pos):
        """
        Finds the closest vertex in the graph to the pos argument

        :param pos: point id 2D
        :return: vertex from graph in 2D closest to the pos
        """
        # closest = pos
        closest = None
        min_dist = float('inf')
        for i in self.parent.keys():
            distance = np.linalg.norm(np.array(i) - pos)
            if distance < min_dist:
                min_dist = distance
                closest = i

        return closest

    def new_pt(self, pt, closest):
        """
        Finds the point on the segment connecting closest with pt, which lies self.step from the closest (vertex in graph)

        :param pt: point in 2D
        :param closest: vertex in the tree (point in 2D)
        :return: point in 2D
        """
        closest_1 = np.array([closest[0], closest[1]])  # Convert tuple to numpy array
        pt_1 = np.array([pt[0], pt[1]])  # Convert tuple to numpy array
        # Check if the new point is valid
        direction = (pt_1 - closest) / np.linalg.norm(pt_1 - closest)
        new_pt = closest_1 + direction * min(self.step, np.linalg.norm(pt_1 - closest_1))
        return (new_pt[0], new_pt[1])


    def search(self):
        """
        RRT search algorithm for start point self.start and desired state self.end.
        Saves the search tree in the self.parent dictionary, with key value pairs representing segments
        (key is the child vertex, and value is its parent vertex).
        Uses self.publish_search() and self.publish_path(path) to publish the search tree and the final path respectively.
        """
        self.parent[self.start] = None
        while True:
            # Draw random point
            random_pt = self.random_point()
            # Find the closest vertex in the graph to the random point
            closest = self.find_closest(random_pt)
            # Find the new point on the segment connecting closest and pt
            new_pt = self.new_pt(random_pt, closest)
            # Check if the new point is valid
            if not self.check_if_valid(closest, new_pt):
                continue
            # Add the new point to the graph
            self.parent[new_pt] = closest
            # pulish the search tree
            self.publish_search()
            # Check if we reached the goal
            # if self.check_if_valid(np.array(new_pt), self.end):
            if self.check_if_valid(new_pt, self.end):
                self.parent[self.end] = new_pt
                print("Goal reached!")
                break
        else:
            print("Maximum iterations reached. Goal not found.")
            return  # Exit if the goal is not found within the iteration limit

        
        # Publish the path
        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()

        # publish the path
        self.publish_path(path)



def main(args=None):
    rclpy.init(args=args)
    rrt = RRT()
    while not rrt.data_received():
        rrt.get_logger().info("Waiting for data...")
        rclpy.spin_once(rrt)
        time.sleep(0.5)

    rrt.get_logger().info("Start graph searching!")
    time.sleep(1)
    rrt.search()


if __name__ == '__main__':
    main()
