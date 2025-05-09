import rclpy
import os
import time
from mapr_rrt.grid_map import GridMap
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import sys
import matplotlib.pyplot as plt


np.random.seed(44)


class RRT(GridMap):
    def __init__(self):
        super(RRT, self).__init__()
        
        # Check python path
        print("Python executable:", sys.executable)
        
        # Load the trained model
        model_path = self.declare_parameter("model_path", "").get_parameter_value().string_value
        if not model_path or not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Please check the path.")
            sys.exit(1)
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

    def query_gradient(self, x, y):
        """
        Query the trained model for the gradient (occupancy probability) at a given point.
        :param x: x-coordinate (normalized)
        :param y: y-coordinate (normalized)
        :return: Occupancy probability (gradient)
        """
        # Normalize the input coordinates
        normalized_input = np.array([[x / self.map.shape[1], y / self.map.shape[0]]])
        
        # Predict the occupancy probability
        prediction = self.model.predict(normalized_input)
        
        # Return the gradient (occupancy probability)
        return prediction[0][0]

    def check_if_valid(self, a, b):
        """
        Checks if the segment connecting a and b lies in the free space.
        :param a: point in 2D
        :param b: point in 2D
        :return: boolean
        """
        num_samples = 100
        a = np.array([a[0], a[1]])  # Convert tuple to numpy array
        b = np.array([b[0], b[1]])  # Convert tuple to numpy array
        height, width = self.map.shape
        for t in np.linspace(b, a, num_samples, endpoint=False):
            # Sample point on the line segment
            t *= 1.0 / self.resolution
            # Check if the sample point is within bounds
            if not (0 <= t[0] < width and 0 <= t[1] < height):
                return False
            # Check if no collision
            if self.map[int(t[1]), int(t[0])] == 100:
                return False
        return True

    def random_point(self):
        """
        Draws random point in 2D
        :return: point in 2D
        """
        x = np.random.random(1) * self.width
        y = np.random.random(1) * self.height
        return (x[0], y[0])

    def find_closest(self, pos):
        """
        Finds the closest point in the graph (closest vertex or closes point on edge) to the pos argument
        If the point is on the edge, modifies the graph to obtain the valid graph with the new point and two new edges
        connecting existing vertices

        :param pos: point in 2D
        :return: point from the graph in 2D closest to the pos
        """
        closest = None
        min_dist = float('inf')
        modifications = []  # Temporary list to store graph modifications

        # Check all vertices in the graph
        for vertex in self.parent.keys():
            distance = np.linalg.norm(np.array(vertex) - pos)
            if distance < min_dist:
                min_dist = distance
                closest = vertex

        # Check all edges in the graph
        for child, parent in list(self.parent.items()):  # Use list() to create a static copy of items
            if parent is None:
                continue  # Skip the root node
            parent = np.array(parent)
            child = np.array(child)

            # Project pos onto the edge (parent -> child)
            edge_vector = child - parent
            edge_length = np.linalg.norm(edge_vector)
            if edge_length == 0:
                continue  # Skip degenerate edges
            edge_unit_vector = edge_vector / edge_length
            projection_length = np.dot(pos - parent, edge_unit_vector)

            # Clamp the projection to the edge segment
            projection_length = max(0, min(projection_length, edge_length))
            projection_point = parent + projection_length * edge_unit_vector

            # Calculate the distance from pos to the projection point
            distance = np.linalg.norm(pos - projection_point)
            if distance < min_dist:
                min_dist = distance
                closest = tuple(projection_point)

                # Store the graph modifications
                modifications.append((tuple(projection_point), tuple(parent)))
                modifications.append((tuple(child), tuple(projection_point)))

        # Apply the modifications to the graph after iteration
        for child, parent in modifications:
            self.parent[child] = parent

        return np.array(closest)

    def new_pt(self, pt, closest):
        """
        Finds last point in the free space on the segment connecting closest with pt
        :param pt: point in 2D
        :param closest: vertex in the tree (point in 2D)
        :return: point in 2D
        """
        closest_1 = np.array([closest[0], closest[1]])  # Convert tuple to numpy array
        pt_1 = np.array([pt[0], pt[1]])  # Convert tuple to numpy array

        # Direction vector from closest to pt
        direction = (pt_1 - closest_1) / np.linalg.norm(pt_1 - closest_1)

        current_point = closest_1

        while True:
            next_point = current_point + direction * 0.01
            if np.linalg.norm(next_point - closest_1) > np.linalg.norm(pt_1 - closest_1):
                return (pt_1[0], pt_1[1])  # Return the original point if it is closer
            if not self.check_if_valid(tuple(current_point), tuple(next_point)):
                return (current_point[0], current_point[1])  # Stop if the next point is not valid
            current_point = next_point

    def search(self):
        """
        RRT search algorithm for start point self.start and desired state self.end.
        Saves the search tree in the self.parent dictionary, with key value pairs representing segments
        (key is the child vertex, and value is its parent vertex).
        Uses self.publish_search() and self.publish_path(path) to publish the search tree and the final path respectively.
        """
        self.get_logger().info("============== RRT Search =============")
        self.get_logger().info("TEST MODEL READ")
        x, y = self.random_point()
        point = self.query_gradient(x, y)
        self.get_logger().info(f"Point: {point}")
        self.parent[tuple(self.start)] = None  # Ensure start is a tuple
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
            self.parent[tuple(new_pt)] = tuple(closest)  # Ensure keys and values are tuples
            # Publish the search tree
            self.publish_search()
            # Check if we reached the goal
            if self.check_if_valid(new_pt, self.end):
                self.parent[tuple(self.end)] = tuple(new_pt)  # Ensure end is a tuple
                print("Goal reached!")
                break
        else:
            print("Maximum iterations reached. Goal not found.")
            return  # Exit if the goal is not found within the iteration limit

        # Publish the path
        path = []
        current = tuple(self.end)  # Ensure end is a tuple
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()

        # Publish the path
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
