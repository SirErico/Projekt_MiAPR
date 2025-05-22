import rclpy
import os
import time
from mapr_rrt.grid_map import GridMap
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING
import tensorflow as tf
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
import copy

np.random.seed(44)

class RRT(GridMap):
    def __init__(self):
        super(RRT, self).__init__()
        
        # # Check python path
        # print("Python executable:", sys.executable)
        
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
        normalized_input = np.array([[x / self.width, y / self.height]])
        
        # Predict the occupancy probability
        prediction = self.model.predict(normalized_input)
        
        # Return the gradient (occupancy probability)
        return prediction[0][0]


    def gradient_at(self, x, y):
        """
        Compute gradient of model output with respect to input (x, y), both in map pixel space.
        :return: gradient vector as np.array([dx, dy])
        """
        input_tensor = tf.convert_to_tensor([[x / self.width, y / self.height]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = self.model(input_tensor)
        grad = tape.gradient(prediction, input_tensor).numpy()[0]

        # Unnormalize gradient to pixel space
        dx = grad[0] * self.width
        dy = grad[1] * self.height
        return np.array([dx, dy])


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
        Finds the closest vertex in the existing tree.
        """
        pos = np.array(pos)
        closest = None
        min_dist = float('inf')

        # Only check existing vertices (no edge modifications)
        for vertex in self.parent.keys():
            vertex = np.array(vertex)
            distance = np.linalg.norm(vertex - pos)
            if distance < min_dist and self.check_if_valid(vertex, pos):
                min_dist = distance
                closest = vertex

        return closest if closest is not None else np.array(self.start)

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
        self.parent[tuple(self.start)] = None  # Ensure start is a tuple
        
        while True:
            # Draw random point
            random_pt = self.random_point()
            original_random_pt = copy.deepcopy(random_pt) # Store the original random point

            # TODO więcej próbek, mniejsza sieć
            # 10 punktow i wyznaczyc gradient
            # optymalizacja stepsize sprobowac
            # === 2. Check if occupied using model prediction ===
            for u in range(30):  
                # Check if point is in bounds
                if not (0 <= random_pt[0] < self.width and 0 <= random_pt[1] < self.height):
                    random_pt = np.clip(random_pt, [0, 0], [self.width-1, self.height-1])
                    break
                occ_prob = self.query_gradient(*random_pt)
                
                # Idk which threshold is good
                if occ_prob < 0.50:
                    break

                # Use gradient to move toward free space
                grad = self.gradient_at(*random_pt)
                
                # Normalize the gradient for consistent step sizes
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 0:
                    grad = grad / grad_norm
                
                # Adaptive step size based on occupancy probability
                # step_size = 0.1 * (1.0 + 2.0 * occ_prob)  # Move more aggressively in highly occupied areas
                step_size = 0.1 * (1.0 + occ_prob)
                
                self.get_logger().info(f"Point: {random_pt}, Occ: {occ_prob:.2f}, Gradient: {grad}, Step: {step_size:.2f}")
                
                # Move against the gradient (toward free space)
                random_pt = random_pt - grad * step_size
                
                # Clip to bounds
                # random_pt[0] = np.clip(random_pt[0], 0, self.width - 1)
                # random_pt[1] = np.clip(random_pt[1], 0, self.height - 1)
                random_pt = np.clip(random_pt, [0, 0], [self.width-1, self.height-1])
                
                if np.linalg.norm(random_pt - original_random_pt) < 0.01:
                    break

            # === 3. Find nearest node in the tree ===
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
                self.parent[tuple(self.end)] = tuple(new_pt)
                self.get_logger().info("Goal reached!")
                break
            # else:
            #     print("Maximum iterations reached. Goal not found.")
            #     return  # Exit if the goal is not found within the iteration limit

        # Publish the path
        path = []
        current = tuple(self.end)  # Ensure end is a tuple
        visited = set()
        if current not in self.parent:
            self.get_logger().warn("End point not in parent dictionary")
            return
        while current is not None and current not in visited:
            path.append(current)
            visited.add(current)
            current = self.parent.get(current, None)
        
        if path and tuple(self.start) not in path:
            path.append(tuple(self.start))
        path.reverse()
        
        
        print("Path found:", path)
        self.get_logger().info(f"Path length: {len(path)}")
        
        # Verify path connectivity
        for i in range(len(path) - 1):
            if not self.check_if_valid(path[i], path[i + 1]):
                self.get_logger().error(f"Invalid segment between {path[i]} and {path[i + 1]}")
     
        
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
    start_time = time.time()
    rrt.search()
    end_time = time.time()
    execution_time = end_time - start_time
    
    rrt.get_logger().info(f"RRT with neural nets path planning took {execution_time:.2f} seconds")


if __name__ == '__main__':
    main()
