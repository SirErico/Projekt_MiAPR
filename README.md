# Implementation of 2D Motion Planning on an Occupancy Map using RRT Algorithm with Gradient-Based Sampling Guided by Neural Network

## Repository Overview
This project implements a Rapidly-exploring Random Tree (RRT) algorithm enhanced with neural network-guided sampling for path planning in 2D environments.

### Prerequisites
1. ROS 2 (tested on Humble) with required ros2 packages
2. Python 3.10
3. Required Python packages (requirements.txt):


## 🛠 Steps to Take

1. Install the required ROS2 packages:
```bash
sudo apt-get install ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager
```

2. Create ROS 2 workspace:
```bash
mkdir -p ros2_ws/src
cd ros2_ws/src
```

3. Clone the repository:
```bash
git clone <repository-url>
```

4. Before building, create a venv
```bash
cd ..
python3 -m venv venv
```

5. Install the Python dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

6. Add 'COLCON_IGNORE' file to your venv folder
```bash
touch ~/venv/COLCON_IGNORE
```

7. Source venv
```bash
source ~/path/to/venv/bin/activate
```
8.  Build and source workspace
```bash
colcon build
source /install/setup.bash
```



## Usage

#### Training the Neural Network
1. Generate training data:
```bash
python3 src/sample_map.py
```

2. Train the neural network:
```bash
python3 src/neural_net.py
```

#### Running the RRT Algorithm
To run the RRT algorithm with the neural net model:
```bash
ros2 launch mapr_rrt rrt_ai_launch.py model_path:=/your/path/model.keras
```

---

## Scripts:
- `map_blur.py`: Applies a Gaussian blur to a map image.
- `sample_map.py`: Samples data points from a map to create a dataset for training a neural network.
- `show_map.py`: Visualization of the map .pgm and .csv file.
- `neural_net.py`: Trains a neural network to learn occupancy probabilities from a map dataset.
- `show_model.py`: Predicts occupancy for the entire map and visualizes the original and predicted occupancy maps.


## ⚠️ Known Problems
### venv Packages Not Visible
If the virtual environment packages are not visible, add the following to your `setup.cfg` file:
```ini
[build]
executable=/usr/bin/env python3
```

---

## 🔗 Links

- [RRT Algorithm (PUT JUG)](https://put-jug.github.io/lab-miapr/Lab%206%20-%20Algorytmy%20poszukiwania%20%C5%9Bcie%C5%BCki%20pr%C3%B3bkuj%C4%85ce%20przestrze%C5%84%20poszukiwa%C5%84%20na%20przyk%C5%82adzie%20RRT%20(Rapidly-exploring%20Random%20Tree).html)
- [RRT Algorithm (Random Article)](https://theclassytim.medium.com/robotic-path-planning-rrt-and-rrt-212319121378)
