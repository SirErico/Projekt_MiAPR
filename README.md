# Implementacja planowania ruchu w 2D na mapie zajętości za pomocą algorytmu RRT z próbkowaniem sterowanym gradientem ograniczeń z sieci neuronowej

Projekt zaliczeniowy z przedmiotu **Metody i algorytmy planowania ruchu**

---

## 📄 OPIS

- Wybranie gotowej / utworzenie własnej mapy zajętości 2D.
- Spróbkowanie mapy - utworzenie datasetu (in: współrzędne, out: wolna/zajęta).
- Wytrenowanie prostej sieci neuronowej (MLP) w PyTorch lub Tensorflow.
- Przygotowanie metody odpytywania (inferencji) sieci z odczytem gradientu (din/dout) w punkcie.
- Integracja sieci z planerem RRT:
  - Wyuczona sieć (teoretycznie może zastąpić mapę przy sprawdzaniu zajętości w punkcie) powinna zwracać zerowy gradient dla miejsc daleko od granic obszarów wolnych/zajętych, natomiast niezerowy gradient w pobliżu przeszkód.
  - Gradient należy uwzględnić w funkcji próbkującej - wylosowane próbki w obszarze przeszkód można przesunąć wzdłuż gradientu do obszaru wolnego.
  - Można uwzględnić gradient w algorytmie również na inny sposób wg własnego pomysłu.
  - W rezultacie należy uzyskać mniejszą liczbę iteracji niż w algorytmie bez gradientu.
- Wizualizacja planowania ścieżki w Rviz, sprawdzenie czasu planowania, długości i kształtu ścieżek.

---

## 🛠 Steps to Take

### ROS2 Packages
Install the required ROS2 packages:
```bash
sudo apt-get install ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager
```

### Create a venv
```bash
python3 -m venv venv
```

### Source venv
```bash
source ~/path/to/venv/bin/activate
```
### Source ros2
```bash
source /opt/ros/humble/setup.bash
```

### Installing Python Packages
Install the Python dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Add 'COLCON_IGNORE' file to your venv folder
```bash
touch ~/venv/COLCON_IGNORE
```

### Building the ROS2 Workspace
Navigate to your ROS2 workspace and build the project:
```bash
cd ~/ros2_ws
colcon build 
source install/setup.bash
```

### Running the RRT Algorithm
To run the RRT algorithm with the neural net model, use the following command:
```bash
ros2 launch mapr_rrt rrt_ai_launch.py model_path:=/your/path/model.keras
```

---

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
