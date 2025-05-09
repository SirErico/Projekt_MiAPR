import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition


def generate_launch_description():

    # Declare the model path launch argument
    # Either launch default value or pass it as an argument
    # Example: ros2 launch mapr_rrt rrt_ai_launch.py model_path:=/some/other/model.keras

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model.keras',
        description='Path to the Keras model file'
    )
    
    model_path = LaunchConfiguration('model_path')
    # map_path = os.path.join(get_package_share_directory('mapr_rrt'), 'maps', 'map_small.yaml')
    map_path = os.path.join(get_package_share_directory('mapr_rrt'), 'maps', 'map_medium.yaml')
    # map_path = os.path.join(get_package_share_directory('mapr_rrt'), 'maps', 'map.yaml')

    map_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mapr_rrt'), 'launch', 'map_launch.py')
        ),
        launch_arguments={'map': map_path}.items()
    )
    
    rrt_cmd = GroupAction(
        actions=[
            Node(
                package='mapr_rrt',
                executable='rrt_ai',
                name='rrt_node',
                output='screen',
                parameters=[{
                'model_path': model_path
                }]
            )
        ]
    )
    
    points_cmd = Node(
            package='mapr_rrt',
            executable='points',
            name='points')

    rviz_config_dir = os.path.join(
        get_package_share_directory('mapr_rrt'), 'rviz', 'rviz_cfg.rviz')

    rviz_cmd = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': True}],
            output='screen')

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(model_path_arg)
    ld.add_action(map_server_cmd)
    ld.add_action(rrt_cmd)
    ld.add_action(points_cmd)
    ld.add_action(rviz_cmd)

    return ld
