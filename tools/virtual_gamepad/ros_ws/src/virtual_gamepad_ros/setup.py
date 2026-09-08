from setuptools import find_packages, setup

package_name = 'virtual_gamepad_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/virtual_gamepad.launch.py',
            'launch/virtual_gamepad_blind.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='equansrobotic',
    maintainer_email='jules.lenezet@gmail.com',
    description='Chef ROS pour virtual_gamepad -- agrege des noeuds de comportement (walk_to...) et publie sur LCM.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chef = virtual_gamepad_ros.chef_node:main',
            'walk_to = virtual_gamepad_ros.walk_to:main',
            'stand = virtual_gamepad_ros.stand:main',
            'lift = virtual_gamepad_ros.lift:main',
            'pivot = virtual_gamepad_ros.pivot:main',
            'depose = virtual_gamepad_ros.depose:main',
        ],
    },
)
