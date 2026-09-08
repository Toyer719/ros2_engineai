from setuptools import find_packages, setup

package_name = 'demo_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='equansrobotic',
    maintainer_email='jules.lenezet@gmail.com',
    description='Node lift_demo -- version simplifiee de test1 pour demonstration_pm01.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lift_demo = demo_ros.lift_demo:main',
        ],
    },
)
