from setuptools import find_packages, setup

package_name = 'architecture'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/task_registry.yaml','config/rules.yaml', 'config/skills.yaml', 'config/rules_init.yaml', 'config/skills_composite.yaml']),  # optional: drop your YAML here
        ('share/' + package_name + '/launch', ['launch/architecture.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='kiototeko@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'event_layer_node = architecture.event_layer_node:main',
            'skills_node = architecture.skills_node:main',
            'broker_node = architecture.broker_node:main',
            'planner_node = architecture.planner_node:main',
            'orchestrator_node = architecture.orchestrator_node:main',
            'hdt_node = architecture.hdt:main',
            'router_node = architecture.router_node:main',
            'reactive_node = architecture.reactive_node:main',
            'coordinator_node = architecture.coordinator_node:main',
            'taskstatemonitor_node = architecture.taskstatemonitor_node:main',
            'hdt2_node = architecture.hdt2_node:main',
        ],
    },
)
