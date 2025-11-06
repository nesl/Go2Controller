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
        ('share/' + package_name + '/config', ['config/task_registry.yaml','config/rules.yaml']),  # optional: drop your YAML here
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
        ],
    },
)
