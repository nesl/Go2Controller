from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sim_humans'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", "sim_humans", "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", "sim_humans", "config"), glob("config/*.yaml")),
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
        'sim_human_agent = sim_humans.sim_human_agent_node:main',
        'sim_optimzed_agent = sim_humans.optimized_agent:main',
        'sim_centralized_agent = sim_humans.plan_follower_agent_node:main',
        'sim_centralized_optimizer = sim_humans.central_joint_optimizer_node:main',
  
        ],
    },
)
