from setuptools import find_packages, setup

package_name = 'odas_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='kiototeko@gmail.com',
    description='UDP receivers for ODAS (audio + tracking) to ROS 2 topics',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odas_audio_receiver = odas_bridge.odas_audio_receiver:main',
            'odas_tracked_receiver = odas_bridge.odas_tracked_receiver:main',
            'save_audio_wav = odas_bridge.save_audio_wav:main',
            'localized_speech_publisher = odas_bridge.localized_speech_publisher:main',
            'active_channel_selector = odas_bridge.active_channel_selector:main',
            'doa_marker = odas_bridge.rviz_doa_marker:main',
            'ssl_doa = odas_bridge.ssldoareader:main',
            'stt_angle = odas_bridge.stt_angle:main'

        ],
    },
)
