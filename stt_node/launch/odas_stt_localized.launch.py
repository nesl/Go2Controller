from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():


    decls = [
        DeclareLaunchArgument('mode', default_value='separated',
                              description="Choose 'postfiltered' (mono) or 'separated' (multi-channel)."),
        DeclareLaunchArgument('audio_port', default_value='9004',
                              description='ODAS audio TCP port (postfiltered OR separated).'),
        DeclareLaunchArgument('tracked_port', default_value='9000',
                              description='ODAS tracked JSON TCP port.'),
        DeclareLaunchArgument('n_channels', default_value='4',
                              description='Max separated channels (only used in separated mode).'),
        DeclareLaunchArgument('hop_samples', default_value='128',
                              description='ODAS hopSize (samples).'),
        DeclareLaunchArgument('stt_model', default_value='base',
                              description='faster-whisper model size: tiny/base/small/…'),
        DeclareLaunchArgument('stt_compute', default_value='auto',
                              description='auto/float16/float32/int8_float16'),
        DeclareLaunchArgument('stt_language', default_value='en',
                              description='Language code (empty = autodetect).'),
        DeclareLaunchArgument('stt_window_sec', default_value='2.0',
                              description='STT window seconds'),
        DeclareLaunchArgument('stt_hop_sec', default_value='1.0',
                              description='STT hop seconds'),
        DeclareLaunchArgument('switch_hysteresis_db', default_value='3.0',
                              description='dB advantage needed to switch active channel'),
        DeclareLaunchArgument('min_switch_ms', default_value='400',
                              description='Minimum hold time before switching (ms)'),
        DeclareLaunchArgument('stt_device',  default_value='cpu'),
        DeclareLaunchArgument('frame_id',    default_value='mic_array'),
        DeclareLaunchArgument('min_activity', default_value='0.25'),
        DeclareLaunchArgument('arrow_length', default_value='0.6'),
        DeclareLaunchArgument('shaft_diameter', default_value='0.03'),
        DeclareLaunchArgument('head_diameter',  default_value='0.08'),
       
        DeclareLaunchArgument('mic_parent', default_value='base_link'),
        DeclareLaunchArgument('mic_child',  default_value='mic_array'),
        DeclareLaunchArgument('mic_x', default_value='0.10'),
        DeclareLaunchArgument('mic_y', default_value='0.00'),
        DeclareLaunchArgument('mic_z', default_value='0.25'),
        # RPY are **radians**
        DeclareLaunchArgument('mic_roll',  default_value='0'),
        DeclareLaunchArgument('mic_pitch', default_value='0'),
        DeclareLaunchArgument('mic_yaw',   default_value='0'),

    ]



    # args
    mode               = LaunchConfiguration('mode')               # 'postfiltered' | 'separated'
    audio_port         = LaunchConfiguration('audio_port')         # TCP port from ODAS Pi
    tracked_port       = LaunchConfiguration('tracked_port')       # TCP port for tracked JSON
    n_channels         = LaunchConfiguration('n_channels')         # used only in separated mode
    hop_samples        = LaunchConfiguration('hop_samples')
    stt_model          = LaunchConfiguration('stt_model')
    stt_compute        = LaunchConfiguration('stt_compute')
    stt_language       = LaunchConfiguration('stt_language')
    stt_window_sec     = LaunchConfiguration('stt_window_sec')
    stt_hop_sec        = LaunchConfiguration('stt_hop_sec')
    switch_hysteresis  = LaunchConfiguration('switch_hysteresis_db')
    min_switch_ms      = LaunchConfiguration('min_switch_ms')
    
    frame_id          = LaunchConfiguration('frame_id')
    min_activity      = LaunchConfiguration('min_activity')
    arrow_length      = LaunchConfiguration('arrow_length')
    shaft_diameter    = LaunchConfiguration('shaft_diameter')
    head_diameter     = LaunchConfiguration('head_diameter')

    mic_parent = LaunchConfiguration('mic_parent')
    mic_child  = LaunchConfiguration('mic_child')
    mic_x      = LaunchConfiguration('mic_x')
    mic_y      = LaunchConfiguration('mic_y')
    mic_z      = LaunchConfiguration('mic_z')
    mic_roll   = LaunchConfiguration('mic_roll')
    mic_pitch  = LaunchConfiguration('mic_pitch')
    mic_yaw    = LaunchConfiguration('mic_yaw')    

    # TCP servers that ingest ODAS streams from the Pi
    audio_tcp = Node(
        package='odas_bridge',
        executable='odas_audio_receiver',
        name='odas_audio_receiver',
        output='screen',
        parameters=[{'port': audio_port}],
    )
    tracked_tcp = Node(
        package='odas_bridge',
        executable='odas_tracked_receiver',
        name='odas_tracked_receiver',
        output='screen',
        parameters=[{'port': tracked_port}, {'min_activity': min_activity}],
    )

    # If separated: pick active channel → /mic/audio/active
    active_selector = Node(
        package='odas_bridge',
        executable='active_channel_selector',
        name='active_channel_selector',
        output='screen',
        parameters=[{
            'input_topic': '/mic/audio',
            'n_channels': n_channels,
            'hop_samples': hop_samples,
            'switch_hysteresis_db': switch_hysteresis,
            'min_switch_ms': min_switch_ms
        }],
        condition=None  # we'll enable/disable below via opaque grouping
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_mic_array_tf',
        output='screen',
        arguments=[mic_x, mic_y, mic_z, mic_roll, mic_pitch, mic_yaw, mic_parent, mic_child],
    )

    # STT subscribes to either /mic/audio (postfiltered) or /mic/audio/active (separated)
    stt_post_params = [
        {'input_topic': '/mic/audio'},
        {'output_topic': '/stt/text'},
        {'partial_topic': '/stt/text_partial'},
        {'model_size': stt_model},
        {'compute_type': stt_compute},
        {'language': stt_language},  # empty string -> None inside node is okay
        {'vad_filter': True},
        {'window_sec': stt_window_sec},
        {'hop_sec': stt_hop_sec},
    ]
    stt_sep_params = [dict(p) for p in stt_post_params]
    stt_sep_params[0] = {'input_topic': '/mic/audio/active'}

    stt_post = Node(
        package='stt_node',
        executable='stt_subscriber',
        name='stt_subscriber_post',
        output='screen',
        parameters=stt_post_params,
    )
    stt_sep = Node(
        package='stt_node',
        executable='stt_subscriber',
        name='stt_subscriber_active',
        output='screen',
        parameters=stt_sep_params,
    )

    doa_marker = Node(
        package='odas_bridge', executable='doa_marker',
        name='doa_marker', output='screen',
        parameters=[{
            'frame_id': frame_id,
            'marker_topic': '/audio/doa_marker',
            'arrow_length': arrow_length,
            'shaft_diameter': shaft_diameter,
            'head_diameter': head_diameter,
            'ttl_sec': 1.5,
            'prefer_vector': True,  # prefer /audio/odas/doa
        }],
    )


    # Publish {text, channel, azimuth} as one message
    localized_pub = Node(
        package='odas_bridge',
        executable='localized_speech_publisher',
        name='localized_speech_publisher',
        output='screen',
    )

    # Mode selection: small trick—launch both STT variants but gate them by remapping
    # Simpler: actually include both and let the one with data produce output:
    # - postfiltered mode: /mic/audio exists, /mic/audio/active never published
    # - separated mode: active_channel_selector publishes /mic/audio/active, STT(sep) runs
    # If you prefer hard gating, split into two LaunchDescription branches.

    nodes = [audio_tcp, tracked_tcp, localized_pub, doa_marker, static_tf]

    # Always add active_selector; in postfiltered mode it will see 1ch frames and still publish /mic/audio/active,
    # but with identical bytes to /mic/audio it's harmless. If you prefer strictness, comment next line and only
    # use it when mode == 'separated'.
    nodes.append(active_selector)

    # Add both STT nodes; whichever input has data will emit text.
    nodes += [stt_post, stt_sep]

    return LaunchDescription(decls + nodes)
