# contour_waypoints.py
import rclpy
from rclpy.node import Node
import numpy as np, cv2, math
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid

def yaw_from_tangent(p_prev, p_next):
    dx, dy = (p_next[0]-p_prev[0]), (p_next[1]-p_prev[1])
    return math.atan2(dy, dx)

class ContourWaypoints(Node):
    def __init__(self):
        super().__init__('contour_waypoints')
        self.declare_parameter('clearance_m', 0.40)
        self.declare_parameter('downsample_m', 0.75)
        self.declare_parameter('frame_id', 'map')
        self.clearance = float(self.get_parameter('clearance_m').value)
        self.ds = float(self.get_parameter('downsample_m').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self._sub = self.create_subscription(OccupancyGrid, '/map', self.on_map, 1)
        self._ac = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.got = False

    def on_map(self, msg: OccupancyGrid):
        if self.got:
            return
        self.got = True
        res = msg.info.resolution
        origin = msg.info.origin  # geometry_msgs/Pose
        w, h = msg.info.width, msg.info.height
        data = np.array(msg.data, dtype=np.int16).reshape(h, w)

        # Build binary free/occupied masks
        occ = (data >= 50).astype(np.uint8)   # occupied = 1
        free = (data == 0).astype(np.uint8)   # free = 1

        # Distance (in pixels) from obstacles, computed over free space
        # Use 8-connectivity and normalize by resolution later
        dist = cv2.distanceTransform((free*255), cv2.DIST_L2, 3)
        # Target distance in pixels
        target_pix = max(1, int(self.clearance / res))
        # Extract a narrow band around the iso-contour
        band = np.logical_and(dist >= target_pix-1, dist <= target_pix+1).astype(np.uint8)*255

        # Clean up and keep the largest loop
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            self.get_logger().error('No contour found at the requested clearance. Try smaller clearance.')
            return
        # Choose the longest contour (outer loop)
        c = max(contours, key=lambda cnt: len(cnt))

        # Simplify the contour for fewer points
        epsilon = max(1.5, 0.01*cv2.arcLength(c, True))
        c = cv2.approxPolyDP(c, epsilon, True).reshape(-1,2)

        # Downsample to roughly ds meters between points
        pts = []
        acc = 0.0
        for i in range(len(c)):
            p0 = c[i]
            p1 = c[(i+1) % len(c)]
            seg_len = np.linalg.norm(p1 - p0) * res
            if not pts:
                pts.append(p0)
                acc = 0.0
            acc += seg_len
            if acc >= self.ds:
                pts.append(p1)
                acc = 0.0
        if len(pts) < 3:
            self.get_logger().error('Too few points after downsampling.')
            return

        # Convert pixel coords to map frame (meters)
        def pix_to_xy(px, py):
            # OpenCV image coords: (x=col, y=row); map origin at bottom-left of grid?
            # OccupancyGrid data is row-major, origin is the world pose of cell (0,0).
            x = origin.position.x + (px + 0.5) * res
            y = origin.position.y + (py + 0.5) * res
            return (x, y)

        # Build PoseStamped array with tangential headings
        poses = []
        for i in range(len(pts)):
            p_prev = pts[i-1]
            p_curr = pts[i]
            p_next = pts[(i+1) % len(pts)]
            x, y = pix_to_xy(p_curr[0], p_curr[1])
            yaw = yaw_from_tangent(pix_to_xy(p_prev[0], p_prev[1]), pix_to_xy(p_next[0], p_next[1]))
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            # yaw -> quaternion
            cy = math.cos(yaw*0.5); sy = math.sin(yaw*0.5)
            pose.pose.orientation.z = sy
            pose.pose.orientation.w = cy
            poses.append(pose)

        # Send FollowWaypoints goal (single loop)
        goal = FollowWaypoints.Goal()
        goal.poses = poses

        self._ac.wait_for_server()
        self.get_logger().info(f'Sending {len(poses)} contour waypoints...')
        send_fut = self._ac.send_goal_async(goal)
        # Optionally, add callbacks to monitor result; we’ll keep it simple
        return

def main():
    rclpy.init()
    node = ContourWaypoints()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
