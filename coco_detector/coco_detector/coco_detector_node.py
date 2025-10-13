#!/usr/bin/env python3
"""
YOLO + CameraInfo + PointCloud2 → map-frame human positions.

Subscribes:
- /robot0/front_cam/rgb            (sensor_msgs/Image)
- /robot0/point_cloud2             (sensor_msgs/PointCloud2)
- /robot0/camera_info              (sensor_msgs/CameraInfo)
- TF frames for cloud->camera and camera->map

Publishes:
- detected_objects                 (vision_msgs/Detection2DArray) — with pose for persons in map
- annotated_image                  (sensor_msgs/Image)            — optional
- visualization_marker_array       (visualization_msgs/MarkerArray) — spheres + labels in map

Key steps:
1) Transform cloud to camera frame and rasterize a per-pixel depth (Z-buffer).
2) YOLO detections → pick (u,v) at bottom-center; sample depth; back-project to 3D (camera).
3) Transform to map; fill Detection2D pose + emit RViz markers.
"""

import collections
import numpy as np
import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import BoundingBox2D, ObjectHypothesis, ObjectHypothesisWithPose
from vision_msgs.msg import Detection2D, Detection2DArray

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
from ultralytics import YOLO

import sensor_msgs_py.point_cloud2 as pc2
import image_geometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

import cv2
#import tf_transformations as tft
import pdb

Detection = collections.namedtuple("Detection", "label, bbox, score")  # bbox: [x1,y1,x2,y2], label: int

def quat_to_mat44(x, y, z, w, tx, ty, tz):
    """Minimal quaternion (x,y,z,w) + translation → 4x4 homogeneous matrix."""
    # normalized quaternion assumed from TF
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = tx; T[1, 3] = ty; T[2, 3] = tz
    return T
    
class CocoDetectorNode(Node):
    def __init__(self):
        super().__init__("coco_detector_node")

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter('device', 'cuda')                       # 'cuda' or 'cpu'
        self.declare_parameter('detection_threshold', 0.25)            # YOLO conf threshold
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('pointcloud_topic', '/point_cloud2')

        # Frames (adapt to your tree)
        self.declare_parameter('camera_frame', 'front_camera')  # camera *optical* frame preferred
        self.declare_parameter('target_frame', 'map')                  # RViz fixed frame or map frame

        # Depth rasterization settings
        self.declare_parameter('min_depth', 0.05)      # meters
        self.declare_parameter('search_win', 5)        # +/- pixels to search when depth is missing

        # Read params
        self.device = self.get_parameter('device').value
        self.det_thresh = float(self.get_parameter('detection_threshold').value)
        self.pub_annot = bool(self.get_parameter('publish_annotated_image').value)

        self.image_topic = self.get_parameter('image_topic').value
        self.caminfo_topic = self.get_parameter('camera_info_topic').value
        self.cloud_topic = self.get_parameter('pointcloud_topic').value

        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value

        self.min_depth = float(self.get_parameter('min_depth').value)
        self.search_win = int(self.get_parameter('search_win').value)

        # --------------------
        # QoS for camera/pointcloud (best effort typical)
        # --------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --------------------
        # Publishers
        # --------------------
        self.detected_objects_pub = self.create_publisher(Detection2DArray, "detected_objects", 10)
        self.annotated_image_pub = None
        if self.pub_annot:
            self.annotated_image_pub = self.create_publisher(Image, "annotated_image", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

        # --------------------
        # Subscribers
        # --------------------
        self.create_subscription(CameraInfo, self.caminfo_topic, self.camera_info_callback, qos_profile)
        self.create_subscription(PointCloud2, self.cloud_topic, self.pointcloud_callback, qos_profile)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile)

        # --------------------
        # TF
        # --------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------
        # Utilities
        # --------------------
        self.bridge = CvBridge()
        self.cam = None                            # image_geometry.PinholeCameraModel
        self.latest_cloud = None                   # PointCloud2
        self.latest_cloud_stamp = None

        # YOLO model (expects weights available locally)
        self.model = YOLO("yolo11x.pt")     # change path if needed
        self.class_labels = None                   # set per-result from YOLO

        self.get_logger().info("CocoDetectorNode started. Publishing map-frame human positions & markers.")

    # --------------------
    # Callbacks
    # --------------------
    def camera_info_callback(self, msg: CameraInfo):
        if self.cam is None:
            self.cam = image_geometry.PinholeCameraModel()
        self.cam.fromCameraInfo(msg)

    def pointcloud_callback(self, msg: PointCloud2):
        self.latest_cloud = msg
        self.latest_cloud_stamp = msg.header.stamp

    def _lookup_tf_with_fallback(self, target, source, stamp, timeout_sec=0.2):
        try:
            return self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time.from_msg(stamp),
                rclpy.duration.Duration(seconds=timeout_sec)
            )
        except TransformException as e:
            '''
            self.get_logger().warn(
                f"TF @stamp failed ({e}). Falling back to latest (time=0)."
            )
            '''
            return self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(),  # latest available
                rclpy.duration.Duration(seconds=timeout_sec)
            )
            
    def image_callback(self, msg: Image):
        # Need intrinsics and a cloud frame to compute a depth map
        if self.cam is None or self.latest_cloud is None:
            return

        # --- TFs for this timestamp ---
        try:
            cloud_frame = self.latest_cloud.header.frame_id
            stamp = self.latest_cloud.header.stamp
            tf_cloud_to_cam = self._lookup_tf_with_fallback(self.camera_frame, cloud_frame, stamp)
            tf_cam_to_map   = self._lookup_tf_with_fallback(self.target_frame, self.camera_frame, stamp)

        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return

        T_cloud_cam = self._tf_to_matrix(tf_cloud_to_cam)

        # --- Get/rectify RGB ---
        cv_img_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        cv_img_rect = cv_img_raw.copy()
        try:
            self.cam.rectifyImage(cv_img_raw, cv_img_rect)
        except Exception:
            cv_img_rect = cv_img_raw

        # --- YOLO inference ---
        results = self.model.predict(cv_img_rect, verbose=False, device=self.device)
        dets: list[Detection] = []
        annot_img_bgr = cv_img_rect.copy()[:, :, ::-1]  # fallback if result.plot not called

        for result in results:
            # names is dict: id->name
            self.class_labels = result.names
            # Plot gives BGR image
            annot_img_bgr = result.plot()
            # xyxy: tensor N x 4; cls: N; conf: N
            for cls_id, box, conf in zip(result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
                conf_f = float(conf)
                if conf_f < self.det_thresh:
                    continue
                dets.append(Detection(int(cls_id), box.cpu().numpy(), conf_f))

        # --- Depth Z-buffer from cloud in camera frame ---
        vals = pc2.read_points_numpy(self.latest_cloud, field_names=('x', 'y', 'z'))
        if vals.size == 0:
            return
        pts_cam = self._transform_points(T_cloud_cam, vals)  # (N,3) camera frame

        W, H = self.cam.fullResolution()  # returns (width, height)
        fx, fy, cx, cy = self.cam.fx(), self.cam.fy(), self.cam.cx(), self.cam.cy()
        depth_img = self._build_depth_image(pts_cam, int(W), int(H), fx, fy, cx, cy)

        # --- Build Detection2DArray and fill person 3D poses in map ---
        det_array = Detection2DArray()
        det_array.header = msg.header

        marker_array = MarkerArray()
        marker_id = 0

        for d in dets:
            detection2d = self._yolo_to_detection2d(d, msg.header)

            label = self._label_str(d.label)
            if label == "person":
                x1, y1, x2, y2 = [float(v) for v in d.bbox]
                u = 0.5 * (x1 + x2)
                v = y2  # bottom of bbox ~ feet

                Z = self._depth_at(depth_img, u, v, win=self.search_win)
                if np.isfinite(Z) and Z > self.min_depth:
                    ray = np.array(self.cam.projectPixelTo3dRay((u, v)), dtype=np.float32)  # unit vector
                    p_cam = ray * float(Z)  # (X,Y,Z) in camera frame

                    ps_cam = PointStamped()
                    ps_cam.header.stamp = msg.header.stamp
                    ps_cam.header.frame_id = self.camera_frame
                    ps_cam.point.x, ps_cam.point.y, ps_cam.point.z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

                    try:
                        ps_map = do_transform_point(ps_cam, tf_cam_to_map)
                        # Write pose in map into results[0]
                        if not detection2d.results:
                            oh = ObjectHypothesisWithPose()
                            oh.hypothesis.class_id = label
                            oh.hypothesis.score = d.score
                            detection2d.results.append(oh)
                        detection2d.results[0].pose.pose.position.x = ps_map.point.x
                        detection2d.results[0].pose.pose.position.y = ps_map.point.y
                        detection2d.results[0].pose.pose.position.z = ps_map.point.z
                        detection2d.header.frame_id = self.target_frame

                        # RViz markers
                        m = Marker()
                        m.header.frame_id = self.target_frame
                        m.header.stamp = msg.header.stamp
                        m.ns = "humans"
                        m.id = marker_id; marker_id += 1
                        m.type = Marker.SPHERE
                        m.action = Marker.ADD
                        m.pose.position.x = ps_map.point.x
                        m.pose.position.y = ps_map.point.y
                        m.pose.position.z = ps_map.point.z
                        m.scale.x = m.scale.y = m.scale.z = 0.25
                        m.color.a = 0.9; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
                        marker_array.markers.append(m)

                        t = Marker()
                        t.header.frame_id = self.target_frame
                        t.header.stamp = msg.header.stamp
                        t.ns = "humans_text"
                        t.id = marker_id; marker_id += 1
                        t.type = Marker.TEXT_VIEW_FACING
                        t.action = Marker.ADD
                        t.pose.position.x = ps_map.point.x
                        t.pose.position.y = ps_map.point.y
                        t.pose.position.z = ps_map.point.z + 0.5
                        t.scale.z = 0.25
                        t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
                        t.text = f"person ({d.score:.2f})"
                        marker_array.markers.append(t)

                    except Exception as e:
                        self.get_logger().warn(f"camera->map transform failed: {e}")

            det_array.detections.append(detection2d)

        # --- Publish ---
        self.detected_objects_pub.publish(det_array)

        if self.annotated_image_pub is not None:
            # result.plot() returns BGR; convert to RGB for 'rgb8'
            annot_rgb = cv2.cvtColor(annot_img_bgr, cv2.COLOR_BGR2RGB)
            ros_img = self.bridge.cv2_to_imgmsg(annot_rgb, encoding="rgb8")
            ros_img.header = msg.header
            self.annotated_image_pub.publish(ros_img)

        self.marker_pub.publish(marker_array)

    # --------------------
    # Helpers
    # --------------------
    def _label_str(self, idx: int) -> str:
        if isinstance(self.class_labels, dict):
            return self.class_labels.get(idx, str(idx))
        if isinstance(self.class_labels, list) and 0 <= idx < len(self.class_labels):
            return self.class_labels[idx]
        return str(idx)

    def _yolo_to_detection2d(self, detection: Detection, header):
        detection2d = Detection2D()
        detection2d.header = header

        oh = ObjectHypothesis()
        oh.class_id = self._label_str(detection.label)
        oh.score = float(detection.score)

        ohp = ObjectHypothesisWithPose()
        ohp.hypothesis = oh
        detection2d.results.append(ohp)

        x1, y1, x2, y2 = [float(v) for v in detection.bbox]
        bbox = BoundingBox2D()
        bbox.center.position.x = (x1 + x2) * 0.5
        bbox.center.position.y = (y1 + y2) * 0.5
        bbox.center.theta = 0.0
        bbox.size_x = (x2 - x1)
        bbox.size_y = (y2 - y1)
        detection2d.bbox = bbox
        return detection2d

    def _tf_to_matrix(self, tf_msg):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        return quat_to_mat44(q.x, q.y, q.z, q.w, t.x, t.y, t.z)

    def _transform_points(self, T, pts_xyz):  # (N,3) -> (N,3)
        if pts_xyz.ndim == 1:
            pts_xyz = pts_xyz.reshape(1, 3)
        ones = np.ones((pts_xyz.shape[0], 1), dtype=np.float32)
        hom = np.hstack([pts_xyz.astype(np.float32), ones])   # (N,4)
        out = (T @ hom.T).T                                   # (N,4)
        return out[:, :3]

    def _build_depth_image(self, pts_cam, W, H, fx, fy, cx, cy):
        """Rasterize a Z-buffer depth image (meters) from points in camera frame."""
        X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        valid = Z > self.min_depth
        X = X[valid]; Y = Y[valid]; Z = Z[valid]
        depth = np.full((H, W), np.nan, dtype=np.float32)
        if Z.size == 0:
            return depth

        u = (fx * X / Z + cx).astype(np.int32)
        v = (fy * Y / Z + cy).astype(np.int32)
        keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[keep]; v = v[keep]; Z = Z[keep]
        if Z.size == 0:
            return depth

        # For duplicates, keep the nearest Z (classic Z-buffer)
        # Use flat indexing + segment-wise minimum
        flat = v * W + u
        order = np.argsort(flat)
        flat_sorted = flat[order]
        Z_sorted = Z[order]

        seg_start = np.ones_like(flat_sorted, dtype=bool)
        seg_start[1:] = flat_sorted[1:] != flat_sorted[:-1]

        cummins = Z_sorted.copy()
        for i in range(1, Z_sorted.size):
            if not seg_start[i]:
                cummins[i] = min(cummins[i-1], Z_sorted[i])

        write_mask = np.zeros_like(seg_start, dtype=bool)
        write_mask[:-1] = seg_start[1:]
        write_mask[-1] = True

        depth_flat = np.full(H * W, np.nan, dtype=np.float32)
        depth_flat[flat_sorted[write_mask]] = cummins[write_mask]
        return depth_flat.reshape(H, W)

    def _depth_at(self, depth_img, u, v, win=5):
        H, W = depth_img.shape
        ui = int(round(u)); vi = int(round(v))
        if 0 <= ui < W and 0 <= vi < H and np.isfinite(depth_img[vi, ui]):
            return float(depth_img[vi, ui])
        u0 = max(0, ui - win); u1 = min(W, ui + win + 1)
        v0 = max(0, vi - win); v1 = min(H, vi + win + 1)
        patch = depth_img[v0:v1, u0:u1]
        if np.isfinite(patch).any():
            return float(np.nanmin(patch))
        return float('nan')


    

def main():
    rclpy.init()
    node = CocoDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
