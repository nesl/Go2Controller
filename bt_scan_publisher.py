#!/usr/bin/env python3
import asyncio
import threading
import re
import rclpy
from rclpy.node import Node
from bt_msgs.msg import BtReading

class BtScanPublisher(Node):
    def __init__(self):
        super().__init__('bt_scan_publisher')

        # ---- Parameters
        self.declare_parameter('scanner_id', 'pi-1')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('adapter', 'hci0')           # choose adapter
        self.declare_parameter('scan_duration_sec', 0.1)    # sleep cadence in the loop
        self.declare_parameter('min_rssi', -120)            # filter very weak
        self.declare_parameter('include_unnamed', True)     # publish even if no name
        self.declare_parameter('allowlist_regex', '')       # optional: e.g. '^AA:BB:'
        self.declare_parameter('denylist_regex', '')        # optional: e.g. 'BeaconXYZ'

        self.scanner_id = self.get_parameter('scanner_id').value
        self.frame_id = self.get_parameter('frame_id').value
        self.adapter = self.get_parameter('adapter').value
        self.loop_sleep = float(self.get_parameter('scan_duration_sec').value)
        self.min_rssi = int(self.get_parameter('min_rssi').value)
        self.include_unnamed = bool(self.get_parameter('include_unnamed').value)
        aw = self.get_parameter('allowlist_regex').value
        dw = self.get_parameter('denylist_regex').value
        self.allow_pat = re.compile(aw) if aw else None
        self.deny_pat  = re.compile(dw) if dw else None

        self.pub = self.create_publisher(BtReading, '/bt/readings', 100)

        # ---- Start asyncio scanner in background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"BT scanner starting (adapter={self.adapter}, scanner_id={self.scanner_id}, frame_id={self.frame_id})"
        )

    # -------- Asyncio runner thread
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._scan_task())

    # -------- Version-agnostic Bleak scanner
    async def _scan_task(self):
        from bleak import BleakScanner

        def on_adv(device, adv):
            # Compatible fields across Bleak versions
            name = getattr(device, 'name', None) or getattr(adv, 'local_name', '') or ''
            rssi = int(getattr(adv, 'rssi', -100) or -100)
            addr = getattr(device, 'address', '') or ''

            if rssi < self.min_rssi:
                return
            if self.allow_pat and not (self.allow_pat.search(addr) or (name and self.allow_pat.search(name))):
                return
            if self.deny_pat and (self.deny_pat.search(addr) or (name and self.deny_pat.search(name))):
                return
            if not name or not re.fullmatch(r"CNode\d+", name):
                return

            msg = BtReading()
            msg.scanner_id = self.scanner_id
            msg.device_id = addr
            msg.device_name = name
            msg.rssi = rssi
            msg.stamp = self.get_clock().now().to_msg()
            msg.frame_id = self.frame_id
            self.pub.publish(msg)

        # Try new API first; fall back to old constructor-callback style
        try:
            scanner = BleakScanner(adapter=self.adapter)
            if hasattr(scanner, 'register_detection_callback'):
                scanner.register_detection_callback(on_adv)
                await scanner.start()
            else:
                raise AttributeError
        except AttributeError:
            scanner = BleakScanner(on_adv, adapter=self.adapter)
            await scanner.start()

        try:
            while rclpy.ok():
                await asyncio.sleep(self.loop_sleep)
        finally:
            await scanner.stop()

    # -------- Clean shutdown
    def destroy_node(self):
        try:
            if self._loop.is_running():
                # Stop loop after task ends
                def stop_loop():
                    self._loop.stop()
                self._loop.call_soon_threadsafe(stop_loop)
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = BtScanPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
