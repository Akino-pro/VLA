import os
import sys

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient

from kortex_api.autogen.messages import Session_pb2, DeviceConfig_pb2, VisionConfig_pb2


def get_vision_device_id(device_manager: DeviceManagerClient) -> int:
    all_devices = device_manager.ReadAllDevices()
    vision_handles = [hd for hd in all_devices.device_handle if hd.device_type == DeviceConfig_pb2.VISION]
    if not vision_handles:
        print("No VISION device found.")
        return 0
    if len(vision_handles) > 1:
        print("Multiple VISION devices found; using the first.")
    vid = vision_handles[0].device_identifier
    print(f"VISION device id = {vid}")
    return vid


def try_print_intrinsics(vc: VisionConfigClient, vision_id: int, sensor: int, sensor_name: str):
    sid = VisionConfig_pb2.SensorIdentifier()
    sid.sensor = sensor
    print(f"\nTrying GetIntrinsicParameters for {sensor_name} ...")
    try:
        intr = vc.GetIntrinsicParameters(sid, vision_id)
        print(f"{sensor_name} intrinsics OK:")
        print(f"  resolution enum: {intr.resolution}")
        print(f"  cx={intr.principal_point_x:.6f}, cy={intr.principal_point_y:.6f}")
        print(f"  fx={intr.focal_length_x:.6f}, fy={intr.focal_length_y:.6f}")
        dc = intr.distortion_coeffs
        print(f"  distortion: k1={dc.k1:.6f}, k2={dc.k2:.6f}, p1={dc.p1:.6f}, p2={dc.p2:.6f}, k3={dc.k3:.6f}")
        return True
    except Exception as e:
        print(f"{sensor_name} intrinsics FAILED: {type(e).__name__}: {e}")
        return False


def main():
    ROBOT_IP = "192.168.1.10"
    USERNAME = "admin"
    PASSWORD = "admin"

    transport = TCPTransport()
    transport.connect(ROBOT_IP, 10000)
    router = RouterClient(transport, RouterClient.basicErrorCallback)

    session_manager = SessionManager(router)
    session = Session_pb2.CreateSessionInfo()
    session.username = USERNAME
    session.password = PASSWORD
    session.session_inactivity_timeout = 60000
    session.connection_inactivity_timeout = 2000
    session_manager.CreateSession(session)

    try:
        dm = DeviceManagerClient(router)
        vc = VisionConfigClient(router)

        vision_id = get_vision_device_id(dm)
        if vision_id == 0:
            return

        ok_color = try_print_intrinsics(vc, vision_id, VisionConfig_pb2.SENSOR_COLOR, "COLOR")
        ok_depth = try_print_intrinsics(vc, vision_id, VisionConfig_pb2.SENSOR_DEPTH, "DEPTH")

        print("\n--- Summary ---")
        print("COLOR supported:", ok_color)
        print("DEPTH supported:", ok_depth)
        if ok_color and not ok_depth:
            print("\nConclusion: Vision module is present but depth is not available/enabled on this system.")
            print("This matches: Web app shows color only + /depth RTSP fails.")
    finally:
        try:
            session_manager.CloseSession()
        except Exception:
            pass
        transport.disconnect()


if __name__ == "__main__":
    main()
