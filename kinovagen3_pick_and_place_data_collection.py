import csv
import time
import math
from datetime import datetime

from kortex_api.UDPTransport import UDPTransport
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

UDP_PORT = 10001  # cyclic feedback
TCP_PORT = 10000  # base commands/queries (typical)


def connect_udp(ip: str, username: str, password: str):
    transport = UDPTransport()
    transport.connect(ip, UDP_PORT)

    router = RouterClient(transport, RouterClient.basicErrorCallback)

    session_manager = SessionManager(router)
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = username
    session_info.password = password
    session_info.session_inactivity_timeout = 60000
    session_info.connection_inactivity_timeout = 2000
    session_manager.CreateSession(session_info)

    return transport, session_manager, router


def connect_tcp(ip: str, username: str, password: str):
    transport = TCPTransport()
    transport.connect(ip, TCP_PORT)

    router = RouterClient(transport, RouterClient.basicErrorCallback)

    session_manager = SessionManager(router)
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = username
    session_info.password = password
    session_info.session_inactivity_timeout = 60000
    session_info.connection_inactivity_timeout = 2000
    session_manager.CreateSession(session_info)

    return transport, session_manager, router


def safe_disconnect(transport, session_manager):
    try:
        session_manager.CloseSession()
    except Exception:
        pass
    try:
        transport.disconnect()
    except Exception:
        pass


def deg2rad(x_deg: float) -> float:
    return x_deg * math.pi / 180.0


def main():
    # --------- EDIT THESE ----------
    ROBOT_IP = "192.168.1.10"
    USERNAME = "admin"
    PASSWORD = "admin"

    SAMPLE_HZ = 10
    DURATION_S = 10
    N_JOINTS = 7

    EPISODE_INDEX = 0
    TASK_INDEX = 0

    # inference parameters
    GRIPPER_DEADBAND = 0.002  # in normalized [0,1] units; tune if too jittery
    FLUSH_EVERY_N = 25        # flush every N rows to reduce data loss risk
    # ------------------------------

    out_csv = f"libero_like_no_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    period = 1.0 / float(SAMPLE_HZ)

    # UDP for fast arm feedback
    udp_transport, udp_session, udp_router = connect_udp(ROBOT_IP, USERNAME, PASSWORD)
    # TCP for gripper measured movement query
    tcp_transport, tcp_session, tcp_router = connect_tcp(ROBOT_IP, USERNAME, PASSWORD)

    try:
        base_cyclic = BaseCyclicClient(udp_router)
        base = BaseClient(tcp_router)

        # prepare gripper request (measured)
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION

        last_grip = None  # last measured gripper position (float) or None
        frame_index = 0
        rows_since_flush = 0

        t0 = time.perf_counter()
        next_t = t0

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)

            header = (
                ["timestamp_sec", "frame_index", "episode_index", "task_index"]
                + [f"state_q{i+1}_rad" for i in range(N_JOINTS)]
                + [
                    "action_x_m", "action_y_m", "action_z_m",
                    "action_rx_rad", "action_ry_rad", "action_rz_rad",
                    "action_gripper_delta_cmd",      # 1 / -1 / 0
                    "gripper_pos_measured_01"
                ]
            )
            w.writerow(header)

            while True:
                now = time.perf_counter()
                t_sec = now - t0
                if t_sec >= DURATION_S:
                    break

                fb = base_cyclic.RefreshFeedback()

                # state: 7 joint angles (rad)
                if len(fb.actuators) < N_JOINTS:
                    raise RuntimeError(f"Expected {N_JOINTS} actuators, got {len(fb.actuators)}")
                q_rad = [deg2rad(fb.actuators[i].position) for i in range(N_JOINTS)]

                # action: 6D EEF pose (xyz + rpy rad)
                action_6d = [
                    float(fb.base.tool_pose_x),
                    float(fb.base.tool_pose_y),
                    float(fb.base.tool_pose_z),
                    deg2rad(float(fb.base.tool_pose_theta_x)),
                    deg2rad(float(fb.base.tool_pose_theta_y)),
                    deg2rad(float(fb.base.tool_pose_theta_z)),
                ]

                # measured gripper position (normalized [0,1] typically)
                grip_pos = None
                try:
                    meas = base.GetMeasuredGripperMovement(gripper_request)
                    if len(meas.finger) > 0:
                        grip_pos = float(meas.finger[0].value)
                except Exception:
                    grip_pos = None

                # infer gripper command:
                #   1  if grip_pos increases
                #  -1  if grip_pos decreases
                #   0  if it stays (within deadband) or missing
                if grip_pos is None:
                    gripper_cmd = 0
                else:
                    if last_grip is None:
                        gripper_cmd = 0
                    else:
                        d = grip_pos - last_grip
                        if d > GRIPPER_DEADBAND:
                            gripper_cmd = 1
                        elif d < -GRIPPER_DEADBAND:
                            gripper_cmd = -1
                        else:
                            gripper_cmd = 0

                    last_grip = grip_pos

                row = (
                    [t_sec, frame_index, EPISODE_INDEX, TASK_INDEX]
                    + q_rad
                    + action_6d
                    + [gripper_cmd, "" if grip_pos is None else grip_pos]
                )
                w.writerow(row)

                # periodic flush
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_N:
                    f.flush()
                    rows_since_flush = 0

                frame_index += 1

                next_t += period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        print(f"Saved log to: {out_csv}")

    finally:
        safe_disconnect(udp_transport, udp_session)
        safe_disconnect(tcp_transport, tcp_session)


if __name__ == "__main__":
    main()
