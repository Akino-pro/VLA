import csv
import time
from datetime import datetime

from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2

UDP_PORT = 10001  # Gen3 cyclic feedback port


def connect_udp(ip: str, username: str, password: str):
    transport = UDPTransport()
    transport.connect(ip, UDP_PORT)

    router = RouterClient(transport, RouterClient.basicErrorCallback)

    session_manager = SessionManager(router)
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = username
    session_info.password = password
    session_info.session_inactivity_timeout = 60000   # ms
    session_info.connection_inactivity_timeout = 2000 # ms
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


def main():
    # --------- EDIT THESE ----------
    ROBOT_IP = "192.168.1.10"  # change to your Gen3 IP
    USERNAME = "admin"
    PASSWORD = "admin"

    SAMPLE_HZ = 1            # logging rate
    DURATION_S = 10          # total seconds to record
    N_JOINTS = 7             # Gen3 has 7 joints
    # ------------------------------

    out_csv = f"tool_pose_6d_joints_7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    period = 1.0 / float(SAMPLE_HZ)

    transport, session_manager, router = connect_udp(ROBOT_IP, USERNAME, PASSWORD)

    try:
        base_cyclic = BaseCyclicClient(router)

        t0 = time.perf_counter()
        next_t = t0

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)

            header = [
                "t_sec",
                "x_m", "y_m", "z_m",
                "theta_x_deg", "theta_y_deg", "theta_z_deg",
            ] + [f"q{i+1}_deg" for i in range(N_JOINTS)]
            w.writerow(header)

            while True:
                now = time.perf_counter()
                t = now - t0
                if t >= DURATION_S:
                    break

                fb = base_cyclic.RefreshFeedback()

                # Tool pose (6D)
                row = [
                    t,
                    fb.base.tool_pose_x,
                    fb.base.tool_pose_y,
                    fb.base.tool_pose_z,
                    fb.base.tool_pose_theta_x,
                    fb.base.tool_pose_theta_y,
                    fb.base.tool_pose_theta_z,
                ]

                # Joint angles (7)
                # Cyclic feedback provides actuator list; each has .position
                # Gen3 typically reports actuator position in degrees.
                if len(fb.actuators) < N_JOINTS:
                    raise RuntimeError(
                        f"Expected at least {N_JOINTS} actuators, but got {len(fb.actuators)}"
                    )

                for i in range(N_JOINTS):
                    row.append(fb.actuators[i].position)

                w.writerow(row)

                # simple rate control
                next_t += period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        print(f"Saved tool pose + joint log to: {out_csv}")

    finally:
        safe_disconnect(transport, session_manager)


if __name__ == "__main__":
    main()
