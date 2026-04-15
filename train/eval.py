
from unrealcv import Client  
import cv2  
import numpy as np  
import io
import time
import math
import shlex
import subprocess, threading
import traceback
import airsim
from common import *
import psutil
import requests
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import glob
import os, json, sys
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def ue_camera_pose_from_env():
    """Совпадает с scripts/capture_openfly_ue_frames.py — OPENFLY_UE_CAMERA_*."""

    def _f(key, default):
        v = os.environ.get(key, "").strip()
        return float(v) if v else default

    return (
        _f("OPENFLY_UE_CAMERA_X", 150),
        _f("OPENFLY_UE_CAMERA_Y", 400),
        _f("OPENFLY_UE_CAMERA_Z", 15),
        _f("OPENFLY_UE_CAMERA_PITCH", 0),
        _f("OPENFLY_UE_CAMERA_YAW", 0),
        _f("OPENFLY_UE_CAMERA_ROLL", 0),
    )


def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])


def _openfly_ue_attach_only() -> bool:
    """Подключиться к уже запущенному CitySample + UnrealCV (дашборд перезапускается без убийства UE)."""
    return os.environ.get("OPENFLY_UE_ATTACH_ONLY", "").strip().lower() in ("1", "true", "yes", "on")

class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"
    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type = 'color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")

class UEBridge:
    def __init__(self, ue_ip, ue_port, env_name):
        self.env_name = env_name
        self._client = None
        self.process = None
        self._ue_log_fp = None
        self._sim_init_error = None
        self._attach_only = _openfly_ue_attach_only()

        if self._attach_only:
            print(
                "UEBridge: OPENFLY_UE_ATTACH_ONLY=1 — не kill и не Popen CitySample; только UnrealCV.",
                flush=True,
            )
            raw_port = (os.environ.get("OPENFLY_UNREALCV_PORT", "9030") or "9030").strip()
            self._ue_ip = ue_ip
            self._ue_port = int(raw_port)
            # Как во встроенном старте: там перед connect ждут OPENFLY_UE_WARMUP_SEC (90 по умолчанию),
            # пока CitySample поднимает движок и UnrealCV. Раньше attach ждал только 4 с — из-за этого
            # казалось, что «TCP сломался», хотя просто рано дергали connect.
            raw_w = os.environ.get("OPENFLY_UE_ATTACH_WARMUP_SEC", "").strip()
            if raw_w:
                warmup = float(raw_w)
            else:
                warmup = float(os.environ.get("OPENFLY_UE_WARMUP_SEC", "90"))
            print(
                f"Attach: пауза {warmup}s (как OPENFLY_UE_WARMUP_SEC во встроенном режиме), "
                f"затем UnrealCV к {self._ue_ip}:{self._ue_port} "
                f"(дальше OPENFLY_UNREALCV_WAIT_EXTRA_SEC в _connection_check)…",
                flush=True,
            )
            time.sleep(warmup)
        else:
            self.kill_failed_process()
            # Как capture_openfly_ue_frames.py: короткая пауза после kill, затем сразу старт UE.
            time.sleep(float(os.environ.get("OPENFLY_UE_POST_KILL_SEC", "2")))

            port = int((os.environ.get("OPENFLY_UNREALCV_PORT", "9030") or "9030").strip())
            print(
                f"UnrealCV Port={port} (OPENFLY_UNREALCV_PORT, default 9030 — written to unrealcv.ini before UE start)",
                flush=True,
            )
            self.modify_port_in_ini(port, env_name)

            self._ue_ip = ue_ip
            self._ue_port = int(port)

            self._sim_thread = threading.Thread(target=self._wrapped_init_ue_sim, name="uebridge-init-ue")
            self._sim_thread.start()
            warmup = float(os.environ.get("OPENFLY_UE_WARMUP_SEC", "90"))
            print(
                f"Waiting {warmup}s for UE + UnrealCV to listen on port {self._ue_port}...",
                flush=True,
            )
            time.sleep(warmup)

        if self._sim_init_error is not None:
            err = self._sim_init_error
            raise RuntimeError(f"Старт CitySample в фоне упал: {err!r}") from err
        if not self._attach_only and self.process is None:
            raise RuntimeError(
                "Процесс CitySample не создан (Popen не вызван). См. stderr Python и OPENFLY_UE_LOGFILE."
            )

        self._connection_check()

        post_retries = int(os.environ.get("OPENFLY_UE_POST_CONNECT_RETRIES", "2"))
        for n in range(post_retries + 1):
            try:
                self._camera_init()
                self._apply_ue_exposure_mitigation()
                break
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                last_pipe_err = e
                print(
                    f"UnrealCV: обрыв сокета при инициализации камеры/vrun ({type(e).__name__}: {e!r}), "
                    f"переподключение {n + 1}/{post_retries}…",
                    flush=True,
                )
                if n >= post_retries:
                    raise RuntimeError(
                        "UnrealCV разорвал соединение (Broken pipe / reset) на этапе _camera_init или vrun. "
                        "Часто помогает OPENFLY_UE_SKIP_CAMERAS_SPAWN=1 (не вызывать vset /cameras/spawn в City Sample) "
                        "или OPENFLY_UE_VRUN=off. См. лог CitySample."
                    ) from e
                self._reset_unrealcv_client()
                time.sleep(3)
                self._connection_check()

        # self._drone_init()  
        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def find_available_port(self):
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()

            if f'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        with open(ini_file, "r", encoding="utf-8", errors="replace") as file:
            lines = file.readlines()

        with open(ini_file, "w", encoding="utf-8") as file:
            for line in lines:
                s = line.strip()
                if s.upper().startswith("PORT="):
                    file.write(f"Port={port}\n")
                else:
                    file.write(line)
            file.flush()
            os.fsync(file.fileno())

    def _wrapped_init_ue_sim(self) -> None:
        try:
            self._init_ue_sim()
        except Exception as e:
            self._sim_init_error = e
            traceback.print_exc()

    def kill_failed_process(self):
        result = subprocess.run(['pgrep', '-n', 'CrashReport'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

        # Снимаем все CitySample (pgrep -n оставлял зомби на старом порту UnrealCV).
        for _ in range(8):
            result = subprocess.run(
                ["pgrep", "-f", "City_UE52/Binaries/Linux/CitySample"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            pids = [p for p in result.stdout.strip().split() if p.isdigit()]
            if not pids:
                break
            for pid in pids:
                subprocess.run(["kill", "-9", pid], stderr=subprocess.DEVNULL)
            time.sleep(0.5)

    def _init_ue_sim(self):
        env_dir = "envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        env_dir_abs = os.path.abspath(env_dir)
        sh_path = os.path.join(env_dir_abs, "CitySample.sh")
        command = ["bash", sh_path]
        extra_args = os.environ.get("OPENFLY_UE_BINARY_ARGS", "-log").strip()
        if extra_args:
            command.extend(shlex.split(extra_args))
        print("CitySample cmd:", command, "cwd=", env_dir_abs, flush=True)

        # Long-running UE: avoid PIPE (deadlock). Optional log: OPENFLY_UE_LOGFILE=/tmp/ue.log
        # Держим файловый объект на self — иначе после выхода из функции GC может закрыть FD родителя.
        log_path = os.environ.get("OPENFLY_UE_LOGFILE", "").strip()
        if log_path:
            self._ue_log_fp = open(log_path, "ab", buffering=0)
            out, err = self._ue_log_fp, self._ue_log_fp
        else:
            out, err = subprocess.DEVNULL, subprocess.DEVNULL
        self.process = subprocess.Popen(
            command,
            cwd=env_dir_abs,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def __del__(self):
        c = getattr(self, "_client", None)
        if c is not None:
            try:
                c.disconnect()
            except Exception:
                pass

    def _reset_unrealcv_client(self) -> None:
        """После BrokenPipe клиент может оставить «живой» isconnected — сбрасываем перед reconnect."""
        c = getattr(self, "_client", None)
        if c is None:
            return
        try:
            c.disconnect()
        except Exception:
            pass
        self._client = None

    def _connection_check(self):
        """UnrealCV: каждая попытка с новым Client (устойчивее к «зависшему» сокету)."""
        extra = float(os.environ.get("OPENFLY_UNREALCV_WAIT_EXTRA_SEC", "300"))
        deadline = time.time() + extra
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            proc = getattr(self, "process", None)
            if proc is not None and proc.poll() is not None:
                rc = proc.returncode
                raise RuntimeError(
                    f"CitySample завершился до готовности UnrealCV (exit code={rc}). "
                    "См. OPENFLY_UE_LOGFILE, docker logs контейнера, на хосте dmesg при подозрении на OOM."
                )
            c_old = getattr(self, "_client", None)
            if c_old is not None:
                try:
                    c_old.disconnect()
                except Exception:
                    pass
            port_i = int(self._ue_port)
            unix_path = f"/tmp/unrealcv_{port_i}.socket"
            tcp_only = os.environ.get("OPENFLY_UNREALCV_TCP_ONLY", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            # UE 5.x UnrealCV часто поднимает UDS; для встроенного старта UE без attach TCP иногда «молчит».
            # В OPENFLY_UE_ATTACH_ONLY файл сокета может остаться от старого процесса — unrealcv Client(..., 'unix')
            # .connect() тогда долго висит, хотя TCP на порту жив. По умолчанию attach → TCP; UDS явно:
            # OPENFLY_UNREALCV_UDS_ON_ATTACH=1 или OPENFLY_UNREALCV_TCP_ONLY=0 и без attach.
            use_uds = (
                not tcp_only
                and sys.platform.startswith("linux")
                and os.path.exists(unix_path)
            )
            if use_uds and _openfly_ue_attach_only():
                uds_on = os.environ.get("OPENFLY_UNREALCV_UDS_ON_ATTACH", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if not uds_on:
                    if attempt == 1:
                        print(
                            "UnrealCV: OPENFLY_UE_ATTACH_ONLY — подключение по TCP (не UDS), чтобы не "
                            "зависать на устаревшем unix-сокете. Для UDS: OPENFLY_UNREALCV_UDS_ON_ATTACH=1.",
                            flush=True,
                        )
                    use_uds = False
            if use_uds:
                self._client = Client(unix_path, "unix")
                if attempt == 1 or attempt % 10 == 0:
                    print(f"UnrealCV: пробуем Unix socket {unix_path!r}", flush=True)
            else:
                self._client = Client((self._ue_ip, port_i))
                if attempt == 1 or attempt % 10 == 0:
                    print(f"UnrealCV: пробуем TCP {self._ue_ip!r}:{port_i}", flush=True)
            if self._client.connect():
                print("UnrealCV connected successfully", flush=True)
                return
            if attempt == 1 or attempt % 10 == 0:
                print(
                    f"UnrealCV not ready yet (attempt {attempt}), retrying for up to {extra:.0f}s...",
                    flush=True,
                )
            time.sleep(5)
        print("UnrealCV is not connected (timeout)", flush=True)
        raise SystemExit(1)

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        '''Set camera position'''
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        print('camera_settings', camera_settings)

    def _camera_init(self):
        '''Camera initialization'''
        time.sleep(2)
        skip_spawn = os.environ.get("OPENFLY_UE_SKIP_CAMERAS_SPAWN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not skip_spawn:
            self._client.request("vset /cameras/spawn")
        else:
            print("OPENFLY_UE_SKIP_CAMERAS_SPAWN=1 — пропуск vset /cameras/spawn", flush=True)
        self._client.request("vset /camera/1/size 1920 1080")
        time.sleep(2)
        x, y, z, pitch, yaw, roll = ue_camera_pose_from_env()
        self.set_camera_pose(x, y, z, pitch, yaw, roll)
        time.sleep(2)

    def _apply_ue_exposure_mitigation(self):
        """Поджать пересвет (City Sample + UnrealCV / lit). Команды UE через vrun.

        Отключить: OPENFLY_UE_VRUN=off
        Свой список (через ';', без префикса vrun). В UE 5.2 City Sample r.BloomIntensity часто не распознаётся — см. capture_openfly_ue_frames.py.
        """
        flag = os.environ.get("OPENFLY_UE_VRUN", "").strip()
        if flag.lower() in ("0", "off", "false", "no"):
            return
        if flag:
            cmds = [c.strip() for c in flag.split(";") if c.strip()]
        else:
            # Согласовано с scripts/capture_openfly_ue_frames.py (без r.BloomIntensity в UE 5.2)
            cmds = [
                "r.DefaultFeature.AutoExposure.Bias -3.0",
                "r.BloomQuality 2",
                "r.DefaultFeature.AutoExposure 1",
            ]
        for cmd in cmds:
            req = cmd if cmd.startswith("vrun ") else f"vrun {cmd}"
            try:
                self._client.request(req)
                print("UE vrun:", req, flush=True)
            except Exception as e:
                print("UE vrun skipped:", req, repr(e), flush=True)
        time.sleep(1.5)

    def get_camera_data(self, camera_type = 'lit'):
        valid_types = {'lit', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'lit':
            data = self._client.request('vget /camera/1/lit png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'object_mask':
            data = self._client.request('vget /camera/1/object_mask png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'depth':
            data = self._client.request('vget /camera/1/depth npy')
            depth_np = np.load(io.BytesIO(data))
            return depth_np  # Return depth data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='lit'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)

class GSBridge:  
    def __init__(self, env_name):  
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def _init_gs_sim(self):
        # dataset_dir = "envs/gs/" + self.env_name  
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"  
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw
    
    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text) 
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass



def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result

def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):

    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)
        
    prompt = text
    inputs = processor(prompt, images).to("cuda:0")
    # Нельзя .to(dtype=bfloat16) на весь BatchFeature — input_ids должны остаться long (иначе ломается causal mask в generate).
    for _k, _v in list(inputs.items()):
        if isinstance(_v, torch.Tensor) and _v.is_floating_point():
            inputs[_k] = _v.to(dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]

def main():
    eval_info = os.environ.get("OPENFLY_EVAL_JSON", "configs/eval_test.json")

    f = open(eval_info, 'r')
    all_eval_info = json.loads(f.read())
    f.close()
    
    # Load model (local dir or HF id; OPENFLY_MODEL overrides default)
    model_name_or_path = os.environ.get(
        "OPENFLY_MODEL", "IPEC-COMMUNITY/openfly-agent-7b"
    )
    processor = AutoProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    load_kw = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    attn_impl = os.environ.get(
        "OPENFLY_ATTN_IMPLEMENTATION", "flash_attention_2"
    )
    try:
        policy = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_impl,
            **load_kw,
        ).to("cuda:0")
    except Exception as e:
        print(
            "Model load failed with attn_implementation=",
            attn_impl,
            "(",
            repr(e),
            "); retrying eager",
            flush=True,
        )
        policy = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            attn_implementation="eager",
            **load_kw,
        ).to("cuda:0")

    # Test metrics
    acc = 0
    stop = 0
    data_num = 0
    MAX_STEP = int(os.environ.get("OPENFLY_MAX_STEPS", "100"))

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)
    
    # Process each environment type sequentially
    for env_name, eval_info in env_groups.items():
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_info)} data entries")
        time.sleep(5)
        
        # Create appropriate environment bridge based on environment type
        if "airsim" in env_name:
            env_bridge = AirsimBridge(env_name)
            pos_ratio = 1.0
        elif "ue" in env_name:
            env_bridge = UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name)
            pos_ratio = 1.0
        elif "gs" in env_name:
            env_bridge = GSBridge(env_name)
            pos_ratio = 5.15
        else:
            print(f"Unknown environment type: {env_name}, skipping")
            continue
        
        # Evaluate all data for current environment
        for idx, item in enumerate(eval_info):
            acts = []  # Reset action list
            data_num += 1
            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")

            save_frames = os.environ.get("OPENFLY_SAVE_ACTION_FRAMES", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            rollout_base = os.environ.get("OPENFLY_ROLLOUT_BASE", "").strip()
            if save_frames and rollout_base:
                slug = item["image_path"].replace("/", "__")
                frame_dir = os.path.join(rollout_base, f"{idx:02d}_{slug}")
                os.makedirs(frame_dir, exist_ok=True)
                with open(os.path.join(frame_dir, "instruction.txt"), "w", encoding="utf-8") as tf:
                    tf.write(text)
                with open(os.path.join(frame_dir, "episode.json"), "w", encoding="utf-8") as jf:
                    json.dump(
                        {
                            "image_path": item.get("image_path"),
                            "start_xyz": list(start_postion),
                            "start_yaw_rad": float(start_yaw),
                            "goal_xyz": list(end_position),
                        },
                        jf,
                        indent=2,
                    )
            else:
                frame_dir = (
                    os.environ.get("OPENFLY_ACTION_FRAMES_DIR", "test/action_frames").strip()
                    or "test/action_frames"
                )
            
            stop_error = 1
            image_error = False
            
            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_postion[0]/pos_ratio, 
                start_postion[1]/pos_ratio, 
                start_postion[2]/pos_ratio, 
                pitch, 
                np.rad2deg(start_yaw), 
                0
            )
            
            step = 0
            flag_osr = 0
            image_list = []
            env_bridge.pass_len = 1e-3
            old_pose = new_pose
            
            while step < MAX_STEP:
                try:
                    raw_image = env_bridge.get_camera_data()
                    cv2.imwrite("test/cur_img.jpg", raw_image)
                    if save_frames:
                        fd = frame_dir
                        os.makedirs(fd, exist_ok=True)
                        cv2.imwrite(os.path.join(fd, f"step_{step:04d}.png"), raw_image)
                    image = raw_image
                    
                    image_list.append(image)
                    model_action = get_action(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    acts.append(model_action)
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0]/pos_ratio, 
                        new_pose[1]/pos_ratio, 
                        new_pose[2]/pos_ratio, 
                        pitch, 
                        np.rad2deg(new_pose[3]), 
                        0
                    )
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose

                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_postion)
            env_bridge.distance_to_goal.append(dis)
            if dis < 20:
                env_bridge.success.append(1)
                env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)
                acc += 1
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
            if flag_osr == 0:
                env_bridge.osr.append(0)
            env_bridge.print_info()

            if save_frames and rollout_base and os.path.isdir(frame_dir):
                fps = float(os.environ.get("OPENFLY_ROLLOUT_FPS", "4"))
                mp4_path = os.path.join(frame_dir, "rollout.mp4")
                pat = os.path.join(frame_dir, "step_*.png")

                if glob.glob(pat):
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        str(fps),
                        "-i",
                        os.path.join(frame_dir, "step_%04d.png"),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        mp4_path,
                    ]
                    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if r.returncode == 0:
                        print(f"Wrote rollout video: {os.path.abspath(mp4_path)}", flush=True)
                    else:
                        print("ffmpeg failed:", r.stderr[:500], flush=True)

            if image_error:
                continue
                
        
        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        del env_bridge
        import gc
        gc.collect()
    
    # Final results
    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = 1 - stop / data_num if data_num > 0 else 0
    
    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final stop rate: {final_stop:.4f}")


if __name__ == '__main__':
    main()
