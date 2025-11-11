# -*- coding: utf-8 -*-
import os
import sys
import cv2
import yaml
import time
from threading import Lock
import inspect
import traceback
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
                            QGroupBox, QGridLayout, QComboBox)
from PyQt6.QtGui import QImage, QPixmap, QFont, QFontDatabase, QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import torch
from functools import wraps

try:
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    from airbot_camera import RealsenseCamera
    from airbot_segment import AirbotSegment, SegmentMode
    from airbot_grasp_simple import SimpleGrasp
    from airbot_yolo import AirbotYolo
except ImportError as e:
    print(f"Failed to import airbot modules, {e}")
    exit()
    
# temporary delay caused by moveit early return and gripper no blocking
arm_delay = 0.3 # second
gripper_delay = 0.3

def safe_func():
    def decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())[1:]
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                param_cnt = len(param_names)
                if param_cnt == 0:
                    return func(self)
                else:
                    return func(self, *args, **kwargs)
            except Exception as e:
                detail = traceback.format_exc()
                error_info = f"{func.__name__} with params: [{param_names}] error"
                self.log(f"{error_info}: {e}\n{detail}")
        return wrapper
    return decorator

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int, int)  # x, y, button
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        button = 0
        if event.button() == Qt.MouseButton.LeftButton:
            button = 0
        elif event.button() == Qt.MouseButton.RightButton:
            button = 1
        elif event.button() == Qt.MouseButton.MiddleButton:
            button = 2
            
        pos = event.position().toPoint()  # 转为 QPoint（整数）
        self.clicked.emit(pos.x(), pos.y(), button)
        
class ObserveThread(QThread):
    update_frame = pyqtSignal(np.ndarray, dict)
    
    def __init__(self, interface, robot_port, camera):
        super().__init__()
        self.interface = interface
        self.robot = AIRBOTPlay(port=robot_port)
        self.robot.connect()
        self.camera = camera
        
    def run(self):
        while not QThread.currentThread().isInterruptionRequested():
            color_frame = None
            depth_frame = None
            depth_map = None
            try:
                color_frame, depth_frame, depth_map = self.camera.get_frame(frame_type=["bgr", "depth", "depth_map"], align=True)
            except Exception as e:
                if "Device disconnected" in str(e):
                    print("Reconnecting to the camera...")
                    del self.interface.realsense
                    self.interface.realsense = RealsenseCamera()
                    self.camera = self.interface.realsense
           
            pose = [[],[]]
            joints = []
            eef = []
            eeff = []
            
            try:
                pose = self.robot.get_end_pose()
                joints = self.robot.get_joint_pos()
                eef = self.robot.get_eef_pos()
                eeff = self.robot.get_eef_eff()
            except Exception as e:
                if "Not connected to the server" in str(e):
                    print("Reconnecting to the server...")
                    self.robot.connect()
            
            if color_frame is None or depth_frame is None or depth_map is None:
                color_frame = np.zeros_like((self.camera.HEIGHT,self.camera.WIDTH))
                depth_frame = np.zeros_like((self.camera.HEIGHT,self.camera.WIDTH))
                depth_map = np.zeros_like((self.camera.HEIGHT,self.camera.WIDTH))
                
            with self.interface.observe_frame_lock:
                self.interface.color_frame = color_frame.copy()
                self.interface.depth_frame = depth_frame.copy()
                self.interface.depth_map = depth_map.copy()
                self.interface.current_state["trans"] = pose[0]
                self.interface.current_state["orient"] = pose[1]
                self.interface.current_state["joints"] = joints
                self.interface.current_state["eef"] = eef
                self.interface.current_state["eeff"] = eeff
            self.update_frame.emit(color_frame.copy(), {"trans": pose[0], "orient": pose[1], "joints": joints, "eef": eef, "eeff": eeff})
    
    def stop(self):
        self.requestInterruption()
        self.wait()
        self.robot.disconnect()
        self.camera.deinit()

class RobotGraspThread(QThread):
    log = pyqtSignal(str)
    capture_signal = pyqtSignal()
    
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.robot = AIRBOTPlay(port=self.interface.robot_port)
        self.robot.connect()
        self.running_lock = Lock()
        self.predicted_info = None
    
    def set_predicted_info(self, predicted_info):
        with self.running_lock:
            self.predicted_info = predicted_info
    
    def run(self):
        with self.running_lock:
            trans = self.predicted_info["trans"]
            orient = self.predicted_info["orient"]
            o_height = self.predicted_info["o_height"]
            o_width = self.predicted_info["o_width"]
            try:
                with open("waypoints.txt", "a") as f:
                    self.robot.switch_mode(RobotMode.PLANNING_POS)
                                        
                    # 先移动到抓取位置
                    self.log.emit("步骤1: 打开夹爪并移动到抓取位置")
                    self.robot.move_eef_pos(o_width*1.8)  # 打开夹爪
                    time.sleep(0.3)
                    self.interface.gripper_opened = True
                    
                    # # move to method
                    # #-------------------------------
                    # res = self.robot.move_to_cart_pose(
                    #     [[trans[0], trans[1], trans[2] + o_height * 2], orient]
                    # )
                    # if res is None:
                    #     self.log("无法移动到预测位姿: ", [[trans[0], trans[1], trans[2] + o_height * 2], orient])
                    #     return
                    # time.sleep(arm_delay)
                    # res = self.robot.move_to_cart_pose([trans, orient])
                    # if res is None:
                    #     self.log("无法移动到预测位姿: ", [trans, orient])
                    #     return
                    # time.sleep(arm_delay)
                    # #-------------------------------
                    
                    
                    # move with waypoints method
                    #-------------------------------
                    waypoints = [
                        [[trans[0], trans[1], trans[2]+0.1], orient], 
                        [trans, orient]
                    ]
                    for waypoint in waypoints:
                        f.write(str(waypoint) + ",\n")
                    self.robot.switch_mode(RobotMode.PLANNING_WAYPOINTS)                
                    res = self.robot.move_with_cart_waypoints(waypoints=waypoints)
                    if res is None:
                        self.log.emit(f"无法移动到预测位姿: {waypoints}")
                        return
                    time.sleep(arm_delay)
                    #-------------------------------
                    
                    
                    # 关闭夹爪
                    self.log.emit("步骤2: 关闭夹爪抓取物体")
                    self.robot.move_eef_pos(0)
                    time.sleep(gripper_delay)
                    self.interface.gripper_opened = False
                                        
                    # 抬起物体
                    self.log.emit("步骤3: 抬起物体")
                    
                    # # # move to method
                    # # # -------------------------------
                    # # res = self.robot.move_to_cart_pose(
                    # #     [[trans[0], trans[1], trans[2] + 0.2], orient]
                    # # )
                    # # if res is None:
                    # #     self.log("无法移动到预测位姿: ", [[trans[0], trans[1], trans[2] + 0.2], orient])
                    # #     return

                    # time.sleep(arm_delay)
                    # self.log.emit("步骤4: 移动到放置位置")
                    # self.robot.move_to_cart_pose(self.interface.pre_place_pose)
                    # time.sleep(arm_delay)
                    # self.robot.move_to_cart_pose(self.interface.place_pose) 
                    # time.sleep(arm_delay)
                    # # -------------------------------
                    
                    # move with waypoints method
                    #-------------------------------
                    waypoints = [
                        [[trans[0], trans[1], trans[2] + 0.1], orient],
                        # self.interface.pre_place_pose,
                        self.interface.place_pose
                    ]
                    for waypoint in waypoints:
                        f.write(str(waypoint) + ",\n")
                    self.robot.switch_mode(RobotMode.PLANNING_WAYPOINTS)  
                    res = self.robot.move_with_cart_waypoints(waypoints)
                    if res is None:
                        self.log.emit(f"无法移动到预测位姿: {waypoints}")
                        return
                    #-------------------------------
                                        
                    # 松开夹爪
                    self.log.emit("步骤5: 松开夹爪释放物体")
                    self.robot.move_eef_pos(1)
                    time.sleep(gripper_delay)
                    self.interface.gripper_opened = True
                                        
                    # 回到观察位置
                    self.log.emit("步骤6: 返回观察位置")
                    self.robot.switch_mode(RobotMode.PLANNING_POS)
                    f.write(str(self.interface.observe_pose) + ",\n")
                    self.robot.move_to_cart_pose(self.interface.observe_pose)
                    time.sleep(arm_delay)
                                            
                    self.log.emit("抓取执行完成")
                    self.capture_signal.emit()
                    
            except Exception as e:
                detail = traceback.format_exc()
                print(f"Failed to perform grasp: {e}\n{detail}")

class AutoGraspThread(QThread):
    finish_signal = pyqtSignal()
    capture_signal = pyqtSignal()
    update_info = pyqtSignal(dict)
    update_image = pyqtSignal(np.ndarray)
    log = pyqtSignal(str)
    
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.grasping_object = None
        self.grasping_mask = None
        self.airbot_segment = self.interface.airbot_segment
        self.airbot_yolo = self.interface.airbot_yolo
        self.realsense = self.interface.realsense
        self.airbot_grasp = self.interface.airbot_grasp
        self.finish_cnt = 0

    def run(self):
        self.finish_cnt = 0
        with self.interface.predict_lock:
            self.interface.predicted_info = None
        self.log.emit("开始自动抓取")
        # 捕获图像     
        self.capture_signal.emit()
        time.sleep(0.1)
        while not QThread.currentThread().isInterruptionRequested():
            try:                                              
                with self.interface.captured_frame_lock:
                    captured_color = self.interface.captured_color
                    captured_depth = self.interface.captured_depth
                    captured_pose = self.interface.captured_pose
                
                
                # 获取边界框和标签
                object_bbox, object_label, object_conf = self.airbot_yolo.get_max_conf_bbox_and_label(captured_color, label_filter=[self.grasping_object, "gripper"])
                if object_bbox is None or len(object_bbox) == 0:
                    self.log.emit("未检测到物体，跳过当前循环")
                    self.capture_signal.emit()
                    if self.interface.robot_grasp_thread.isRunning():
                        self.interface.robot_grasp_thread.requestInterruption()
                        self.interface.robot_grasp_thread.wait()
                    time.sleep(1)
                    self.grasping_mask = None
                    self.grasping_object = None
                    self.finish_cnt += 1
                    if self.finish_cnt == self.interface.auto_stop_cnt:
                        self.finish_cnt = 0
                        self.log.emit("自动抓取结束，切换到手动选择模式")
                        self.finish_signal.emit()
                        return

                    continue
                
                self.airbot_segment.add_bbox(object_bbox)
                mask = self.airbot_segment.inference(captured_color)
                
                # 创建掩码预览
                masked_color = captured_color.copy()
                masked_color[mask] = [191, 214, 238]  # 预览掩码
                # cv2.imwrite("mask_color.png", masked_color)
                if self.grasping_mask is not None:
                    masked_color[self.grasping_mask] = [0, 255, 0]
                
                x1, y1, x2, y2 = object_bbox

                cv2.rectangle(masked_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    masked_color,
                    f"{object_label} {object_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                self.update_image.emit(masked_color)
                
                self.grasping_mask = mask
                self.grasping_object = object_label
                
                cloud = self.realsense.create_point_cloud(captured_depth)
                if cloud.shape[0] == 0:
                    self.log.emit("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.capture_signal.emit()
                    if self.interface.robot_grasp_thread.isRunning():
                        self.interface.robot_grasp_thread.requestInterruption()
                        self.interface.robot_grasp_thread.wait()
                    time.sleep(1)
                    self.grasping_mask = None
                    self.grasping_object = None
                    self.finish_cnt += 1
                    if self.finish_cnt == self.interface.auto_stop_cnt:
                        self.finish_cnt = 0
                        self.log.emit("自动抓取结束，切换到手动选择模式")
                        self.finish_signal.emit()
                        return
                    continue
                trans, orient, cloud_base = self.airbot_grasp.inference(
                    color_image=captured_color, 
                    depth_image=captured_depth, 
                    end_pose=captured_pose, 
                    cloud_cam_raw=cloud,
                    mask=mask
                )
                if trans is None or orient is None or cloud_base is None:
                    self.log.emit("未检测到物体，跳过当前循环")
                    self.capture_signal.emit()
                    if self.interface.robot_grasp_thread.isRunning():
                        self.interface.robot_grasp_thread.requestInterruption()
                        self.interface.robot_grasp_thread.wait()
                    time.sleep(1)
                    self.grasping_mask = None
                    self.grasping_object = None
                    self.finish_cnt += 1
                    if self.finish_cnt == self.interface.auto_stop_cnt:
                        self.finish_cnt = 0
                        self.log.emit("自动抓取结束，切换到手动选择模式")
                        self.finish_signal.emit()
                        return
                    continue
                
                object_height = np.max(cloud_base[:,2])
                f = (self.realsense.intrinsic[0][0] + self.realsense.intrinsic[1][1]) / 2
                y_indices, x_indices = np.where(mask)
                mask_center_y = int(np.mean(y_indices))
                mask_center_x = int(np.mean(x_indices))
                distance = captured_depth[mask_center_y][mask_center_x]
                object_width = self.airbot_grasp.length_minor * (distance / self.realsense.depth_factor) / f
                pbject_angle = self.airbot_grasp.angle
                
                predicted_info = {
                    "trans": trans.tolist(),
                    "orient": orient.tolist(),
                    "o_height": object_height,
                    "o_width": object_width,
                    "o_angle": pbject_angle,
                    "o_label": object_label,
                    "o_bbox": object_bbox
                }
                
                with self.interface.predict_lock:
                    self.interface.predicted_info = predicted_info.copy()
                
                self.update_info.emit(predicted_info)
                
                if self.interface.dump:
                    import json
                    with self.interface.captured_frame_lock:
                        dump_info = {
                            "observe": self.interface.capture_state,
                            "predicted": predicted_info
                        }
                    file_name = time.strftime("%Y%m%d%H%M%S", time.localtime()) + "_info.json"
                    file_path = os.path.join(self.interface.dump_path, file_name)
                    with open(file_path, "w") as f:
                        json.dump(dump_info, f, indent=4)

                # 等待完成上一次抓取
                self.interface.robot_grasp_thread.wait()            
                self.interface.robot_grasp_thread.set_predicted_info(predicted_info)
                self.interface.robot_grasp_thread.start() 
                
            except Exception as e:
                error_info = traceback.format_exc()
                self.log.emit(f"自动抓取异常: {str(e)}\n详细信息:\n{error_info}")
                self.capture_signal.emit()
                time.sleep(1)
        
        self.capture_signal.emit()
    
    def stop(self):
        self.wait()

class AirbotControlInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        config = yaml.safe_load(open(config_path, "r"))
        
        # 初始化变量
        self.color_frame = None
        self.depth_frame = None
        self.depth_map = None
        self.current_state = {"trans": None, "orient": None, "joints": None, "eef": None, "eeff": None}
        self.observe_frame_lock = Lock()
        
        self.captured_color = None
        self.captured_depth = None
        self.captured_depth_map = None
        self.captured_pose = {"trans": None, "orient": None}
        self.captured_state = {"trans": None, "orient": None, "joints": None, "eef": None, "eeff": None}
        self.mask_map = None
        self.captured_frame_lock = Lock()
        
        self.predicted_info = {"trans": None, "orient": None, "o_height": None, "o_width": None, "o_angle": None, "o_label": None, "o_bbox": None}
        self.predict_lock = Lock()
        
        # 初始化相机和分割模块
        self.realsense = RealsenseCamera()
        self.frame_width = self.realsense.WIDTH
        self.frame_height = self.realsense.HEIGHT
        self.airbot_segment = AirbotSegment()
        self.airbot_segment.clear_prompt()
        self.airbot_grasp = SimpleGrasp()
        self.airbot_yolo = AirbotYolo()
        
        # 视频流标志
        self.running = True
        
        # 初始化GUI
        if self.realsense.resolution == "480p":
            factor = 1
        else:
            factor = 2
        self.video_width = int(self.frame_width/factor)
        self.video_height = int(self.frame_height/factor)
        self.init_ui()
        
        # 启动相机线程
        self.gripper_opened = True
        self.robot_speed = SpeedProfile.DEFAULT
        self.robot_port = config["ArmParams"]["port"]
        self.observe_thread = ObserveThread(self, self.robot_port, self.realsense)
        self.observe_thread.update_frame.connect(self.update_observe_frame)
        self.observe_thread.start()
        
        # 初始化抓取线程和自动抓取线程
        self.auto_mode = False
        self.robot_grasp_thread = RobotGraspThread(self)
        self.robot_grasp_thread.log.connect(self.log)
        self.robot_grasp_thread.capture_signal.connect(self.capture_frame)
        self.auto_grasp_thread = AutoGraspThread(self)
        self.auto_grasp_thread.finish_signal.connect(self.change_auto_mode)
        self.auto_grasp_thread.capture_signal.connect(self.capture_frame)
        self.auto_grasp_thread.update_image.connect(lambda image: self.update_capture_view(image))
        self.auto_grasp_thread.update_info.connect(self.update_info)
        self.auto_grasp_thread.log.connect(self.log)
        
        # 加载运行时参数
        self.observe_pose = config["AirbotGrasp"]["observe_pose"]
        self.place_pose = config["AirbotGrasp"]["place_pose"]
        self.pre_place_pose = config["AirbotGrasp"]["pre_place_pose"]
        
        # 日志
        self.dump = config["RunTime"]["dump"]
        self.dump_path = config["RunTime"]["dump_path"]
        self.auto_stop_cnt = config["RunTime"]["auto_stop_cnt"]
        os.makedirs(self.dump_path, exist_ok=True)
        
        # 初始化robot
        self.od_arm_safe_threshold = config["ArmParams"]["od_arm_safe_threshold"]
        self.dm_arm_safe_threshold = config["ArmParams"]["dm_arm_safe_threshold"]
        self.gripper_safe_threshold = config["ArmParams"]["gripper_safe_threshold"]
        self.move_to_observe()
    
    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("AIRBOT AI Grasp")
        self.setMinimumSize(1280, 720)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 创建左侧相机视图组
        camera_group = QVBoxLayout()
        
        # 实时相机预览
        self.camera_view_group = QGroupBox("Camera Video")
        camera_view_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(self.video_width, self.video_height)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        camera_view_layout.addWidget(self.camera_label)
        self.camera_view_group.setLayout(camera_view_layout)
        
        # 捕获和分割视图
        self.captured_view_group = QGroupBox("Capture/Segmentation")
        captured_view_layout = QVBoxLayout()
        self.capture_instruction = QLabel("待捕获图像帧")
        self.capture_instruction.setFixedHeight(20)
        self.captured_label = ClickableLabel()
        self.captured_label.setFixedSize(self.video_width, self.video_height)
        self.captured_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captured_label.setStyleSheet("background-color: black;")
        self.captured_label.clicked.connect(self.handle_segmentation_click)
        captured_view_layout.addWidget(self.capture_instruction)
        captured_view_layout.addWidget(self.captured_label)
        self.captured_view_group.setLayout(captured_view_layout)
        
        # 添加到左侧布局
        camera_group.addWidget(self.camera_view_group)
        camera_group.addWidget(self.captured_view_group)
        
        # 创建右侧控制面板
        control_panel = QVBoxLayout()
        
        # 机械臂状态信息
        self.info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()
        
        # 当前位姿
        info_layout.addWidget(QLabel("Robot Current State:"))
        self.current_state_text = QTextEdit()
        self.current_state_text.setReadOnly(True)
        self.current_state_text.setFixedHeight(75)
        info_layout.addWidget(self.current_state_text)
        
        # 预测位姿
        info_layout.addWidget(QLabel("Robot Predicted Pose:"))
        self.predicted_info_text = QTextEdit()
        self.predicted_info_text.setReadOnly(True)
        self.predicted_info_text.setFixedHeight(50)
        info_layout.addWidget(self.predicted_info_text)
        
        # 物体信息
        info_layout.addWidget(QLabel("Detected Object Info:"))
        self.object_info_text = QTextEdit()
        self.object_info_text.setReadOnly(True)
        self.object_info_text.setFixedHeight(100)
        info_layout.addWidget(self.object_info_text)
        
        self.info_group.setLayout(info_layout)
        
        # 控制控制台
        self.button_group = QGroupBox("Console")
        button_layout = QGridLayout()
        
        # 创建按钮
        self.capture_btn = QPushButton("Capture")
        self.gravity_btn = QPushButton("Grivity Compensation")
        self.observe_btn = QPushButton("Move to Observe pose")
        self.predict_btn = QPushButton("Move to Predicted pose")
        self.grasp_btn = QPushButton("Pick and Place")
        self.gripper_btn = QPushButton("Trigger Gripper")
        self.set_place_btn = QPushButton("Set Place Pose")
        self.auto_btn = QPushButton("Auto Grasp Mode")
       
        button_set = {
            'capture_btn': (0, 0),
            'gravity_btn': (0, 1),
            'observe_btn': (1, 0),
            'predict_btn': (1, 1),
            'grasp_btn': (2, 0),
            'gripper_btn': (2, 1),
            'set_place_btn': (3, 0),
            'auto_btn': (3, 1),
        }
        
        # 设置按钮样式和大小
        self.button_style = """
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6aa2ff;
            }
            QPushButton:pressed {
                background-color: #3a76d8;
            }
        """
        
        self.button_style_red = """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6655;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """
        
        self.button_style_grey = """
            QPushButton {
                background-color: #858b93;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #858b93;
            }
            QPushButton:pressed {
                background-color: #858b93;
            }
        """
        
        for btn_name in button_set.keys():
            btn = getattr(self, btn_name)
            btn.setStyleSheet(self.button_style)
            btn.setMinimumHeight(60)
            
        # 添加按钮到布局
        for btn_name, pos in button_set.items():
            btn = getattr(self, btn_name)
            button_layout.addWidget(btn, pos[0], pos[1])
            
        # 连接按钮信号和槽
        self.capture_btn.clicked.connect(self.capture_frame)
        self.gravity_btn.clicked.connect(self.trigger_gravity)
        self.observe_btn.clicked.connect(self.move_to_observe)
        self.predict_btn.clicked.connect(self.move_to_predicted)
        self.grasp_btn.clicked.connect(self.predict_and_grasp)
        self.gripper_btn.clicked.connect(self.trigger_gripper)
        self.set_place_btn.clicked.connect(self.set_place_pose)
        self.auto_btn.clicked.connect(self.change_auto_mode)
        
        self.button_group.setLayout(button_layout)
        
        # 创建下拉菜单
        self.select_group = QGroupBox("Select")
        select_layout = QVBoxLayout()
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Robot Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["SLOW", "DEFAULT", "MEDIUM", "FAST"])
        self.speed_combo.setCurrentText("DEFAULT")
        self.speed_combo.currentTextChanged.connect(self.change_speed_profile)
        speed_layout.addWidget(self.speed_combo)
        select_layout.addLayout(speed_layout)
        
        
        self.select_group.setLayout(select_layout)
        
        # 日志输出区
        self.log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output, 5)
        
        self.log_group.setLayout(log_layout)
        

        
        # 添加到右侧布局
        control_panel.addWidget(self.info_group)
        control_panel.addWidget(self.button_group)
        control_panel.addWidget(self.select_group)
        control_panel.addWidget(self.log_group)
        
        # 构建主布局
        left_widget = QWidget()
        left_widget.setLayout(camera_group)
        
        right_widget = QWidget()
        right_widget.setLayout(control_panel)
        
      # 设置主窗口部件
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        self.setCentralWidget(main_widget)
        
        # 美化UI
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 16px;
                color: #000000; 
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #000000; 
            }
            QLabel {
                font-size: 14px;
                color: #000000; 
            }
            QTextEdit {
                font-size: 12px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
                color: #000000; 
            }            
            QComboBox QAbstractItemView {
                font-size: 12px;
                background-color: #3d3d3d;
                color: #ffffff;
                selection-background-color: #555555;
                selection-color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.lightGray)
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.lightGray)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Link, Qt.GlobalColor.blue)
        palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.blue)
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
        self.setPalette(palette)
        
        # 记录日志
        self.log("AIRBOT AI Grasp 界面已初始化")
    
    @pyqtSlot(str)
    @safe_func()
    def log(self, message):
        """将消息添加到日志输出区域"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_output.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")
    
    def qpixmap_from_image(self, frame):
        resized_frame = cv2.resize(frame, (self.video_width, self.video_height))
        h, w, ch = resized_frame.shape
        qt_image = QImage(resized_frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        return pixmap
    @pyqtSlot(np.ndarray, dict)
    @safe_func()
    def update_observe_frame(self, frame, pose):
        """更新观测"""
        self.camera_label.setPixmap(self.qpixmap_from_image(frame))
        trans_str = ", ".join([f"{v:.6f}" for v in pose["trans"]])
        orient_str = ", ".join([f"{v:.6f}" for v in pose["orient"]])
        joints_str = ", ".join([f"{v:.6f}" for v in pose["joints"]])
        
        eef_pos_str = ", ".join([f"{v:.2f}" for v in pose["eef"]])
        eef_eff_str = ", ".join([f"{v:.2f}" for v in pose["eeff"]])
        
        
        refer_pose = [[0.188852, -0.005085, 0.259439], [-0.001318, 0.554186, -0.015383, 0.832249]]
        # error = np.linalg.norm(np.array(pose["trans"]) - np.array(refer_pose[0])) + np.linalg.norm(np.array(pose["orient"] - np.array(refer_pose[1])))
        # print(error)
        self.current_state_text.setText(f"平移: [{trans_str}]\n旋转: [{orient_str}]\n关节角: [{joints_str}]\n末端执行器: [{eef_pos_str}] m  -- [{eef_eff_str}] N")
    
    @pyqtSlot(np.ndarray)
    @safe_func()
    def update_capture_view(self, frame):
        """更捕获视图"""
        self.captured_label.setPixmap(self.qpixmap_from_image(frame))
    
    @pyqtSlot(dict)
    @safe_func()
    def update_info(self, info):
        """更新信息"""
        trans_str = ", ".join([f"{v:.6f}" for v in info["trans"]])
        orient_str = ", ".join([f"{v:.6f}" for v in info["orient"]])
        object_height = info["o_height"]
        object_width = info["o_width"]
        object_angle = info["o_angle"]
        object_label = info["o_label"]
        object_bbox = info["o_bbox"]
        
        self.predicted_info_text.setText(f"平移: [{trans_str}]\n旋转: [{orient_str}]")
        self.object_info_text.setText(f"高度: {object_height:.4f}\n宽度: {object_width:.4f}\n角度: {object_angle:.4f}\n标签: {object_label}\n边界框: {object_bbox}")
    
    @pyqtSlot(int, int, int)
    @safe_func()
    def handle_segmentation_click(self, x, y, button):
        print(f"handle_segmentation_click: {x}, {y}, {button}")
        """处理分割视图上的点击"""
        if self.airbot_segment.mode != SegmentMode.POINT or self.captured_color is None:
            self.log("处于非手动选择模式或未捕获图像")
            return
        with self.captured_frame_lock:
            _captured_color = self.captured_color
            
        # 调整坐标到原始图像大小
        width_ratio = _captured_color.shape[1] / self.video_width
        height_ratio = _captured_color.shape[0] / self.video_height
        
        orig_x = int(x * width_ratio)
        orig_y = int(y * height_ratio)

        if button == 0:  # 左键-前景
            self.airbot_segment.add_point(orig_x, orig_y, True)
            self.log(f"({orig_x}, {orig_y}) : 左键点击，添加为前景")
        elif button == 1:  # 右键-背景
            self.airbot_segment.add_point(orig_x, orig_y, False)
            self.log(f"({orig_x}, {orig_y}) : 右键点击，添加为背景")
        elif button == 2:  # 中键-清除
            self.airbot_segment.clear_prompt()
            self.log("中键点击，清除所有点")
            self.update_capture_view(_captured_color)
            return
        
        # 执行分割
        mask = self.airbot_segment.inference(_captured_color)
        if mask is None:
            self.log("分割失败，请重试")
            return
        
        # cv2.imwrite("mask.png", mask.astype(np.uint8) * 255)
        print("mask shape:", mask.shape)
        
        # 应用分割结果到图像
        masked_color = _captured_color.copy()
        print("masked_color shape:", masked_color.shape)
        masked_color[mask] = [191, 214, 238]  # 预览掩码
        # cv2.imwrite("mask_color.png", masked_color)
        with self.captured_frame_lock:
            self.mask_map = mask
        
        # 更新预览
        self.update_capture_view(masked_color)
        
        # if self.dump:
        #     # 保存结果
        #     cv2.imwrite("mask.png", mask.astype(np.uint8) * 255)
        #     cv2.imwrite("masked_color.png", masked_color)
    
    @pyqtSlot()
    @safe_func()
    def capture_frame(self):
        """捕获当前图像"""
        with self.observe_frame_lock and self.captured_frame_lock:
            if self.color_frame is None:
                self.log("错误：无法捕获图像，相机未连接或未初始化")
                return
            
            self.captured_color = self.color_frame.copy()
            self.captured_depth = self.depth_frame.copy()
            self.captured_depth_map = self.depth_map.copy()
            self.capture_state = self.current_state.copy()
            self.captured_pose = [self.current_state["trans"], self.current_state["orient"]]
                  
        # 更新捕获视图
        with self.captured_frame_lock:
            self.update_capture_view(self.captured_color)
            self.airbot_segment.clear_prompt()
                    
        # 更新提示
        if not self.auto_mode:
            self.capture_instruction.setText("点击图片添加前景(左键)或背景(右键)点，中键清除所有点")
        else:
            self.capture_instruction.setText("自动模式, 持续捕获图像帧")

        self.log("图像已捕获")

    @safe_func()
    def move_to_observe(self):
        self.gravity_btn.setStyleSheet(self.button_style)
        """移动机械臂到观察姿态"""
        with AIRBOTPlay(port=self.robot_port) as robot:
            robot.switch_mode(RobotMode.PLANNING_POS)
            robot.set_speed_profile(self.robot_speed)
            robot.set_params({"od_motor_can0_1.over_effort_thres": 0.2 * self.od_arm_safe_threshold})
            robot.set_params({"od_motor_can0_2.over_effort_thres": 1.2 * self.od_arm_safe_threshold})
            robot.set_params({"od_motor_can0_3.over_effort_thres": 1.5 *self.od_arm_safe_threshold})
            robot.set_params({"dm_motor_can0_4.over_effort_thres": self.dm_arm_safe_threshold})
            robot.set_params({"dm_motor_can0_5.over_effort_thres": self.dm_arm_safe_threshold})
            robot.set_params({"dm_motor_can0_6.over_effort_thres": 0.5 * self.dm_arm_safe_threshold})
            robot.set_params({"dm_motor_can0_7.over_effort_thres": self.gripper_safe_threshold})
            robot.move_to_cart_pose(self.observe_pose)
            time.sleep(arm_delay)
            self.log("已移动到观察姿态")

    @safe_func()
    def predict_grasp_pose(self):
        """预测抓取位姿"""
        with self.captured_frame_lock:
            if self.captured_color is None or self.mask_map is None:
                self.log("请先捕获图像并进行分割")
                return False
            mask_map = self.mask_map.copy()
            color = self.captured_color.copy()
            depth = self.captured_depth.copy()
            pose = self.captured_pose.copy()

        # 计算抓取位姿
        cloud = self.realsense.create_point_cloud(depth)
        trans, orient, cloud_base = self.airbot_grasp.inference(
            color_image=color, 
            depth_image=depth, 
            end_pose=pose, 
            cloud_cam_raw=cloud,
            mask=mask_map
        )
        
        print(trans, orient)
        
        object_height = np.max(cloud_base[:,2])
        f = (self.realsense.intrinsic[0][0] + self.realsense.intrinsic[1][1]) / 2
        y_indices, x_indices = np.where(mask_map)
        mask_center_y = int(np.mean(y_indices))
        mask_center_x = int(np.mean(x_indices))
        distance = depth[mask_center_y][mask_center_x]
        object_width = self.airbot_grasp.length_minor * (distance / self.realsense.depth_factor) / f
        object_angle = self.airbot_grasp.angle
        
        masked_color = np.ones_like(color)
        masked_color[mask_map] = color[mask_map]

        bbox, label, conf = self.airbot_yolo.get_max_conf_bbox_and_label(masked_color)
        
        predict_info = {
            "trans": trans,
            "orient": orient,
            "o_height": object_height,
            "o_width": object_width,
            "o_angle": object_angle,
            "o_label": label,
            "o_bbox": bbox
        }
        
        # 更新预测位姿
        with self.predict_lock:
            self.predicted_info = predict_info.copy() 
        
        # 更新UI
        self.update_info(predict_info)
        
        return True
    
    @safe_func()
    def move_to_predicted(self):
        self.gravity_btn.setStyleSheet(self.button_style)
        """移动到预测位姿"""
        self.predict_grasp_pose()
        with self.predict_lock:
            trans = self.predicted_info["trans"]
            orient = self.predicted_info["orient"]
        with AIRBOTPlay(port=self.robot_port) as robot:
            robot.switch_mode(RobotMode.PLANNING_POS)
            robot.move_eef_pos(1)  # 打开夹爪
            time.sleep(gripper_delay)
            self.gripper_opened = True
            res = robot.move_to_cart_pose([
                [trans[0], trans[1], trans[2] + 0.2], 
                orient
            ])
            if res is None:
                self.log("无法移动到预测位姿: ", [
                    [trans[0], trans[1], trans[2] + 0.2], 
                    orient
                ])
                return
            time.sleep(arm_delay)
            robot.move_eef_pos(0 * self.predicted_info["o_width"])
            time.sleep(gripper_delay)
            self.log("已移动到预测位姿")
    
    @safe_func()
    def predict_and_grasp(self):
        """执行抓取并移动到放置位置"""
        self.predict_grasp_pose()
        with self.predict_lock:
            self.robot_grasp_thread.set_predicted_info(self.predicted_info)
        self.robot_grasp_thread.start()
        
    @safe_func()
    def trigger_gripper(self):
        """控制夹爪开合"""
        with self.predict_lock:
            if self.predicted_info is not None:
                o_width = self.predicted_info["o_width"]
        with AIRBOTPlay(port=self.robot_port) as robot:
            if self.gripper_opened:
                if o_width is not None:
                    robot.move_eef_pos(0)
                    time.sleep(gripper_delay)
                                        
                else:
                    robot.move_eef_pos(0)
                    time.sleep(gripper_delay)
                                        
                self.gripper_opened = False
                self.log("夹爪已关闭")
            else:
                robot.move_eef_pos(1)
                time.sleep(0.3)
                self.gripper_opened = True
                                
                self.log("夹爪已打开")
    
    @safe_func()
    def set_place_pose(self):
        """设置放置姿态"""
        with AIRBOTPlay(port=self.robot_port) as robot:
            pose = robot.get_end_pose()
            self.place_pose = pose
            self.pre_place_pose = [[pose[0][0], pose[0][1], pose[0][2] + 0.2], pose[1]]
            self.log("已更新放置姿态")
    
    @safe_func()
    def change_speed_profile(self, speed_text):
        """根据下拉菜单选择更改机器人速度配置"""
        self.robot_speed = SpeedProfile[speed_text]

        with AIRBOTPlay(port=self.robot_port) as robot:
            robot.set_speed_profile(self.robot_speed)
            
        self.log(f"机器人速度配置已更改为: {speed_text}")
    
    @safe_func()
    def trigger_gravity(self):
        """控制重力开关"""
        with AIRBOTPlay(port=self.robot_port) as robot:
            control_mode = robot.get_control_mode()
            if control_mode != RobotMode.GRAVITY_COMP:
                robot.switch_mode(RobotMode.GRAVITY_COMP)
                self.gravity_btn.setStyleSheet(self.button_style_red)
                self.log("切换到重力补偿模式")
            else:
                robot.switch_mode(RobotMode.PLANNING_POS)
                self.gravity_btn.setStyleSheet(self.button_style)
                self.log("切换到PLANNING模式")
    
    def change_auto_mode(self):
        self.gravity_btn.setStyleSheet(self.button_style)
        """修改抓取模式"""
        if not self.auto_mode:
            try:
                self.auto_btn.setEnabled(False)
                self.auto_btn.setText("Waiting...") 
                self.auto_btn.setStyleSheet(self.button_style_grey)

                self.auto_mode = True
                self.airbot_segment.mode = SegmentMode.BBOX
                self.airbot_segment.clear_prompt()
                self.auto_grasp_thread.start()
                
                self.auto_btn.setText("Manual Select Mode")
                self.auto_btn.setStyleSheet(self.button_style_red)
                self.auto_btn.setEnabled(True)
            except Exception as e:
                details = traceback.format_exc()
                self.log(f"更改抓取模式失败: {e}\n{details}")
        else:
            try:
                self.auto_btn.setEnabled(False)
                self.auto_btn.setText("Waiting...") 
                self.auto_btn.setStyleSheet(self.button_style_grey)
                
                if hasattr(self, 'robot_grasp_thread') and self.robot_grasp_thread.isRunning():
                    self.robot_grasp_thread.requestInterruption()
                    self.robot_grasp_thread.wait()
                
                if hasattr(self, 'auto_grasp_thread') and self.auto_grasp_thread.isRunning():
                    self.auto_grasp_thread.requestInterruption()
                    self.auto_grasp_thread.wait()
                    
                self.auto_mode = False
                self.airbot_segment.mode = SegmentMode.POINT
                self.airbot_segment.clear_prompt()
                
                self.auto_btn.setText("Auto Grasp Mode")
                self.auto_btn.setStyleSheet(self.button_style)
                self.auto_btn.setEnabled(True)
            except Exception as e:
                details = traceback.format_exc()
                self.log(f"更改抓取模式失败: {e}\n{details}")
                
        self.log(f"抓取模式已更改为: {'自动抓取' if self.auto_mode else '手动选择'}")
    
    @safe_func()
    def closeEvent(self, event):
        """关闭窗口时的处理"""
        # 停止所有线程
        self.running = False
        
        if hasattr(self, 'robot_grasp_thread') and self.robot_grasp_thread.isRunning():
            self.robot_grasp_thread.requestInterruption()
            self.robot_grasp_thread.wait()
        
        if hasattr(self, 'auto_grasp_thread') and self.auto_grasp_thread.isRunning():
            self.auto_grasp_thread.requestInterruption()
            self.auto_grasp_thread.wait()

        if hasattr(self, 'observe_thread') and self.observe_thread.isRunning():
            self.observe_thread.stop()
            
        self.log("正在关闭应用...")
        super().closeEvent(event)
    
def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 尝试加载自定义字体
    try:
        font_id = QFontDatabase.addApplicationFont("MapleMonoBold.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            custom_font = QFont(font_family, 12)
            app.setFont(custom_font)
    except:
        custom_font = QFont("Monospace", 12)
        app.setFont(custom_font)
    
    # 创建并显示主窗口
    window = AirbotControlInterface()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()