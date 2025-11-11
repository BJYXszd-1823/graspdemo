import numpy as np
import torch
import yaml
import os
import time
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from utils import plot_gripper, masks_to_boxes


class SimpleGrasp:
    def __init__(self):
        self.safe_height_bias = None
        self.base_height = None
        self.gripper_width = None
        self.gripper_length = None
        self.cam2end = None
        self.grasp_depth = None
        self.angle = None     
        self.length_minor = None

        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        config = yaml.safe_load(open(config_path, "r"))
        
        self.robot_port = config["ArmParams"]["port"]
        self.base_height = config["ArmParams"]["base_height"]
        self.gripper_width = config["ArmParams"]["gripper_width"]
        self.gripper_length = config["ArmParams"]["gripper_length"]
        
        camera_type = config["AirbotGrasp"]["camera_type"]
        resolution = config[camera_type]["resolution"]
        self.cam2end = config[camera_type][resolution]["extrinsic"]
        self.grasp_depth = config["AirbotGrasp"]["grasp_depth"]
        self.safe_height_bias = config["AirbotGrasp"]["safe_height_bias"]
        self.observe_pose = config["AirbotGrasp"]["observe_pose"]
        self.place_pose = config["AirbotGrasp"]["place_pose"]
        self.pre_place_pose = config["AirbotGrasp"]["pre_place_pose"]
        
        self.dump = config["RunTime"]["dump"]
        self.dump_path = config["RunTime"]["dump_path"]
        os.makedirs(self.dump_path, exist_ok=True)

    def cam_cloud_to_base(self, cloud_cam, end_pose):
        cloud_cam_homogeneous = np.hstack((cloud_cam, np.ones((cloud_cam.shape[0], 1))))
        gripper2base = np.eye(4)
        gripper2base[:3, :3] = Rotation.from_quat(end_pose[1]).as_matrix()
        gripper2base[:3, 3] = end_pose[0]
        cloud_base_homogeneous = (
            gripper2base @ self.cam2end @ cloud_cam_homogeneous.T
        ).T
        cloud_base = cloud_base_homogeneous[:, :3]
        return cloud_base
    
    def find_narrowest(self, mask: np.ndarray, limit = None) -> float:
        coords = np.argwhere(mask)
        if coords.shape[0] < 2:
            return None, None

        # PCA
        pca = PCA(n_components=2)
        pca.fit(coords)

        # 主轴和副轴上的投影长度
        projected = pca.transform(coords)
        self.length_minor = projected[:, 1].max() - projected[:, 1].min()

        # 副轴方向就是最窄方向
        narrowest_vector = pca.components_[1]
        angle_rad = np.arctan2(narrowest_vector[0], narrowest_vector[1])  # 注意顺序
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 180:
            angle_deg = angle_deg - 180
        elif angle_deg < -180:
            angle_deg = angle_deg + 180
            
        if limit and angle_deg > limit:
                angle_deg = angle_deg - 180

        self.angle = angle_deg
        
        if self.dump:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            import matplotlib.patches as patches
            
            # Calculate center of mass of the mask
            center_y, center_x = np.mean(coords, axis=0).astype(int)
            
            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Display the mask
            ax.imshow(mask, cmap='gray', alpha=0.7)
            
            # Plot points
            ax.scatter(coords[:, 1], coords[:, 0], color='skyblue', s=5, alpha=0.5, label='Mask Points')
            
            # Scale factor for vectors
            scale = min(mask.shape[0], mask.shape[1]) // 4
            
            # Plot narrowest vector
            dx, dy = narrowest_vector
            ax.arrow(center_x, center_y, 
                    scale * dy, scale * dx,  # Note: x corresponds to columns, y to rows
                    head_width=5, head_length=10, fc='green', ec='green', 
                    length_includes_head=True, label='Narrowest Vector')
            
            # Plot main axis vector
            main_vector = pca.components_[0]
            dx_main, dy_main = main_vector
            ax.arrow(center_x, center_y, 
                    scale * dy_main, scale * dx_main, 
                    head_width=5, head_length=10, fc='red', ec='red', 
                    length_includes_head=True, label='Major Axis')
            
            # Draw angle arc
            radius = min(mask.shape[0], mask.shape[1]) // 10
            angle_arc = patches.Arc((center_x, center_y), radius*2, radius*2,
                                theta1=0, theta2=np.degrees(angle_rad) % 360,
                                color='orange', lw=2)
            ax.add_patch(angle_arc)
            
            # Add reference line (horizontal reference)
            ax.plot([center_x, center_x + radius], [center_y, center_y], 
                    'y--', alpha=0.7, label='Reference (0°)')
            
            # Add text for angle
            text_x = center_x + radius * 0.5
            text_y = center_y - radius * 0.3
            ax.text(text_x, text_y, f"Angle: {angle_deg:.1f}°", 
                    color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            # Set plot properties
            ax.set_title('PCA Analysis of Mask Shape')
            ax.legend(loc='lower right')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Display coordinates at the center
            ax.text(center_x + 10, center_y + 10, f"Center: ({center_x}, {center_y})", 
                    fontsize=9, color='purple')
            
            # Customize the plot appearance
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save the figure if a path is provided
            file_name = time.strftime("%Y%m%d%H%M%S") + "_pca_analysis.jpg"
            plt.savefig(os.path.join(self.dump_path, file_name), dpi=150, bbox_inches='tight')
            plt.close()
                
        return angle_deg

    def inference(self, color_image, depth_image, cloud_cam_raw, end_pose, mask=None, bbox=None, preview_cloud = False, save_cloud = True):
        color = color_image.astype(np.float32) / 255.0
        depth = depth_image.astype(np.float32)
        # filter valid cloud
        time1 = time.time()
        cloud_cam = cloud_cam_raw[mask & (depth >= 0)]
        # print("cloud_cam.shape: ", cloud_cam.shape)
        cloud_color = color[mask & (depth >= 0)]
        print("time mask: ", time.time() - time1)
        time1 = time.time()
        # transform to base
        cloud_base = self.cam_cloud_to_base(cloud_cam, end_pose)
        # print(f"cloud_base.shape: {cloud_base.shape}, len: {len(cloud_base)}")
        cloud_base = cloud_base[cloud_base[:, 2] > 0]
        print("time cloud_base: ", time.time() - time1)
        if len(cloud_base) == 0:
            return None, None, None
        # build trans
        time1 = time.time()
        x = np.mean(cloud_base[:, 0])
        y = np.mean(cloud_base[:, 1])
        z_top = np.amax(cloud_base[:, 2])
        print("time mean: ", time.time() - time1)
            
        gripper_bottom = z_top - self.gripper_length
        z = gripper_bottom if  gripper_bottom > self.safe_height_bias else self.safe_height_bias
        trans = np.array([x, y, z])
        # build orient
        limit_angle = np.degrees(np.arctan2(x,y))
        angle = self.find_narrowest(mask, limit_angle)
        # angle = 90
        time1 = time.time()
        r1 = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
        r2 = Rotation.from_euler("xyz", [angle, 0, 0], degrees=True) # direction of x-axis is downward
        orient = (r1 * r2).as_quat()
        print("time euler: ", time.time() - time1)
        if preview_cloud or save_cloud:
            # visualize points cloud
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_base.astype(np.float32))
            cloud.colors = o3d.utility.Vector3dVector(cloud_color.astype(np.float32))
            # visualize gripper
            g = plot_gripper(
                trans,
                Rotation.from_quat(orient).as_matrix(),
                self.gripper_width,
                self.gripper_length,
            )
            if preview_cloud:
                o3d.visualization.draw_geometries([cloud, g])
            if self.dump:
                combined_cloud = o3d.geometry.PointCloud()
                combined_cloud.points = cloud.points
                combined_cloud.colors = cloud.colors

                # 转换gripper mesh为点云
                gripper_points = np.asarray(g.sample_points_uniformly(number_of_points=10000).points)
                gripper_colors = np.tile(np.array([0.5, 0.5, 0.5]), (gripper_points.shape[0], 1)) # 设置夹爪颜色为灰色

                combined_cloud.points.extend(o3d.utility.Vector3dVector(gripper_points))
                combined_cloud.colors.extend(o3d.utility.Vector3dVector(gripper_colors))

                file_name_prefix = os.path.join(self.dump_path, time.strftime("%Y%m%d%H%M%S", time.localtime()))

                o3d.io.write_point_cloud(f"{file_name_prefix}_cloud.ply", combined_cloud)
                cv2.imwrite(f"{file_name_prefix}_mask.png", mask.astype(np.uint8) * 255)
                cv2.imwrite(f"{file_name_prefix}_color.png", color_image)
                cv2.imwrite(f"{file_name_prefix}_depth.png", depth_image.astype(np.uint16))
                masked_color = color_image.copy()
                masked_color[mask] = [191, 214, 238]
                cv2.imwrite(f"{file_name_prefix}_masked_color.png", masked_color)

        return trans, orient, cloud_base


if __name__ == "__main__":
    import cv2
    import time
    import yaml
    import os
    from airbot_py.arm import AIRBOTPlay, RobotMode
    from airbot_camera import RealsenseCamera
    from airbot_segment import AirbotSegment

    realsense = RealsenseCamera()
    airbot_segment = AirbotSegment()
    airbot_grasp = SimpleGrasp()
    color = None
    depth = None
    mask = None
    bbox = None
    masked_color = None
    
    

    # observe pose
    with AIRBOTPlay(port=airbot_grasp.robot_port) as robot:
        robot.switch_mode(RobotMode.PLANNING_POS)
        robot.move_to_cart_pose(airbot_grasp.observe_pose)
    while True:
        # get image
        while True:
            color_frame, depth_frame = realsense.get_frame(align=True)
            depth_map = realsense.get_frame(frame_type="depth_map", align=True)
            if color_frame is not None and depth_frame is not None:
                cv2.imshow("color", color_frame)
            key = cv2.waitKey(1)
            if key == 27:
                color = color_frame
                depth = depth_frame
                cv2.destroyAllWindows()
                airbot_segment.set_image(color)
                airbot_segment.input_labels.clear()
                airbot_segment.input_points.clear()
                break
        # get bbox
        masked_color = color.copy()
        
        cv2.namedWindow("Segment")
        def mouse_callback(event, x, y, flags, param):
            global color, mask, bbox
            masked_color = color.copy()
            if event == cv2.EVENT_LBUTTONDOWN:
                airbot_segment.add_point(x, y, True)
                print("({}, {}) : Left clicked, add as positive".format(x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                airbot_segment.add_point(x, y, False)
                print("({}, {}) : Right clicked, add as negative".format(x, y))
            elif event == cv2.EVENT_MBUTTONDOWN:
                airbot_segment.clear_prompt()
                print("Middle mouse button clicked, clearing points")
                cv2.imshow("Segment", masked_color)
                return
            else:
                return
            masks, _, _ = airbot_segment.inference()          
            mask = masks[0]
            airbot_grasp.find_narrowest(mask)
            print("angle: ",airbot_grasp.angle)
            masked_color[mask] = [191, 214, 238]  # preview mask
            cv2.imshow("Segment", masked_color)
            bboxes = masks_to_boxes(torch.from_numpy(masks))
            bbox = bboxes[0]

        cv2.setMouseCallback("Segment", mouse_callback)
        
        while True:
            cv2.imshow("Segment", masked_color)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break

        # # get grasp
        # end_pose = None
        # with AIRBOTPlay(port=airbot_grasp.robot_port) as robot:
        #     robot.switch_mode(RobotMode.PLANNING_POS)
        #     end_pose = robot.get_end_pose()
        # cloud_cam_raw = realsense.create_point_cloud(depth)
        # trans, orient, cloud_base = airbot_grasp.inference(
        #     color_image=color, depth_image=depth, cloud_cam_raw=cloud_cam_raw, end_pose=end_pose, mask=mask
        # )
        # o_height = np.max(cloud_base[:, 2])
        
        # f = (realsense.intrinsic[0][0] + realsense.intrinsic[1][1]) / 2
        # y_indices, x_indices = np.where(mask)
        # mask_center_y = int(np.mean(y_indices))
        # mask_center_x = int(np.mean(x_indices))
        # distance = depth[mask_center_y][mask_center_x]
        # o_width = airbot_grasp.length_minor * (distance / realsense.depth_factor) / f

        # # grasp
        # with AIRBOTPlay(port=airbot_grasp.robot_port) as robot:
        #     robot.switch_mode(RobotMode.PLANNING_POS)
        #     robot.move_eff_pos(1)
        #     robot.move_to_cart_pose(
        #         [[trans[0], trans[1], trans[2] + o_height * 2], orient]
        #     )
        #     robot.move_to_cart_pose([trans, orient])
        #     robot.move_eff_pos(0.6 * o_width)
        #     robot.move_to_cart_pose(
        #         [[trans[0], trans[1], trans[2] + 0.2], orient]
        #     )
        #     robot.move_to_cart_pose(airbot_grasp.pre_place_pose)
        #     robot.move_to_cart_pose(airbot_grasp.place_pose)
        #     robot.move_eff_pos(1)
        #     robot.move_to_cart_pose(airbot_grasp.observe_pose)
