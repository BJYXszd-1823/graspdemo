"""Shared color-target preparation used by the hardware and simulation demos."""
import numpy as np
from airbot_yolo import draw_detections


def prepare_color_grasp(snapshot, detection, detections, segment, camera, grasp,
                        settings, *, save_cloud=True):
    mask = segment.inference_bbox(snapshot.color, list(detection.bbox))
    if mask is None or not np.any(mask):
        raise ValueError("MobileSAM 未生成有效目标掩码")
    mask_area = int(np.count_nonzero(mask))
    if mask_area < 100:
        raise ValueError("目标掩码面积过小")
    x1, y1, x2, y2 = detection.bbox
    in_box = np.zeros_like(mask, dtype=bool)
    in_box[y1:y2, x1:x2] = True
    mask_in_box_ratio = np.count_nonzero(mask & in_box) / mask_area
    required_ratio = float(settings.get(
        "min_mask_in_box_ratio", 0.50))
    if mask_in_box_ratio < required_ratio:
        raise ValueError("分割掩码与选中检测框不一致")
    if camera.has_hardware_depth:
        valid_depth = snapshot.depth[mask] > 0
        if np.count_nonzero(valid_depth) < max(20, int(mask_area * 0.25)):
            raise ValueError("目标区域没有足够的有效深度")
    cloud = camera.create_point_cloud(
        snapshot.depth, end_pose=[snapshot.state["trans"], snapshot.state["orient"]])
    pose = [snapshot.state["trans"], snapshot.state["orient"]]
    trans, orient, cloud_base = grasp.inference(
        color_image=snapshot.color, depth_image=snapshot.depth,
        end_pose=pose, cloud_cam_raw=cloud, mask=mask, save_cloud=save_cloud)
    if trans is None or orient is None or cloud_base is None or len(cloud_base) == 0:
        raise ValueError("抓取位姿计算失败")
    y_indices, x_indices = np.where(mask)
    center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
    if camera.has_hardware_depth:
        distance = snapshot.depth[center_y][center_x]
        focal = (camera.intrinsic[0][0]
                 + camera.intrinsic[1][1]) / 2
        width = (grasp.length_minor
                 * (distance / camera.depth_factor) / focal)
    else:
        width = float(np.ptp(cloud_base[:, 0]))
    info = {
        "trans": np.asarray(trans).tolist(),
        "orient": np.asarray(orient).tolist(),
        "o_height": float(np.max(cloud_base[:, 2])),
        "o_width": float(width),
        "o_angle": float(grasp.angle),
        "o_label": detection.display_label,
        "o_bbox": list(detection.bbox),
        "snapshot_timestamp": snapshot.timestamp,
    }
    preview = draw_detections(snapshot.color, detections, detection.bbox)
    preview[mask] = (0.65 * preview[mask] + 0.35 * np.array([191, 214, 238])).astype(np.uint8)
    return info, preview
