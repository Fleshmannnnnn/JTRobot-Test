#!/usr/bin/env python3
"""
箱子顶面检测系统 - 多方法融合版
支持方法：
1. 颜色分割法（HSV/LAB空间）- 推荐
2. Canny边缘检测法
3. 霍夫直线检测法
4. 自适应阈值法
5. 混合投票法
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ==================== 检测方法枚举 ====================

class DetectionMethod(Enum):
    COLOR_SEGMENTATION = "color_seg"  # 颜色分割
    CANNY_EDGE = "canny"              # Canny边缘
    HOUGH_LINE = "hough"              # 霍夫直线
    ADAPTIVE_THRESHOLD = "adaptive"   # 自适应阈值
    HYBRID = "hybrid"                 # 混合方法

# ==================== 数据结构 ====================

@dataclass
class Point:
    x: float
    y: float
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y}
    
    def to_tuple(self):
        return (int(self.x), int(self.y))

@dataclass
class Path:
    id: str
    name: str
    path_type: str
    start: Point
    end: Point
    confidence: float
    description: str
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.path_type,
            'start_px': self.start.to_dict(),
            'end_px': self.end.to_dict(),
            'confidence': float(self.confidence),
            'description': self.description
        }

@dataclass
class Edge:
    id: str
    name: str
    start: Point
    end: Point
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'start_px': self.start.to_dict(),
            'end_px': self.end.to_dict()
        }

# ==================== 多方法检测器 ====================

class MultiMethodBoxDetector:
    """多方法融合的箱子检测器"""
    
    def __init__(self):
        self.current_method = DetectionMethod.COLOR_SEGMENTATION
        
        # 颜色分割参数（针对棕色箱子）
        self.color_lower_hsv = np.array([10, 30, 50])   # 棕色下界
        self.color_upper_hsv = np.array([30, 200, 200]) # 棕色上界
        
        # Canny参数
        self.canny_low = 30
        self.canny_high = 100
        
        # 霍夫直线参数
        self.hough_threshold = 80
        self.hough_min_line_length = 100
        self.hough_max_line_gap = 20
        
        # 通用参数
        self.min_area = 5000
        self.max_area = 500000
        self.min_rect_score = 0.50
        
    # ==================== 方法1：颜色分割 ====================
    
    def detect_by_color_segmentation(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """使用颜色分割检测箱子"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建掩码
        mask = cv2.inRange(hsv, self.color_lower_hsv, self.color_upper_hsv)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除噪声
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        # 找最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if not (self.min_area < area < self.max_area):
            return None, mask
        
        # 拟合最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        corners = cv2.boxPoints(rect)
        
        return self.sort_corners(corners), mask
    
    # ==================== 方法2：Canny边缘检测 ====================
    
    def detect_by_canny(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """使用Canny边缘检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # 形态学闭运算
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        corners = self._find_best_rectangle(contours)
        return corners, edges
    
    # ==================== 方法3：霍夫直线检测 ====================
    
    def detect_by_hough_lines(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """使用霍夫直线检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                               minLineLength=self.hough_min_line_length,
                               maxLineGap=self.hough_max_line_gap)
        
        if lines is None or len(lines) < 4:
            return None, edges
        
        # 分类直线为水平和垂直
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            if angle < 20 or angle > 160:  # 水平线
                horizontal_lines.append(line[0])
            elif 70 < angle < 110:  # 垂直线
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None, edges
        
        # 找到最外围的直线
        horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]
        
        # 计算交点
        corners = np.array([
            self._line_intersection(top_line, left_line),
            self._line_intersection(top_line, right_line),
            self._line_intersection(bottom_line, right_line),
            self._line_intersection(bottom_line, left_line)
        ])
        
        return corners, edges
    
    # ==================== 方法4：自适应阈值 ====================
    
    def detect_by_adaptive_threshold(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """使用自适应阈值"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        corners = self._find_best_rectangle(contours)
        return corners, thresh
    
    # ==================== 方法5：混合投票 ====================
    
    def detect_by_hybrid(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """混合多种方法，投票选择最佳结果"""
        results = {}
        
        # 尝试所有方法
        corners_color, mask_color = self.detect_by_color_segmentation(image)
        corners_canny, mask_canny = self.detect_by_canny(image)
        corners_hough, mask_hough = self.detect_by_hough_lines(image)
        corners_adaptive, mask_adaptive = self.detect_by_adaptive_threshold(image)
        
        results['color'] = corners_color
        results['canny'] = corners_canny
        results['hough'] = corners_hough
        results['adaptive'] = corners_adaptive
        
        # 过滤有效结果
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, {
                'color': mask_color,
                'canny': mask_canny,
                'hough': mask_hough,
                'adaptive': mask_adaptive
            }
        
        # 如果只有一个有效结果，直接返回
        if len(valid_results) == 1:
            return list(valid_results.values())[0], {
                'color': mask_color,
                'canny': mask_canny,
                'hough': mask_hough,
                'adaptive': mask_adaptive,
                'selected': list(valid_results.keys())[0]
            }
        
        # 多个结果：计算面积，选择最大的
        best_corners = None
        best_area = 0
        best_method = None
        
        for method, corners in valid_results.items():
            area = cv2.contourArea(corners.astype(np.float32))
            if area > best_area:
                best_area = area
                best_corners = corners
                best_method = method
        
        return best_corners, {
            'color': mask_color,
            'canny': mask_canny,
            'hough': mask_hough,
            'adaptive': mask_adaptive,
            'selected': best_method,
            'candidates': len(valid_results)
        }
    
    # ==================== 主检测入口 ====================
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """根据当前方法检测箱子"""
        if self.current_method == DetectionMethod.COLOR_SEGMENTATION:
            corners, debug = self.detect_by_color_segmentation(image)
            return corners, {'method': 'color', 'debug': debug}
        
        elif self.current_method == DetectionMethod.CANNY_EDGE:
            corners, debug = self.detect_by_canny(image)
            return corners, {'method': 'canny', 'debug': debug}
        
        elif self.current_method == DetectionMethod.HOUGH_LINE:
            corners, debug = self.detect_by_hough_lines(image)
            return corners, {'method': 'hough', 'debug': debug}
        
        elif self.current_method == DetectionMethod.ADAPTIVE_THRESHOLD:
            corners, debug = self.detect_by_adaptive_threshold(image)
            return corners, {'method': 'adaptive', 'debug': debug}
        
        elif self.current_method == DetectionMethod.HYBRID:
            corners, debug = self.detect_by_hybrid(image)
            return corners, {'method': 'hybrid', 'debug': debug}
        
        return None, {}
    
    # ==================== 辅助方法 ====================
    
    def _find_best_rectangle(self, contours: List) -> Optional[np.ndarray]:
        """从轮廓中找最佳矩形"""
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (self.min_area < area < self.max_area):
                continue
            
            # 计算矩形度
            rect = cv2.minAreaRect(contour)
            box_area = rect[1][0] * rect[1][1]
            
            if box_area < 1:
                continue
            
            rect_score = area / box_area
            
            if rect_score < self.min_rect_score:
                continue
            
            if rect_score > best_score:
                best_score = rect_score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        rect = cv2.minAreaRect(best_contour)
        corners = cv2.boxPoints(rect)
        
        return self.sort_corners(corners)
    
    def _line_intersection(self, line1, line2):
        """计算两条直线的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return [(x1+x2)/2, (y1+y2)/2]
        
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        return [px, py]
    
    def sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """排序角点：左上、右上、右下、左下"""
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # 找左上角
        top_left_idx = np.argmin(sorted_corners[:, 0] + sorted_corners[:, 1])
        sorted_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        return sorted_corners

# ==================== 路径计算器 ====================

class PathCalculator:
    """切割路径计算器"""
    
    def __init__(self):
        self.top_margin = 0.12
        self.bottom_margin = 0.12
        self.side_margin = 3
        self.center_offset = 0
    
    def calculate_paths_and_edges(self, corners: np.ndarray) -> Tuple[List[Path], List[Edge]]:
        """计算切割路径和边缘线"""
        if len(corners) != 4:
            return [], []
        
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]
        
        width = top_right[0] - top_left[0]
        height = bottom_left[1] - top_left[1]
        
        center_x = (top_left[0] + top_right[0]) / 2 + self.center_offset
        top_y = top_left[1] + height * self.top_margin
        bottom_y = top_left[1] + height * (1 - self.bottom_margin)
        
        vertical_start_y = top_y
        vertical_end_y = bottom_y
        
        paths = []
        
        paths.append(Path(
            id='left_horizontal',
            name='Left horizontal tape',
            path_type='horizontal',
            start=Point(top_left[0] + self.side_margin, top_y),
            end=Point(center_x - self.side_margin, top_y),
            confidence=0.90,
            description='Left cover horizontal tape'
        ))
        
        paths.append(Path(
            id='center_vertical',
            name='Center vertical gap',
            path_type='vertical',
            start=Point(center_x, vertical_start_y),
            end=Point(center_x, vertical_end_y),
            confidence=0.95,
            description='Center gap between two covers'
        ))
        
        paths.append(Path(
            id='right_horizontal',
            name='Right horizontal tape',
            path_type='horizontal',
            start=Point(center_x + self.side_margin, top_y),
            end=Point(top_right[0] - self.side_margin, top_y),
            confidence=0.90,
            description='Right cover horizontal tape'
        ))
        
        edges = []
        
        edges.append(Edge(
            id='left_edge',
            name='Left edge',
            start=Point(top_left[0], top_left[1]),
            end=Point(bottom_left[0], bottom_left[1])
        ))
        
        edges.append(Edge(
            id='right_edge',
            name='Right edge',
            start=Point(top_right[0], top_right[1]),
            end=Point(bottom_right[0], bottom_right[1])
        ))
        
        return paths, edges

# ==================== 可视化器 ====================

class Visualizer:
    """检测结果可视化"""
    
    @staticmethod
    def draw_detection_result(image: np.ndarray, corners: np.ndarray, 
                             paths: List[Path], edges: List[Edge]) -> np.ndarray:
        """绘制检测结果"""
        vis = image.copy()
        
        if corners is None or len(corners) != 4:
            return vis
        
        corners_int = corners.astype(np.int32)
        cv2.polylines(vis, [corners_int], True, (0, 165, 255), 3)
        
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        for i, (corner, label) in enumerate(zip(corners, corner_labels)):
            pt = (int(corner[0]), int(corner[1]))
            cv2.circle(vis, pt, 8, (0, 255, 0), -1)
            cv2.circle(vis, pt, 10, (255, 255, 255), 2)
            cv2.putText(vis, label, (pt[0]-10, pt[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for edge in edges:
            start = edge.start.to_tuple()
            end = edge.end.to_tuple()
            Visualizer.draw_dashed_line(vis, start, end, (255, 0, 0), 2, 10)
        
        for path in paths:
            start = path.start.to_tuple()
            end = path.end.to_tuple()
            
            if path.path_type == 'horizontal':
                color = (0, 255, 255)
                cv2.arrowedLine(vis, start, end, color, 3, tipLength=0.03)
            else:
                color = (0, 0, 255)
                cv2.arrowedLine(vis, start, end, color, 3, tipLength=0.02)
        
        return vis
    
    @staticmethod
    def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
        """绘制虚线"""
        dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
        dashes = int(dist / dash_length)
        
        for i in range(0, dashes, 2):
            start = (
                int(pt1[0] + (pt2[0]-pt1[0])*i/dashes),
                int(pt1[1] + (pt2[1]-pt1[1])*i/dashes)
            )
            end = (
                int(pt1[0] + (pt2[0]-pt1[0])*(i+1)/dashes),
                int(pt1[1] + (pt2[1]-pt1[1])*(i+1)/dashes)
            )
            cv2.line(img, start, end, color, thickness)
    
    @staticmethod
    def draw_info_overlay(image: np.ndarray, info_text: List[str]) -> np.ndarray:
        """绘制信息覆盖层"""
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (500, 30 + len(info_text)*25), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (20, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return image

# ==================== 主检测系统 ====================

class BoxDetectionSystem:
    """箱子检测系统"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        self.detector = MultiMethodBoxDetector()
        self.path_calculator = PathCalculator()
        self.visualizer = Visualizer()
        
        self.detection_buffer = []
        self.buffer_size = 5
        
        self.frozen = False
        self.frozen_result = None
        self.show_debug = False
        
        self.save_dir = "detection_results"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def init_camera(self) -> bool:
        """初始化相机"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Camera initialized")
        return True
    
    def smooth_detection(self, corners: np.ndarray) -> np.ndarray:
        """多帧平均"""
        if corners is None:
            return None
        
        self.detection_buffer.append(corners)
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
        
        avg_corners = np.mean(self.detection_buffer, axis=0)
        return avg_corners
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        corners, debug_info = self.detector.detect(frame)
        
        if corners is None:
            return {
                'success': False, 
                'message': 'No box detected',
                'debug_info': debug_info
            }
        
        smoothed_corners = self.smooth_detection(corners)
        paths, edges = self.path_calculator.calculate_paths_and_edges(smoothed_corners)
        
        width = np.linalg.norm(smoothed_corners[1] - smoothed_corners[0])
        height = np.linalg.norm(smoothed_corners[3] - smoothed_corners[0])
        aspect_ratio = width / height if height > 0 else 0
        
        return {
            'success': True,
            'corners': smoothed_corners,
            'paths': paths,
            'edges': edges,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'debug_info': debug_info
        }
    
    def save_result(self, frame: np.ndarray, result: Dict):
        """保存检测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        img_path = os.path.join(self.save_dir, f"detection_{timestamp}.jpg")
        vis_frame = self.visualizer.draw_detection_result(
            frame, result['corners'], result['paths'], result['edges']
        )
        cv2.imwrite(img_path, vis_frame)
        
        # 转换所有 NumPy 类型为 Python 原生类型
        def convert_to_native(obj):
            """递归转换 NumPy 类型为 Python 原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        json_path = os.path.join(self.save_dir, f"detection_{timestamp}.json")
        json_data = {
            'timestamp': timestamp,
            'method': result.get('debug_info', {}).get('method', 'unknown'),
            'corners': [
                {'id': i+1, 'x': float(c[0]), 'y': float(c[1]), 
                 'label': ['TL', 'TR', 'BR', 'BL'][i]}
                for i, c in enumerate(result['corners'])
            ],
            'paths': [convert_to_native(p.to_dict()) for p in result['paths']],
            'edges': [convert_to_native(e.to_dict()) for e in result['edges']],
            'box_size': {
                'width_px': float(result['width']),
                'height_px': float(result['height']),
                'aspect_ratio': float(result['aspect_ratio'])
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved:")
        print(f"   Image: {img_path}")
        print(f"   Data: {json_path}")
    
    def run(self):
        """运行主循环"""
        if not self.init_camera():
            return
        
        print("\n" + "="*60)
        print("Box Detection System - Multi-Method")
        print("="*60)
        print("Detection Methods:")
        print("  [1] - Color Segmentation (RECOMMENDED)")
        print("  [2] - Canny Edge Detection")
        print("  [3] - Hough Line Detection")
        print("  [4] - Adaptive Threshold")
        print("  [5] - Hybrid (All methods)")
        print("\nControls:")
        print("  [SPACE] - Freeze and detect")
        print("  [s]     - Save result")
        print("  [d]     - Toggle debug mode")
        print("  [r]     - Reset detection")
        print("  [i/k]   - Adjust horizontal path (up/down)")
        print("  [j/l]   - Adjust vertical path (left/right)")
        print("  [q]     - Quit")
        print("="*60 + "\n")
        print("Starting with Color Segmentation method...")
        print("Press [d] to see debug windows\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Cannot read frame")
                break
            
            if self.frozen and self.frozen_result:
                result = self.frozen_result
                display_frame = self.visualizer.draw_detection_result(
                    frame, result['corners'], result['paths'], result['edges']
                )
                
                method_name = result.get('debug_info', {}).get('method', 'unknown')
                info = [
                    f"Status: FROZEN | Method: {method_name}",
                    f"Box size: {int(result['width'])} x {int(result['height'])} px",
                    f"Aspect ratio: {result['aspect_ratio']:.2f}",
                    f"Paths: {len(result['paths'])}"
                ]
                display_frame = self.visualizer.draw_info_overlay(display_frame, info)
                
            else:
                result = self.process_frame(frame)
                
                if result['success']:
                    display_frame = self.visualizer.draw_detection_result(
                        frame, result['corners'], result['paths'], result['edges']
                    )
                    
                    method_name = result.get('debug_info', {}).get('method', 'unknown')
                    info = [
                        f"Status: DETECTING | Method: {method_name}",
                        f"Box size: {int(result['width'])} x {int(result['height'])} px",
                        f"Aspect ratio: {result['aspect_ratio']:.2f}"
                    ]
                    display_frame = self.visualizer.draw_info_overlay(display_frame, info)
                else:
                    display_frame = frame.copy()
                    method = self.detector.current_method.value
                    info = [
                        f"Status: SEARCHING... | Method: {method}",
                        f"Try different methods: [1-5]"
                    ]
                    display_frame = self.visualizer.draw_info_overlay(display_frame, info)
                
                # 调试窗口
                if self.show_debug and 'debug_info' in result:
                    debug_info = result['debug_info']
                    
                    if 'debug' in debug_info:
                        debug_img = debug_info['debug']
                        
                        if len(debug_img.shape) == 2:
                            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                        
                        method_name = debug_info.get('method', 'unknown')
                        cv2.putText(debug_img, f"Method: {method_name}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow('Debug View', debug_img)
            
            cv2.imshow('Box Detection', display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):
                if not self.frozen:
                    result = self.process_frame(frame)
                    if result['success']:
                        self.frozen = True
                        self.frozen_result = result
                        print("\n*** Frame FROZEN ***")
                    else:
                        print("\n*** Cannot freeze - No box detected ***")
                else:
                    self.frozen = False
                    self.frozen_result = None
                    print("\n*** Resumed ***")
            
            elif key == ord('s'):
                if self.frozen and self.frozen_result:
                    self.save_result(frame, self.frozen_result)
                else:
                    result = self.process_frame(frame)
                    if result['success']:
                        self.save_result(frame, result)
            
            elif key == ord('r'):
                self.frozen = False
                self.frozen_result = None
                self.detection_buffer.clear()
                print("\n*** Reset ***")
            
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                if not self.show_debug:
                    cv2.destroyWindow('Debug View')
                print(f"\n*** Debug: {'ON' if self.show_debug else 'OFF'} ***")
            
            elif key == ord('1'):
                self.detector.current_method = DetectionMethod.COLOR_SEGMENTATION
                print("\n*** Method: Color Segmentation ***")
            
            elif key == ord('2'):
                self.detector.current_method = DetectionMethod.CANNY_EDGE
                print("\n*** Method: Canny Edge ***")
            
            elif key == ord('3'):
                self.detector.current_method = DetectionMethod.HOUGH_LINE
                print("\n*** Method: Hough Lines ***")
            
            elif key == ord('4'):
                self.detector.current_method = DetectionMethod.ADAPTIVE_THRESHOLD
                print("\n*** Method: Adaptive Threshold ***")
            
            elif key == ord('5'):
                self.detector.current_method = DetectionMethod.HYBRID
                print("\n*** Method: Hybrid (trying all) ***")
            
            elif key == ord('i'):
                self.path_calculator.top_margin = max(0.05, self.path_calculator.top_margin - 0.01)
                print(f"\n[i] Top margin: {self.path_calculator.top_margin:.2f}")
            
            elif key == ord('k'):
                self.path_calculator.top_margin = min(0.30, self.path_calculator.top_margin + 0.01)
                print(f"\n[k] Top margin: {self.path_calculator.top_margin:.2f}")
            
            elif key == ord('j'):
                self.path_calculator.center_offset -= 2
                print(f"\n[j] Center offset: {self.path_calculator.center_offset}")
            
            elif key == ord('l'):
                self.path_calculator.center_offset += 2
                print(f"\n[l] Center offset: {self.path_calculator.center_offset}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n*** Program exited ***")

# ==================== 主程序入口 ====================

def main():
    system = BoxDetectionSystem(camera_id=2)
    system.run()

if __name__ == "__main__":
    main()