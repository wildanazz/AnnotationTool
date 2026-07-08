# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = (640, 640)  # (height, width)
STRIDES = [8, 16, 32]
NMS_IOU_THRESHOLD = 0.45


class ModelHandler:
    def __init__(self, labels):
        self.labels = labels
        self.model = None
        self.input_name = None
        self.load_network(model="yolox_x_comb.onnx")

    def load_network(self, model):
        device = ort.get_device()
        cuda = device == "GPU"
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if cuda
                else ["CPUExecutionProvider"]
            )
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.input_name = self.model.get_inputs()[0].name
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def _preprocess(self, img):
        # img: HxWx3, BGR, uint8. YOLOX uses letterbox padding (value 114)
        # and does NOT normalize to [0, 1] or apply mean/std.
        padded = np.ones((INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.uint8) * 114
        ratio = min(INPUT_SIZE[0] / img.shape[0], INPUT_SIZE[1] / img.shape[1])
        resized = cv2.resize(
            img,
            (int(round(img.shape[1] * ratio)), int(round(img.shape[0] * ratio))),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.ascontiguousarray(blob[None], dtype=np.float32)  # add batch dim
        return blob, ratio

    @staticmethod
    def _decode(predictions):
        # predictions: (N, 5 + num_classes) raw grid offsets. Convert the box
        # part to absolute cxcywh in the 640x640 input space.
        grids = []
        expanded_strides = []
        for stride in STRIDES:
            hsize = INPUT_SIZE[0] // stride
            wsize = INPUT_SIZE[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((grid.shape[0], 1), stride))
        grids = np.concatenate(grids, 0)
        expanded_strides = np.concatenate(expanded_strides, 0)

        predictions[:, :2] = (predictions[:, :2] + grids) * expanded_strides
        predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * expanded_strides
        return predictions

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]
        return keep

    def infer(self, image, threshold):
        rgb = np.array(image)  # HxWx3, RGB
        bgr = rgb[:, :, ::-1].copy()  # YOLOX was trained on BGR (cv2) input
        blob, ratio = self._preprocess(bgr)

        predictions = self.model.run(None, {self.input_name: blob})[0][0]
        predictions = self._decode(predictions)

        boxes = predictions[:, :4]
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = class_scores.argmax(1)
        scores = objectness * class_scores.max(1)

        # cxcywh -> xyxy and undo the letterbox scaling back to original image
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) / ratio
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) / ratio
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) / ratio
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) / ratio

        mask = scores >= threshold
        xyxy, scores, class_ids = xyxy[mask], scores[mask], class_ids[mask]

        results = []
        img_h, img_w = rgb.shape[:2]
        if len(scores):
            for i in self._nms(xyxy, scores, NMS_IOU_THRESHOLD):
                box = xyxy[i]
                xtl = min(max(int(box[0]), 0), img_w)
                ytl = min(max(int(box[1]), 0), img_h)
                xbr = min(max(int(box[2]), 0), img_w)
                ybr = min(max(int(box[3]), 0), img_h)
                if xbr <= xtl or ybr <= ytl:
                    # box lies (mostly) outside the image after rescaling — skip
                    continue
                confidence = float(scores[i])
                results.append(
                    {
                        "confidence": str(confidence),
                        "label": self.labels.get(int(class_ids[i]), "unknown"),
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                        # returned as an attribute so CVAT can store/display it
                        "attributes": [
                            {"name": "confidence", "value": f"{confidence:.4f}"},
                        ],
                    }
                )
        return results
