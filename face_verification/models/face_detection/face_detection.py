from typing import Callable

import cv2
import torch


def create_face_detector(
    image_size: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def detect_face(image_batch: torch.Tensor) -> torch.Tensor:
        predictions = []

        for image in image_batch:
            permuted_image = image.permute(1, 2, 0)
            grayscale_image = cv2.cvtColor(permuted_image, cv2.COLOR_RGB2GRAY)
            x, y, w, h = detector.detectMultiScale(grayscale_image)[0]
            face = permuted_image[x:x + w, y:y + h]
            resized_face = cv2.resize(face, image_size)
            prediction = resized_face.permute(0, 1, 2)
            predictions.append(prediction)

        return torch.stack(predictions)

    return detect_face
