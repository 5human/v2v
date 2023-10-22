import numpy as np
import cv2
import moviepy.editor as mp
import os

from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


def read_video(video_path, skip, save=False, offset=0, resize_scale=1.0):
    cap = cv2.VideoCapture(video_path)  # NOQA
    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - offset  # NOQA

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # NOQA
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # NOQA

    buffer = np.empty((max_frame // skip + 1, height, width, 3), dtype=np.uint8)

    res, index = True, 0
    for i in tqdm(range(max_frame)):
        if not res:
            break

        res, frame = cap.read()

        if skip > 1 and i % skip != 0:
            continue

        frame = Image.fromarray(frame).resize((width, height))
        frame = np.asarray(frame, dtype=np.uint8)  # NOQA

        buffer[index] = frame
        index += 1

    if save:
        np.save("./temp/input.npy", buffer)

    return buffer


def video_liner(video_path, scale, skip, save=False, offset=0, resize_scale=1.0, sub_offset=0):
    cap = cv2.VideoCapture(video_path)  # NOQA
    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - offset - sub_offset  # NOQA

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale * scale)  # NOQA
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale * scale)  # NOQA

    # width = int(1920.0 * resize_scale * scale)
    # height = int(1080.0 * resize_scale * scale)

    buffer = np.empty((max_frame // skip, height, width, 3), dtype=np.uint8)

    res, index = True, 0
    for i in tqdm(range(max_frame)):
        if not res:
            break

        res, frame = cap.read()

        if (skip > 1 and i % skip != 0) or i < sub_offset:
            continue

        frame = Image.fromarray(frame).resize((width, height))
        frame = np.asarray(frame, dtype=np.uint8)  # NOQA

        if buffer.shape[0] > index:
            buffer[index] = frame
            index += 1

    buffer = np.moveaxis(buffer, 0, 1).reshape((height, width * buffer.shape[0], 3))

    if save:
        np.save("./temp/map.npy", buffer)

    return buffer


def get_best(frame: np.ndarray, pixels: np.ndarray, pixel_map: Image, repeat: int):
    height, width = frame.shape[:-1]
    frame = np.repeat(frame, repeat, axis=0).reshape((height, width * repeat, 3))

    canvas = Image.fromarray(frame)
    canvas.load()

    scores = np.array(
        canvas._new(canvas.im.chop_difference(pixel_map.im)),  # NOQA
        dtype=np.uint8
    )
    scores = np.split(scores, repeat, axis=1)

    index = np.argmin(np.sum(scores, axis=(1, 2, 3)))
    pixel = pixels[:, index * width:(index + 1) * width]
    pixel = np.asarray(pixel, dtype=np.uint8)

    return pixel


def build_video(video, pixels, scale, fps):
    def build_frame(f):
        tmp = np.empty(shape=f.shape, dtype=np.uint8)
        for y in range(scale):
            y_slice = slice(y * height, (y + 1) * height)

            for x in range(scale):
                x_slice = slice(x * width, (x + 1) * width)
                chunk = f[y_slice, x_slice]
                best = get_best(chunk, pixels, pixel_map, repeat)

                tmp[y_slice, x_slice] = best

        return tmp

    height, width = video.shape[1] // scale, video.shape[2] // scale
    repeat = pixels.shape[1] // width

    pixel_map = Image.fromarray(pixels)
    pixel_map.load()

    results = Parallel(n_jobs=-1)(
        delayed(build_frame)(frame) for frame in tqdm(video[:])
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # NOQA
    out = cv2.VideoWriter('./temp/output.mp4', fourcc, fps, (video.shape[2], video.shape[1]), True)  # NOQA

    for frame in tqdm(results):
        out.write(frame)  # NOQA

    out.release()


def v2v(video_path, source_path, video_scale):
    if not os.path.exists("./temp"):
        os.mkdir("./temp")

    # Change ================================
    fps = 30
    resize_scale = 1.0
    offset = 0
    sub_offset = 0

    read_video(video_path, 2, save=True, offset=offset, resize_scale=resize_scale)
    video_liner(source_path, 1 / video_scale, 6, save=True, offset=offset, resize_scale=resize_scale, sub_offset=sub_offset)

    # ========================================

    video = np.load('./temp/input.npy')
    pixels = np.load('./temp/map.npy')

    build_video(video, pixels, video_scale, fps)

    vid = mp.VideoFileClip("./temp/output.mp4")
    aud = mp.VideoFileClip(video_path).audio

    vid.audio = aud
    vid.write_videofile("result.mp4")


if __name__ == '__main__':
    v2v("./vids/bad apple.mp4", "./vids/bad apple.mp4", 50)



