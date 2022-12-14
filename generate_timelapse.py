#!/usr/bin/env python
import argparse
import datetime
import json
import os
import time

import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from utils.argparse import get_args_parser
from utils.distributed import *


def custom_loss(x, y, lmbd=1, device="cuda"):
    # loss0 = torch.mean((x - y)**2, axis=[1,2,3]).sum()
    # loss1 = torch.mean((y[1:]-y[:-1])**2, axis=[1,2,3]).sum()

    # loss0 = torch.mean(torch.abs(x - y), axis=[1,2,3]).sum()
    # loss1 = torch.mean(torch.abs(y[1:]-y[:-1]), axis=[1,2,3]).sum()

    loss0 = torch.nn.HuberLoss(reduction="sum", delta=4)(
        y - x, torch.zeros(x.shape).to(device)
    )
    loss1 = torch.nn.HuberLoss(reduction="sum", delta=1)(
        y[1:] - y[:-1],
        torch.zeros(y.shape[0] - 1, y.shape[1], y.shape[2], y.shape[3]).to(device),
    )

    loss = loss0 + lmbd * loss1

    return loss


def main(args):

    output_dir = os.path.join("outputs", args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    for index in range(args.top_k):

        os.makedirs(
            os.path.join(output_dir, str(args.cluster_id), str(index)), exist_ok=True
        )

        numpy_dir = os.path.join("numpy_arrays", args.dataset, str(args.cluster_id))
        image_stack_homography_alignment = np.load(
            os.path.join(numpy_dir, f"image_stack_homography_alignment_{index}.npy")
        )

        num_images, height, width, _ = image_stack_homography_alignment.shape
        crop_height = int(height * 0.15)
        crop_width = int(width * 0.15)

        image_stack_homography_alignment = image_stack_homography_alignment[
            :, crop_height:-crop_height, crop_width:-crop_width, :
        ]

        print("[INFO] Saving Timelapse 0")
        imageio.mimsave(
            os.path.join(
                output_dir, str(args.cluster_id), str(index), "timelapse0.gif"
            ),
            image_stack_homography_alignment,
            fps=args.fps,
        )

        image_stack_homography_alignment_median_smoothed = []
        image_stack_homography_alignment_mean_smoothed = []
        moving_window_length = 10
        moving_window = list(image_stack_homography_alignment[:moving_window_length])
        image_stack_homography_alignment_median_smoothed.append(
            np.median(moving_window, axis=0)
        )
        image_stack_homography_alignment_mean_smoothed.append(
            np.median(moving_window, axis=0)
        )

        for image in image_stack_homography_alignment[moving_window_length:]:
            moving_window = moving_window[1:]
            moving_window.append(image)
            image_stack_homography_alignment_median_smoothed.append(
                np.median(moving_window, axis=0)
            )
            image_stack_homography_alignment_mean_smoothed.append(
                np.mean(moving_window, axis=0)
            )

        image_stack_homography_alignment_median_smoothed = np.array(
            image_stack_homography_alignment_median_smoothed
        ).astype(np.uint8)
        image_stack_homography_alignment_mean_smoothed = np.array(
            image_stack_homography_alignment_mean_smoothed
        ).astype(np.uint8)

        print("[INFO] Saving Timelapse 1")
        imageio.mimsave(
            os.path.join(
                output_dir, str(args.cluster_id), str(index), "timelapse1.gif"
            ),
            image_stack_homography_alignment_median_smoothed,
            fps=args.fps,
        )

        print("[INFO] Saving Timelapse 2")
        imageio.mimsave(
            os.path.join(
                output_dir, str(args.cluster_id), str(index), "timelapse2.gif"
            ),
            image_stack_homography_alignment_mean_smoothed,
            fps=args.fps,
        )

        # img = cv2.imread("path/to/Lenna.png")

        image_stack_homography_alignment_normalized = []

        for image in image_stack_homography_alignment:
            img_y_cr_cb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
            y, cr, cb = cv.split(img_y_cr_cb)

            # Applying equalize Hist operation on Y channel.
            y_eq = cv.equalizeHist(y)

            img_y_cr_cb_eq = cv.merge((y_eq, cr, cb))
            img_rgb_eq = cv.cvtColor(img_y_cr_cb_eq, cv.COLOR_YCR_CB2BGR)
            image_stack_homography_alignment_normalized.append(img_rgb_eq)
        print("[INFO] Saving Timelapse 3")
        image_stack_homography_alignment_normalized = np.array(
            image_stack_homography_alignment_normalized
        ).astype(np.uint8)
        imageio.mimsave(
            os.path.join(
                output_dir, str(args.cluster_id), str(index), "timelapse3.gif"
            ),
            image_stack_homography_alignment_normalized,
            fps=args.fps,
        )

        indices = []
        for i in range(0, num_images, 100):
            indices.append([i, min(i + 100, num_images)])

        final_timelapse = torch.tensor(image_stack_homography_alignment)
        for index_split in indices:
            print("index_split", index_split)
            left, right = index_split

            x = torch.tensor(final_timelapse[left:right])
            y = x.detach().clone().float()

            x = x.to(device)
            y = y.to(device)

            y.requires_grad = True

            optimizer = torch.optim.Adam([y], lr=1)
            optimizer.zero_grad()
            # losses = []
            for _ in tqdm(range(int(args.num_steps))):
                # loss = custom_loss(x, y, lmbd=args.lmbd, device=device)
                loss = custom_loss(x, y, lmbd=args.lmbd)
                # losses.append(loss.item())
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            final_timelapse[left:right] = y.detach().cpu()
            ## plot losses
            # fig, axs = plt.subplots(1, 1, squeeze=False)
            # fig.set_figheight(8)
            # fig.set_figwidth(20)

            # axs[0][0].plot(np.arange(num_steps), losses)
            # axs[0][0].set_title('Loss vs. Step')
            # axs[0][0].set_xlabel('Step')
            # axs[0][0].set_ylabel('Loss')
            # # axs[0][0].set_yscale('log')
            # axs[0][0].set_xticks(np.arange(0, num_steps, num_steps//10))
            # axs[0][0].grid(True)
            # plt.savefig(os.path.join(plot_dir, str(index), 'losses.png'))
            # plt.close()

        image_stack_homography_alignment_smoothed = (
            y.cpu().detach().numpy().astype(np.uint8)
        )
        image_stack_homography_alignment_smoothed = np.clip(
            image_stack_homography_alignment_smoothed, 0, 255
        )
        print("[INFO] Saving Timelapse 4")
        imageio.mimsave(
            os.path.join(
                output_dir, str(args.cluster_id), str(index), "timelapse4.gif"
            ),
            image_stack_homography_alignment_smoothed,
            fps=args.fps,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Timelapse", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
