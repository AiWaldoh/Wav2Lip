import argparse
import os


class Arguments:
    def __init__(self, checkpoint_path, audio_path, video_frames_path, output_path):
        self.parser = argparse.ArgumentParser(
            description="Inference code to lip-sync videos in the wild using Wav2Lip models"
        )
        self.checkpoint_path = checkpoint_path
        self.audio_path = audio_path
        self.video_frames_path = video_frames_path
        self.output_path = output_path
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument(
            "--checkpoint_path",
            type=str,
            default=self.checkpoint_path,
            help="Path to the Wav2Lip model checkpoint",
        )

        self.parser.add_argument(
            "--audio",
            type=str,
            default=self.audio_path,
            help="Path to the input audio file",
        )

        self.parser.add_argument(
            "--face",
            type=str,
            default=self.video_frames_path,
            help="Path to the input face image or video frames directory",
        )

        self.parser.add_argument(
            "--outfile",
            type=str,
            default=self.output_path,
            help="Path to the output lip-synchronized video file",
        )

        self.parser.add_argument(
            "--nosmooth",
            default=False,
            action="store_true",
            help="Prevent smoothing face detections over a short temporal window",
        )

        self.parser.add_argument(
            "--static",
            type=bool,
            help="If True, then use only first video frame for inference",
            default=False,
        )

        self.parser.add_argument(
            "--fps",
            type=float,
            help="Can be specified only if input is a static image (default: 25)",
            default=25.0,
            required=False,
        )

        self.parser.add_argument(
            "--pads",
            nargs="+",
            type=int,
            default=[0, 10, 0, 0],
            help="Padding (top, bottom, left, right). Please adjust to include chin at least",
        )

        self.parser.add_argument(
            "--face_det_batch_size",
            type=int,
            help="Batch size for face detection",
            default=16,
        )
        self.parser.add_argument(
            "--wav2lip_batch_size",
            type=int,
            help="Batch size for Wav2Lip model(s)",
            default=128,
        )

        self.parser.add_argument(
            "--resize_factor",
            default=1,
            type=int,
            help="Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p",
        )

        self.parser.add_argument(
            "--crop",
            nargs="+",
            type=int,
            default=[0, -1, 0, -1],
            help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. "
            "Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width",
        )

        self.parser.add_argument(
            "--box",
            nargs="+",
            type=int,
            default=[-1, -1, -1, -1],
            help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected."
            "Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).",
        )

        self.parser.add_argument(
            "--rotate",
            default=False,
            action="store_true",
            help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg."
            "Use if you get a flipped result, despite feeding a normal looking video",
        )

    def parse_args(self):
        args = self.parser.parse_args()
        args.img_size = 96

        if os.path.isfile(args.face) and args.face.split(".")[1] in [
            "jpg",
            "png",
            "jpeg",
        ]:
            args.static = True
        else:
            args.static = False

        return args
