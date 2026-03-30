import os
import subprocess
from tqdm import tqdm

class MovieMaker:
    """Manages video generation using ffmpeg."""
    def __init__(self, frame_rate=10, height=800):
        self.frame_rate = frame_rate
        self.height = height

    def create_side_by_side(self, frame_dir, output_path, 
                            sim_pattern="frame-%03d.png", 
                            viz_pattern="viz-%03d.png"):
        """Combines simulation and visualization frames into a side-by-side video."""
        cmd = [
            'ffmpeg', '-y',
            '-r', str(self.frame_rate), '-i', os.path.join(frame_dir, sim_pattern),
            '-r', str(self.frame_rate), '-i', os.path.join(frame_dir, viz_pattern),
            '-filter_complex', f'[0:v]scale=-1:{self.height}[v0];[1:v]scale=-1:{self.height}[v1];[v0][v1]hstack',
            '-vcodec', 'mpeg4',
            output_path
        ]
        
        # Run ffmpeg and suppress output unless there's an error
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            print(f"Error: ffmpeg failed to create {output_path}")
            return False

    def create_single(self, frame_dir, output_path, pattern="frame-%03d.png"):
        """Creates a video from a single sequence of frames."""
        cmd = [
            'ffmpeg', '-y',
            '-r', str(self.frame_rate), '-i', os.path.join(frame_dir, pattern),
            '-vcodec', 'mpeg4',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            print(f"Error: ffmpeg failed to create {output_path}")
            return False
