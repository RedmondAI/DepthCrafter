import os
import subprocess
import sys

def combine_mp4s(directory):
    # List to store the paths of the processed videos
    processed_videos = []

    # Get a list of all folders in the directory
    folders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    folders.sort()  # Sort folders alphabetically

    # Iterate through each sorted folder
    for folder in folders:
        for file in os.listdir(folder):
            if file == "sbs_high_quality.mp4":
                video_path = os.path.join(folder, file)
                folder_name = os.path.basename(folder)
                
                # Create a temporary video with the folder name in the bottom left corner
                temp_video_path = os.path.abspath(os.path.join(folder, f"temp_{folder_name}.mp4"))  # Changed to absolute path
                
                # Check if the temporary video already exists
                if not os.path.exists(temp_video_path):
                    subprocess.run([
                        "ffmpeg", "-y", "-i", video_path, "-vf", 
                        f"drawtext=text='{folder_name}':fontcolor=yellow:fontsize=48:x=10:y=h-th-10",
                        "-codec:v", "libx264", "-crf", "12", "-preset", "slow", "-codec:a", "copy", temp_video_path
                    ])
                
                # Add the path of the temporary video to the list
                processed_videos.append(temp_video_path)

    # Create a text file with the list of processed videos
    concat_list_path = os.path.join(directory, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for video in processed_videos:
            f.write(f"file '{video}'\n")

    # Combine all processed videos into a single large MP4 video using concat filter
    output_video_path = os.path.join(directory, "combined_video.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", output_video_path
    ])

    # Clean up temporary videos and concat list
    # for video in processed_videos:
    #     os.remove(video)
    # os.remove(concat_list_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python combine_mp4s.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    combine_mp4s(directory)

# python create_sbs_depthcrafters.py --input_rgb "quietplace_input/scene_00030" --input_depth "quietplace_depth/scene_00030_11-5-1400" --output_dir "quietplace_sbs/scene_00030" --deviation 14 --blur 5 --dilate 2 --extend_depth 4 --max_workers 4
# python run.py --input-path "quietplace_input/scene_00030" --input-type image_sequence --save-folder "quietplace_depth/scene_00030_11-5-1400" --guidance-scale 1.1 --max-res 1400 --window-size 100 --overlap 50 --num-inference-steps 35 --target-fps 24