#!/bin/bash

# Loop through each directory in input2/
for dir in input2/*/; do
    # Remove trailing slash and extract directory name
    current_dir=$(basename "$dir")

    # Run the first Python script
    python run.py --input-path "input2/${current_dir}" \
                  --input-type image_sequence \
                  --save-folder "output_depth/${current_dir}_12-5-1400" \
                  --guidance-scale 1.2 \
                  --max-res 1400 \
                  --window-size 100 \
                  --overlap 50 \
                  --num-inference-steps 5 \
                  --target-fps 24

    # Run the second Python script
    python create_sbs_depthcrafters.py --input_rgb "input2/${current_dir}" \
                                       --input_depth "output_depth/${current_dir}_12-5-1400" \
                                       --output_dir "output_sbs/${current_dir}" \
                                       --deviation 16 \
                                       --blur 5 \
                                       --dilate 3 \
                                       --extend_depth 8 \
                                       --max_workers 4
done
