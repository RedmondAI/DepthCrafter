#!/bin/bash

# Get the input directory name from the command line argument
input_dir=$1

# Loop through each directory in input2/
for dir in input_${input_dir}/*/; do
    # Remove trailing slash and extract directory name
    current_dir=$(basename "$dir")
    echo "Processing directory: ${current_dir}"

    # Check if the output directory already exists
    if [ -d "${input_dir}_depth/${current_dir}_11-5-1400" ]; then
        echo "Skipping ${current_dir}, output already exists."
        continue
    fi

    # Run the first Python script
    python run.py --input-path "${input_dir}_input/${current_dir}" \
                  --input-type image_sequence \
                  --save-folder "${input_dir}_depth/${current_dir}_11-5-1400" \
                  --guidance-scale 1.1 \
                  --max-res 1400 \
                  --window-size 100 \
                  --overlap 50 \
                  --num-inference-steps 5 \
                  --target-fps 24

    # Run the second Python script
    python create_sbs_depthcrafters.py --input_rgb "${input_dir}_input/${current_dir}" \
                                       --input_depth "${input_dir}_depth/${current_dir}_12-5-1400" \
                                       --output_dir "${input_dir}_sbs/${current_dir}" \
                                       --deviation 14 \
                                       --blur 5 \
                                       --dilate 2 \
                                       --extend_depth 4 \
                                       --max_workers 4
done
