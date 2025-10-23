#!/bin/bash

case "$1" in
    "test-imports")
        echo "Testing all imports..."
        python test_imports.py
        ;;
    
    "generate-synthetic")
        echo "Generating synthetic dataset..."
        python scripts/generate_data.py \
            --output_dir data/synthetic \
            --num_train 70000 \
            --num_val 15000 \
            --num_test 15000 \
            --num_cells_min 50 \
            --num_cells_max 500 \
            --k_neighbors 10
        ;;
    
    "train-egnn")
        echo "Training EGNN model..."
        python scripts/train.py \
            --data_dir data/synthetic \
            --output_dir results/ \
            --model egnn \
            --hidden_channels 256 \
            --num_layers 5 \
            --dropout 0.15 \
            --epochs 200 \
            --batch_size 32 \
            --lr 0.001 \
            --weight_decay 0.0001
        ;;
    
    "evaluate")
        echo "Evaluating model..."
        python scripts/evaluate.py \
            --model_path results/best_model.pth \
            --data_dir data/synthetic \
            --output_dir results/evaluation/ \
            --batch_size 32
        ;;
    
    "pipeline-single")
        if [ -z "$2" ]; then
            echo "Usage: $0 pipeline-single <image_path>"
            exit 1
        fi
        echo "Running full pipeline on single image..."
        python scripts/pipeline_full.py \
            --model_path results/best_model.pth \
            --input "$2" \
            --output_dir output/ \
            --explain
        ;;
    
    "pipeline-batch")
        if [ -z "$2" ]; then
            echo "Usage: $0 pipeline-batch <images_dir>"
            exit 1
        fi
        echo "Running full pipeline on directory..."
        python scripts/pipeline_full.py \
            --model_path results/best_model.pth \
            --input "$2" \
            --output_dir output/ \
            --pattern "*.tif" \
            --explain
        ;;
    
    "explain")
        echo "Generating explanations..."
        python -m visualization.interpretability \
            --model_path results/best_model.pth \
            --data_dir data/test/ \
            --output_dir explanations/ \
            --method gradcam
        ;;
    
    "separate")
        if [ -z "$2" ]; then
            echo "Usage: $0 separate <mask_file>"
            exit 1
        fi
        echo "Separating organoids..."
        python -m utils.clustering \
            --mask_file "$2" \
            --output_dir separated/ \
            --eps 30.0 \
            --min_samples 20 \
            --min_cells 20 \
            --max_cells 5000
        ;;
    
    *)
        echo "Usage: $0 {command} [args]"
        echo ""
        echo "Commands:"
        echo "  test-imports                  - Test all module imports"
        echo "  generate-synthetic            - Generate synthetic dataset"
        echo "  train-egnn                    - Train EGNN model"
        echo "  evaluate                      - Evaluate trained model"
        echo "  pipeline-single <image>       - Process single image"
        echo "  pipeline-batch <dir>          - Process image directory"
        echo "  explain                       - Generate explanations"
        echo "  separate <mask>               - Separate organoids from mask"
        exit 1
        ;;
esac

