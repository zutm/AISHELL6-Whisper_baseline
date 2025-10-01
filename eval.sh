#!/bin/bash

# =============================================================================
# Configuration Parameters - Users can modify these variables as needed
# =============================================================================

# Basic configuration
CHECKPOINT_PATH="models/checkpoint/whisper_zh_largev3_AISHELL6-Whisper/last.ckpt"
LANGUAGE="zh"
MODALITIES="avsr"  # Options: "asr" or "avsr"
MODEL_TYPE="large-v3"
NOISE_SNR=1000
BEAM_SIZE=1
FP16=1  # Set to 0 if GPU doesn't support fp16 or using CPU

# AV-HuBERT related configuration (only used when modalities=avsr)
USE_AV_HUBERT_ENCODER=1
AV_FUSION="separate"
AV_HUBERT_PATH="av_hubert/avhubert/"
AV_HUBERT_CKPT="models/large_noise_pt_noise_ft_433h_only_weights.pt"

# Decoding options
ALIGN=1  # Set to 1 for Whisper speech data
NOISE_FILE="data/noise.tsv"

# Output directory
BASE_OUTPUT_DIR="decode"

# =============================================================================
# Parameter Validation
# =============================================================================

if [ "$MODALITIES" != "asr" ] && [ "$MODALITIES" != "avsr" ]; then
    echo "Error: MODALITIES must be either 'asr' or 'avsr'"
    exit 1
fi

if [ "$MODALITIES" = "avsr" ] && [ $USE_AV_HUBERT_ENCODER -eq 0 ]; then
    echo "Warning: Using AVSR mode but AV-HuBERT encoder is not enabled"
fi

# =============================================================================
# Build Decoding Command
# =============================================================================

decode_command="python -u whisper_decode_video.py \
    --lang $LANGUAGE \
    --model-type $MODEL_TYPE \
    --noise-snr $NOISE_SNR \
    --noise-fn $NOISE_FILE \
    --modalities $MODALITIES \
    --checkpoint-path $CHECKPOINT_PATH \
    --fp16 $FP16"

# Add AV-HuBERT related parameters (only in AVSR mode)
if [ "$MODALITIES" = "avsr" ]; then
    decode_command="$decode_command \
        --use_av_hubert_encoder $USE_AV_HUBERT_ENCODER \
        --av_fusion $AV_FUSION \
        --av-hubert-path $AV_HUBERT_PATH \
        --av-hubert-ckpt $AV_HUBERT_CKPT"
fi

# Add alignment parameter
if [ $ALIGN -eq 1 ]; then
    decode_command="$decode_command --align"
fi

# =============================================================================
# Execute Decoding
# =============================================================================

echo "Starting decoding process..."
echo "Command: $decode_command"
eval $decode_command

if [ $? -ne 0 ]; then
    echo "Decoding process failed!"
    exit 1
fi

# =============================================================================
# Build Output Directory Path and Compute CER
# =============================================================================

# Extract model name from checkpoint path
model_name=$(basename $(dirname $CHECKPOINT_PATH))
DECODE_OUTPUT_DIR="$BASE_OUTPUT_DIR/$CHECKPOINT_PATH/$LANGUAGE/test/$MODALITIES/snr-$NOISE_SNR/visible-0/beam-$BEAM_SIZE/data"

echo "Output directory: $DECODE_OUTPUT_DIR"

# Check if output files exist
if [ ! -f "${DECODE_OUTPUT_DIR}/ref.txt" ]; then
    echo "Error: Reference file not found: ${DECODE_OUTPUT_DIR}/ref.txt"
    exit 1
fi

if [ ! -f "${DECODE_OUTPUT_DIR}/hypo.txt" ]; then
    echo "Error: Hypothesis file not found: ${DECODE_OUTPUT_DIR}/hypo.txt"
    exit 1
fi

# =============================================================================
# Compute CER
# =============================================================================

echo "Computing Character Error Rate (CER)..."
python tools/compute-cer.py --char=1 --v=1 \
    "${DECODE_OUTPUT_DIR}/ref.txt" \
    "${DECODE_OUTPUT_DIR}/hypo.txt" \
    > "${DECODE_OUTPUT_DIR}/cer_result.txt"

if [ $? -eq 0 ]; then
    echo "CER computation completed successfully!"
    echo "Results saved to: ${DECODE_OUTPUT_DIR}/cer_result.txt"
    
    # Display CER results
    echo "=== CER Results ==="
    cat "${DECODE_OUTPUT_DIR}/cer_result.txt"
else
    echo "CER computation failed!"
    exit 1
fi