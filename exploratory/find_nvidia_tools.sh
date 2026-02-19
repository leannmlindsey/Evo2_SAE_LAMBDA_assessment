#!/bin/bash
# Find NVIDIA profiling tools on the system

echo "Searching for nsys (Nsight Systems)..."
find /usr /opt /usr/local -name "nsys" -type f 2>/dev/null | head -5

echo ""
echo "Searching for ncu (Nsight Compute)..."
find /usr /opt /usr/local -name "ncu" -type f 2>/dev/null | head -5

echo ""
echo "Searching for CUDA toolkit directories..."
ls -d /usr/local/cuda* 2>/dev/null
ls -d /opt/nvidia* 2>/dev/null

echo ""
echo "Checking common locations..."
for path in \
    /usr/local/cuda/bin \
    /usr/local/cuda-12*/bin \
    /opt/nvidia/nsight-systems/*/bin \
    /opt/nvidia/nsight-compute/*/bin \
    /usr/local/NVIDIA-Nsight-Systems*/bin \
    /usr/local/NVIDIA-Nsight-Compute*/bin; do
    if [ -d "$path" ]; then
        echo "Found: $path"
        ls "$path" | grep -E "^(nsys|ncu)$" 2>/dev/null
    fi
done

echo ""
echo "Checking module system (if available)..."
if command -v module &> /dev/null; then
    echo "Module system available. Try:"
    echo "  module avail nsight"
    echo "  module avail cuda"
    module avail 2>&1 | grep -i nsight
fi

echo ""
echo "============================================"
echo "Once found, add to PATH with:"
echo "  export PATH=/path/to/nsight/bin:\$PATH"
echo "============================================"
