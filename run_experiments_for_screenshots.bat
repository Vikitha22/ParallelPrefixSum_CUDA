@echo off
title CUDA Parallel Prefix Sum - Experiment Execution
echo =====================================================================
echo  CUDA PARALLEL PREFIX SUM (SCAN) - EXPERIMENT RUNNER
echo  GPU: NVIDIA GeForce RTX 5060 Ti (16 GB)
echo  CUDA Toolkit: 13.4 | Driver: 610.88
echo =====================================================================
echo.

echo [1/8] Capturing CUDA Environment (nvidia-smi & nvcc --version)...
nvidia-smi
nvcc --version
echo.
echo -> Take screenshot: Screenshots\01_cuda_environment.png
pause

echo.
echo =====================================================================
echo [2/8] Running Hillis-Steele Inclusive Scan (8 elements)...
echo =====================================================================
parallel_scan.exe
echo.
echo -> Take screenshot: Screenshots\02_hillis_steele_inclusive.png
pause

echo.
echo =====================================================================
echo [3/8] Running Blelloch Exclusive Scan (8 elements)...
echo =====================================================================
blelloch_scan.exe
echo.
echo -> Take screenshot: Screenshots\03_blelloch_exclusive.png
pause

echo.
echo =====================================================================
echo [4/8] Running Multi-Block Blelloch Scan (N = 1,048,576)...
echo =====================================================================
large_scan.exe
echo.
echo -> Take screenshots:
echo    Screenshots\04_large_blelloch_output.png
echo    Screenshots\05_large_blelloch_performance.png
pause

echo.
echo =====================================================================
echo [5/8] Running Multi-Block Hillis-Steele Exclusive Scan (N = 1,048,576)...
echo =====================================================================
hillis_steele_large.exe
echo.
echo -> Take screenshots:
echo    Screenshots\06_large_hillis_steele_output.png
echo    Screenshots\07_large_hillis_steele_performance.png
pause

echo.
echo =====================================================================
echo [6/8] Running Block-Size Experiment (N = 65,536, Block Size = 128)...
echo =====================================================================
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -gencode=arch=compute_120,code=sm_120 -DBLOCK_SIZE=128 block_size_test.cu -o bs_128.exe
bs_128.exe
del bs_128.exe
echo.
echo -> Take screenshot: Screenshots\08_block_size_128.png
pause

echo.
echo =====================================================================
echo [7/8] Running Block-Size Experiment (N = 65,536, Block Size = 256, 512, 1024)...
echo =====================================================================
nvcc -gencode=arch=compute_120,code=sm_120 -DBLOCK_SIZE=256 block_size_test.cu -o bs_256.exe
bs_256.exe
del bs_256.exe
echo.
echo -> Take screenshot: Screenshots\09_block_size_256.png
pause

nvcc -gencode=arch=compute_120,code=sm_120 -DBLOCK_SIZE=512 block_size_test.cu -o bs_512.exe
bs_512.exe
del bs_512.exe
echo.
echo -> Take screenshot: Screenshots\10_block_size_512.png
pause

nvcc -gencode=arch=compute_120,code=sm_120 -DBLOCK_SIZE=1024 block_size_test.cu -o bs_1024.exe
bs_1024.exe
del bs_1024.exe
echo.
echo -> Take screenshot: Screenshots\11_block_size_1024.png
pause

echo.
echo =====================================================================
echo [8/8] Directory Tree Structure
echo =====================================================================
tree /F /A
echo.
echo -> Take screenshot: Screenshots\12_final_project_structure.png
echo.
echo All runs completed!
pause
