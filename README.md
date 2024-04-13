# Canny-Edge-Detection-with-Numba

## Black and White transformation

The first step  of the project, consist in transform the image in black and white. 

- [ ] Check the version of the Code, if he fit with the number of channels of the image

## Gaussian blur

The second step of the project, consist in apply a gaussian blur to the image.

- [X] Fix sanitizer error, ask to the teacher

The implementation of the gaussian blur is in the file `gaussian_blur.py`

## Sobel Filter

The third step of the project, consist in apply the sobel filter to the image, and extract  
the magnitude and the direction of the gradient. 

- [X] Fix sanitizer error, ask to the teacher

The implementation of the sobel filter is in the file `sobel.py`

## Thresholding

The fourth step of the project, we need to find the potential edges based on thresholds


## Hysterisis Thresholding

The fifth step of the project, we need to find the edges based on the hysteresis thresholding

## Tools

To check if the implementation was correct and memory efficient, we use 
the (compute-sanitizer)[https://docs.nvidia.com/cuda/compute-sanitizer/index.html]  tool provided by 
Nvidia.

Compute Sanitizer provide different checking mechanisms through different tools.   
Currently the supported tools are: 

- Memcheck - memory acces error and leak detection tool
- Racecheck - shared memory data acces hazard detection tool
- Initcheck - uninitialized device global memory access detection tool
- Synccheck - thread synchronization hazard detection tool

All those tools are used sequentially after launching the program with the following command: 

```bash
/usr/local/cuda/bin/compute-sanitizer python3 my_file.py
```
but is hard to detect from where the error is coming from. So the best way to use the tool   
is to launch the compute-sanityzer without parameters the first time and if an error is detected, 
launch the tool with the specific tool that detected the error. 

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck python3 my_file.py
/usr/local/cuda/bin/compute-sanitizer --tool racecheck python3 my_file.py
/usr/local/cuda/bin/compute-sanitizer --tool initcheck python3 my_file.py
/usr/local/cuda/bin/compute-sanitizer --tool synccheck python3 my_file.py
```

Normally the error are caused by the way we acces to memory or check the bounds of the array.    
So our style of programming should be smart and efficient.

## Idea TODO 

If you have a Idea to enhance the project, please write it here.