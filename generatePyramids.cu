//Desription: Image processing algorithm for "pyramid generation". See "Pyramidal Implementation of the Lucas Kanade Feature Tracker Description of the algorithm" for a detailed description
//It uses a “tiled convolution” structure, where each block is responsible for generating a corresponding “tile” on the pyramid image.

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

//Time stamp function in seconds
double getTimeStamp() {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

//When generating multiple levels of pyramids, i corresponds to the pyramid level the current kernel call is generating
__global__ void generatePyramid (float *pyramids, int i, int origImgWidth, int origImgHeight) {
  int ix = threadIdx.x + blockIdx.x*blockDim.x;
  int iy = threadIdx.y + blockIdx.y*blockDim.y ;

  int origImgSize = origImgWidth*origImgHeight;
  
  int imgOffset = int(origImgSize * (1-pow(0.25, i-1))/(1-0.25));
  int pyramidOffset = int(origImgSize * (1-pow(0.25, i))/(1-0.25));

  int imgHeight = origImgHeight >> (i-1);
  int imgWidth = origImgWidth >> (i-1);
  
  int pyrmHeight = imgHeight >> 1;
  int pyrmWidth = imgWidth >> 1;
  
  int idx = iy* pyrmWidth + ix ;
  
  extern __shared__ float sImg [];

  //Move data block uses to shared memory for faster reads
  //center
  sImg[threadIdx.x*2 + 1 + (threadIdx.y*2 + 1)*(blockDim.x*2 + 1)] = pyramids[imgOffset + iy*2*imgWidth + ix*2];
  //center right
  sImg[threadIdx.x*2 + 2 + (threadIdx.y*2 + 1)*(blockDim.x*2 + 1)] = pyramids[imgOffset + iy*2*imgWidth + ix*2 + 1];
  //bottom center
  sImg[threadIdx.x*2 + 1 + (threadIdx.y*2 + 2)*(blockDim.x*2 + 1)] = pyramids[imgOffset + (iy*2 + 1)*imgWidth + ix*2];
  //bottom right
  sImg[threadIdx.x*2 + 2 + (threadIdx.y*2 + 2)*(blockDim.x*2 + 1)] = pyramids[imgOffset + (iy*2 + 1)*imgWidth + ix*2 + 1];
  
  if (threadIdx.y == 0) {
     //top center
     sImg[threadIdx.x*2 + 1 + (threadIdx.y*2)*(blockDim.x*2 + 1)] = blockIdx.y == 0 ? pyramids[imgOffset + iy*2*imgWidth + ix*2]:pyramids[imgOffset + (iy*2-1)*imgWidth + ix*2];
     //top right
     sImg[threadIdx.x*2 + 2 + (threadIdx.y*2)*(blockDim.x*2 + 1)] = blockIdx.y == 0 ? pyramids[imgOffset + iy*2*imgWidth + ix*2 + 1]:pyramids[imgOffset + (iy*2-1)*imgWidth + ix*2 + 1];	  
  }
  if (threadIdx.x == 0) {
     //center left
     sImg[threadIdx.x*2 + (threadIdx.y*2 + 1)*(blockDim.x*2 + 1)] = blockIdx.x == 0 ? pyramids[imgOffset + iy*2*imgWidth + ix*2]:pyramids[imgOffset + iy*2*imgWidth + ix*2 - 1];
     //bottom left
     sImg[threadIdx.x*2 + (threadIdx.y*2 + 2)*(blockDim.x*2 + 1)] = blockIdx.x == 0 ? pyramids[imgOffset + (iy*2 + 1)*imgWidth + ix*2]:pyramids[imgOffset + (iy*2 + 1)*imgWidth + ix*2 - 1];	  
  }
  if (threadIdx.x == 0 && threadIdx.y == 0)
     //top left
     sImg[threadIdx.x*2 + (threadIdx.y*2)*(blockDim.x*2 + 1)] = (blockIdx.x == 0 || blockIdx.y == 0) ? pyramids[imgOffset + iy*2*imgWidth + ix*2]:pyramids[imgOffset + (iy*2-1)*imgWidth + ix*2 - 1];
   
  __syncthreads();
    
  if( (ix<pyrmWidth) && (iy<pyrmHeight) ) {
   
      #ifdef DEBUG
	 int centerX = min(max(2*ix, 0), imgWidth);
         int centerY = min(max(2*iy, 0), imgHeight);

         int left = min(max(2*ix - 1, 0), imgWidth);
         int down = min(max(2*iy - 1, 0), imgHeight);

         int right = min(max(2*ix + 1, 0), imgWidth);
         int up = min(max(2*iy + 1, 0), imgHeight);
	  
         printf("Index: (%d, %d)\n", ix, iy);
      	 printf("Center (%d, %d): %lf\n", centerX, centerY, pyramids[centerX + centerY*imgWidth + imgOffset]);
         printf("Center Left (%d, %d): %lf\n", left, centerY, pyramids[left + centerY*imgWidth + imgOffset]);
         printf("Center Right (%d, %d): %lf\n", right, centerY, pyramids[right + centerY*imgWidth + imgOffset]);	    
         printf("Up Left (%d, %d): %lf\n", left, up, pyramids[left + up*imgWidth + imgOffset]);
         printf("Up Center (%d, %d): %lf\n", centerX, up, pyramids[centerX + up*imgWidth + imgOffset]);	    
         printf("Up Right (%d, %d): %lf\n", right, up, pyramids[right + up*imgWidth + imgOffset]);
         printf("Down Left (%d, %d): %lf\n", left, down, pyramids[left + down*imgWidth + imgOffset]);
         printf("Down Center (%d, %d): %lf\n", centerX, down, pyramids[centerX + down*imgWidth + imgOffset]);	    
         printf("Down Right (%d, %d): %lf\n", right, down, pyramids[right + down*imgWidth + imgOffset]);
      #endif

      float pValue = 0;
      
      pValue += sImg[threadIdx.x*2 + 1 + (threadIdx.y*2 + 1)*(blockDim.x*2 + 1)]/4.0;
      pValue += 1/8*sImg[threadIdx.x*2 + 2 + (threadIdx.y*2 + 1)*(blockDim.x*2 + 1)]/8.0;
      pValue += 1/8*sImg[threadIdx.x*2 + 1 + (threadIdx.y*2 + 2)*(blockDim.x*2 + 1)]/8.0;
      pValue += 1/16*sImg[threadIdx.x*2 + 2 + (threadIdx.y*2 + 2)*(blockDim.x*2 + 1)]/16.0;
      pValue += 1/8*sImg[threadIdx.x*2 + 1 + (threadIdx.y*2)*(blockDim.x*2 + 1)]/8.0;
      pValue += 1/16*sImg[threadIdx.x*2 + 2 + (threadIdx.y*2)*(blockDim.x*2 + 1)]/16.0;
      pValue += 1/8*sImg[threadIdx.x*2 + (threadIdx.y*2 + 1)*(blockDim.x*2 + 1)]/8.0;
      pValue += 1/16*sImg[threadIdx.x*2 + (threadIdx.y*2 + 2)*(blockDim.x*2 + 1)]/16.0;
      pValue += 1/16*sImg[threadIdx.x*2 + (threadIdx.y*2)*(blockDim.x*2 + 1)]/16.0;

      pyramids[idx + pyramidOffset] = pValue;
  }
}
int main() {
   FILE *img = fopen("img.bmp", "rb");

   unsigned char info[54];
   fread(info, sizeof(unsigned char), 54, img); // read the 54-byte header

   //Extract image height and width from header
   int imgWidth = *(int*)&info[18];
   int imgHeight = *(int*)&info[22];
   printf("Size: %d %d\n", imgWidth, imgHeight);
   
   int imgSize = imgWidth * imgHeight;
   unsigned char* data = (unsigned char*)malloc(3*imgSize*sizeof(unsigned char)); // allocate 3 bytes per pixel
   fread(data, sizeof(unsigned char), 3*imgSize, img); // read the rest of the data at once
   fclose(img);

   int pyramidLevels = 3;

   //In terms of #elements
   //Geometric series formula
   int pyramidsSize = int(imgSize * (1-pow(0.25, pyramidLevels+1))/(1-0.25)); 

   float* h_pyramids;
   cudaHostAlloc( (void **) &h_pyramids, pyramidsSize*sizeof(float), 0) ;

   //Init data...
   for (int i = 0; i < imgHeight; i++) {
      for (int j = 0; j < imgWidth; j++) {
         h_pyramids[i*imgWidth + j] =  data[((imgHeight - 1 - i)*imgWidth + j)*3];
      }
   }
   
   float* d_pyramids;
   cudaMalloc((void**) &d_pyramids, pyramidsSize*sizeof(float));
	     
   double timeStampA = getTimeStamp() ;
      
   cudaMemcpy(d_pyramids, h_pyramids, imgSize*sizeof(float), cudaMemcpyHostToDevice);
   double timeStampB = getTimeStamp() ;
   
   dim3 block(16, 16);
   
   for (int i = 1; i <= pyramidLevels; i++) {
       int gridX = ceil((imgWidth >> i)/block.x);
       int gridY = ceil((imgWidth >> i)/block.y);
       
       dim3 grid(gridX, gridY);
       generatePyramid<<<grid, block, (2*block.x + 1) * (2*block.y + 1)*sizeof(float)>>> (d_pyramids, i, imgWidth, imgHeight);		 
   }
   
   cudaDeviceSynchronize();
   double timeStampC = getTimeStamp() ;
      
   cudaMemcpy (h_pyramids, d_pyramids, pyramidsSize*sizeof(float), cudaMemcpyDeviceToHost);

   double timeStampD = getTimeStamp();
   
   cudaError_t err = cudaPeekAtLastError();

   if (err != cudaSuccess) {
      printf("Error: %s", cudaGetErrorString(err));
      exit(-1);
   }

   cudaFree(d_pyramids);
   cudaFreeHost(h_pyramids);
   cudaDeviceReset();

   printf("\n\n\n%.6f\n", timeStampB-timeStampA);
   printf("%.6f\n", timeStampC-timeStampA);
   printf("%.6f\n", timeStampD-timeStampC);
}
