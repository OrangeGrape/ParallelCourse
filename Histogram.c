#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)
//#include "utils.h"
//#include "bmp-utils.h"

static const int HIST_BINS = 256;

int main(int argc , char **argv)
{
  //int * hInputImage = NULL;
  int * hOutputHistogram = NULL;
 
  //int imageRows ;
  //int imageCols ;
  //hInputImage = readBmp("./cat.dat", &imageRows, &imageCols) ;
  //const int imageElements = imageRows * imageCols ;
  //const size_t imageSize = imageElements * sizeof(int);

///////////////////////////////////////////////////////////////////////////////////
  int imageRows = 700 ;
  int imageCols = 1000 ;
  const int imageElements = imageRows * imageCols ;
  const size_t imageSize = imageElements * sizeof(int);
  
  int *hInputImage = (int*)malloc(imageSize);
  FILE *file = fopen("cat.dat","r");
  fread(hInputImage, sizeof(int), imageRows * imageCols, file);
  fclose(file);
///////////////////////////////////////////////////////////////////////////////////

  const int histogramSize = HIST_BINS * sizeof(int);
  hOutputHistogram = (int*)malloc(histogramSize);
  if(!hOutputHistogram){ exit(-1) ; }

  cl_int status ;

  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform , NULL);
  //check( status ) ;

  cl_device_id device;
  status = clGetDeviceIDs ( platform , CL_DEVICE_TYPE_GPU , 1 , &device , NULL) ;
  //check( status ) ;

  cl_context context;
  context = clCreateContext(NULL, 1, &device , NULL, NULL, &status );
  //check( status ) ;

  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue( context , device , 0, &status ) ;
  //check( status ) ;

  cl_mem bufInputImage ;
  bufInputImage = clCreateBuffer (context , CL_MEM_READ_ONLY, imageSize , NULL, &status);
  //check( status ) ;

  cl_mem bufOutputHistogram ;
  bufOutputHistogram = clCreateBuffer (context , CL_MEM_WRITE_ONLY, histogramSize , NULL, &status ) ;
  //check( status ) ;
  status = clEnqueueWriteBuffer(cmdQueue, bufInputImage , CL_TRUE,0, imageSize ,hInputImage , 0,NULL,NULL) ;
  //check( status ) ;

  int  zero = 0;
  status = clEnqueueFillBuffer(cmdQueue, bufOutputHistogram , &zero ,sizeof(int) , 0, histogramSize , 0, NULL, NULL) ;
  //check( status ) ;

  //char * programSource = readFile ("histogram.cl") ;
///////////////////////////////////////////////////////////////////////////////////

  FILE *fp;
  const char fileName[] = "./histogram.cl";
  size_t source_size;
  char *programSource;
  fp = fopen(fileName,"r");
  programSource = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(programSource,1,MAX_SOURCE_SIZE,fp);
  fclose(fp);

///////////////////////////////////////////////////////////////////////////////////


  size_t programSourceLen = strlen (programSource) ;
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, &programSourceLen, &status ) ;
  //check( status ) ;

  status = clBuildProgram( program , 1 , &device , NULL, NULL, NULL) ;
  if (status != CL_SUCCESS) {
    printCompilerError(program , device);
    exit (-1) ;
  }

  cl_kernel kernel;
  kernel = clCreateKernel(program , "histogram", &status );
  //check( status ) ;

  status = clSetKernelArg(kernel , 0,sizeof(cl_mem) , &bufInputImage) ;
  status |= clSetKernelArg(kernel , 1,sizeof(int) , &imageElements) ;
  status |= clSetKernelArg(kernel , 2,sizeof(cl_mem) , &bufOutputHistogram) ;
  //check( status ) ;

  size_t globalWorkSize [1];
  globalWorkSize [0] = 1024;

  size_t localWorkSize [1];
  localWorkSize [0] = 64;

  status = clEnqueueNDRangeKernel(cmdQueue , kernel , 1, NULL,globalWorkSize , localWorkSize , 0, NULL, NULL) ;
  //check( status ) ;

  status = clEnqueueReadBuffer(cmdQueue, bufOutputHistogram ,CL_TRUE , 0 ,histogramSize , hOutputHistogram , 0, NULL, NULL) ;
  //check( status ) ;

  clReleaseKernel(kernel);
  clReleaseProgram(program) ;
  clReleaseCommandQueue(cmdQueue) ;
  clReleaseMemObject(bufInputImage) ;
  clReleaseMemObject(bufOutputHistogram ) ;
  clReleaseContext(context);

  free (hInputImage) ;
  free (hOutputHistogram) ;
  free (programSource) ;

  return 0;
}

