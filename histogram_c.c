#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef unsigned char uint8;

int main() {

  //intial input properties
  int w = 1000;
  int h = 700;
  size_t size = w*h*sizeof(uint8);

  //read input
  uint8 *img = (uint8*)malloc(size);
  FILE *file = fopen("cat.dat","r");
  fread(img, sizeof(uint8), w*h, file);
  fclose(file);
  
  //preallocate output result
  int *xbin = (int*)malloc(256*sizeof(int));
  int *ycount = (int*)malloc(256*sizeof(int));

  //initial output result
  for(int i=0;i<256;i++){
    xbin[i]=i;
    ycount[i]=0;
  }

//start time record
clock_t start,stop;
start =clock();  

  //create histogram
  for(int i=0;i<w*h;i++){
    ycount[img[i]]++;
  }
//stop time record
stop = clock();
double cpu_time = (double) (stop-start) / CLOCKS_PER_SEC;
printf("CPU time = %lf\n", cpu_time);
  //print out the result
  /*
  for(int i=0;i<256;i++){
    printf("%d\t%d\n", xbin[i], ycount[i]);
  }*/

  //save output
  FILE *output = fopen("histogram.dat","w");
  
  for(int i=0;i<256;i++){
    fprintf(output,"%d\t%d\n",xbin[i],ycount[i]);
  }
  
  fclose(output); 
  
  return 0;
}
