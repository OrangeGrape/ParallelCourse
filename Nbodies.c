#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct{
  float x;
  float y;
  float z;
} vector;

typedef struct{
	float m;
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;
	float ax;
	float ay;
	float az;
} object;

int main(){
	//Model parameter
	float G = 6.67408*(10^(-11));//unit m/(kg*s^2)

	//for test
	//Earth mass
	float nG = 6.67408*(864^2)*(10^7);//unit 10^5 km/(10^22 kg * day^2)
	vector tmp;
	int maxiter = 1000,i, j, N = 3;
	float t = 0, tmax = 365,r;
	float dt = tmax-t/maxiter;
	object objlist[N];
//////////////////////////////////////////////
	//Sun
	objlist[0].m = 198910000;
	objlist[0].x = 0;
	objlist[0].y = 0;
	objlist[0].z = 0;
	objlist[0].vx = 0;
	objlist[0].vy = 0;
	objlist[0].vz = 0;
	objlist[0].ax = 0;
	objlist[0].ay = 0;
	objlist[0].az = 0;
	//Earth
	objlist[0].m = 597;
	objlist[0].x = 1495.98;
	objlist[0].y = 0;
	objlist[0].z = 0;
	objlist[0].vx = 0;
	objlist[0].vy = 1445.98/365.25;
	objlist[0].vz = 0;
	objlist[0].ax = 0;
	objlist[0].ay = 0;
	objlist[0].az = 0;
	//Moon
	objlist[0].m = 7;
	objlist[0].x = 1495.98 + 3.84;
	objlist[0].y = 0;
	objlist[0].z = 0;
	objlist[0].vx = 0;
	objlist[0].vy = 1445.98/365.25 + 3.84/30.4;
	objlist[0].vz = 0;
	objlist[0].ax = 0;
	objlist[0].ay = 0;
	objlist[0].az = 0;
//////////////////////////////////////////////
	for(t=0;t<tmax;t+=dt){
		for(i=0;i<N;i++){
			tmp.x = 0;
			tmp.y = 0;
			tmp.z = 0;
			for(j=0,j<N,j++){
				if(i==j){
					continue();
				}else{
					r = sqrt((objlist[j].x-objlist[i].x)^2 +(objlist[j].y-objlist[i].y)^2 +(objlist[j].z-objlist[i].z)^2);
					tmp.x += nG*objlist[j].m*(objlist[j].x-objlist[i].x)/r^3;
					tmp.y += nG*objlist[j].m*(objlist[j].y-objlist[i].y)/r^3;
					tmp.z += nG*objlist[j].m*(objlist[j].z-objlist[i].z)/r^3;
				}
			}
			//update accerelation
			objlist[i].ax = tmp.x;
			objlist[i].ay = tmp.y;
			objlist[i].az = tmp.z;
			//update position
			objlist[i].x += objlist[i].vx*dt;
			objlist[i].y += objlist[i].vy*dt;
			objlist[i].z += objlist[i].vz*dt;
			//update velocity
			objlist[i].vx += objlist[i].ax*dt;
			objlist[i].vy += objlist[i].ay*dt;
			objlist[i].vz += objlist[i].az*dt;
		}
	}
	return 0;
}
