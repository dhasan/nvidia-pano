#include 	<iostream>
#include <stdio.h>

unsigned int data[1200][1200];
int main(){


	FILE *ptr;
	int x,y;

	ptr = fopen("front.rgb","wb+");
	for (y=0;y<1200;y++){
		for(x=0;x<1200;x++){
			if (((x%120)<8) || ((y%120)<8)){
				data[y][x] = 0x00000080;
			}else{
				data[y][x] = 0xFFFFFF80;
			}
		}
	}

	fwrite(data, 1200*1200, 4, ptr);
	fclose(ptr);
	return 0;

}