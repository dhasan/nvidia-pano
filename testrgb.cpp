#include 	<iostream>
#include <stdio.h>

unsigned int data[1200][1200];
int main(){


	FILE *ptr;
	int x,y;unsigned int ci;
	unsigned int c;
	ptr = fopen("bottom.rgb","wb+");
	for (y=0;y<1200;y++){
		for(x=0;x<1200;x++){
			if (((x%120)<5) || ((y%120)<5)){
				data[y][x] = 0x00330080;
				ci =y*255/1200;
				
		//		printf("c: %d\n",c);
				data[y][x] |= ci << 16 | ci<<8;//(255*1200/x)<<24;
			}else{
				data[y][x] = 0xFFFFFF80;
			}
		}
	}

	fwrite(data, 1200*1200, 4, ptr);
	fclose(ptr);
	return 0;

}