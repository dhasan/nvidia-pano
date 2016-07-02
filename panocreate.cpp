#include 	"private.h"

using namespace arma;

#define 	GET_SOURCE(id,prop)	SOURCE_##id##_##prop
#define 	GET_PLANE(id,prop)	PLANE_##id##_##prop
#define		GET_PLANE_DOTS(id,dotid,prop)	PLANE_##id##_DOT_##dotid##_##prop

struct int4 {
	int x;
	int y;
	int z;
	int w;
};

struct float4 {
	float x;
	float y;
	float z;
	float w;
};
struct source {

	int 	id;
	char 	*name;
	char 	*devname;
	int 	fd;

	/*in*/
	int		x;
	int 	y;

	/*transpose origin to top left corner*/
	mat 	transp;

	/*rotate origin, so x - from left to right, y - top to bottom*/
	vec		rotate; //3 radians

	/*out*/
	vec 	normal;
	mat 	world_matrix;
	unsigned int 	*data;
};

struct plane {
	
	int 	id;
	struct source *source;

	/*in*/
	vec		dots[3];

	mat 	scale;

	/*transpose source origin to plane top left*/
	mat		offset; //this should be applied after scale

	/*out*/
	vec 	normal;
	mat 	source_matrix;
};

struct outdot {
	int sourceid;
	double x;
	double y;
};


struct outdot dotarray[OUT_Y][OUT_X];
unsigned int test[OUT_Y][OUT_X];

struct int4 xymap[OUT_Y][OUT_X];
struct float4 bmap[OUT_Y][OUT_X];



inline double to_degrees(double radians) {
    return radians * (180.0 / arma::datum::pi);
}

inline double to_radians(double degree) {
    return radians * ( arma::datum::pi/180.0);
}
void defsource(int id, struct source *source){
	FILE *ptr;
	size_t size;
	if (source==NULL){
		cout<<"Error: Unable to allocate source class!"<<endl;
		return;
	}else
		cout<<"Source 0 class allocated."<<endl;

	source->id = source_ids[id];
	source->name = new char[10];
	strcpy(source->name, source_names[id]);

	source->x = source_x[id];
	source->y = source_y[id];

	source->transp = mat(4,4);
	source->world_matrix = mat(4,4);
	source->rotate = vec(3);

	source->world_matrix.eye();

	source->transp.eye();
	source->transp(0,3) = source_transp[id](0);
	source->transp(1,3) = source_transp[id](1);
	source->transp(2,3) = source_transp[id](2);

	source->world_matrix = source->transp * source->world_matrix;

	source->rotate(0) = source_rotate[id](0);  
	source->rotate(1) = source_rotate[id](1);
	source->rotate(2) = source_rotate[id](2);

	mat rx = mat(4,4);
	rx.eye();
	rx(1,1) = cos(source->rotate(0));
	rx(1,2) = -sin(source->rotate(0));
	rx(2,1) = sin(source->rotate(0));
	rx(2,2) = cos(source->rotate(0));


	source->world_matrix = rx * source->world_matrix;

	mat ry = mat(4,4);
	ry.eye();
	ry(0,0) = cos(source->rotate(1));
	ry(0,2) = -sin(source->rotate(1));
	ry(2,0) = sin(source->rotate(1));
	ry(2,2) = cos(source->rotate(1));
	source->world_matrix = ry * source->world_matrix;

	mat rz = mat(4,4);
	rz.eye();
	rz(0,0) = cos(source->rotate(2));
	rz(0,1) = -sin(source->rotate(2));
	rz(1,0) = sin(source->rotate(2));
	rz(1,1) = cos(source->rotate(2));
	source->world_matrix = rz * source->world_matrix;

	ptr = fopen("front.rgb", "r");
	// fseek(ptr, 0, SEEK_END); // seek to end of file
	// size = ftell(ptr); // get current file pointer
	// fseek(ptr, 0, SEEK_SET); // seek back to beginning of file

	source->data = (unsigned int *)malloc(source->x*source->y*4);
	fread(source->data, source->x*source->y, sizeof(unsigned int), ptr);
	fclose(ptr);	
}

void defplane(int id, struct plane *plane, struct source **src){

	int s;

	plane->id = plane_ids[id];
	plane->source_matrix = mat(4,4);
	plane->source_matrix.eye();

	plane->dots[0] = plane_dots[id][0];
	plane->dots[1] = plane_dots[id][1];
	plane->dots[2] = plane_dots[id][2];
	plane->normal =  vec(3);

	vec a,b,c;

	a = plane->dots[1] - plane->dots[0];
	b = plane->dots[2] - plane->dots[0];

	plane->normal = cross(a,b);

	plane->scale = mat(4,4);
	plane->scale.eye();
	plane->scale(0,0) = plane_scale[id](0);
	plane->scale(1,1) = plane_scale[id](1);
	
	plane->source_matrix = plane->scale;

	plane->offset = mat(4,4);
	plane->offset.eye();
	plane->offset(0,3) = plane_offset[id](0);
	plane->offset(1,3) = plane_offset[id](1);

	plane->source_matrix = plane->offset * plane->scale;

	for (s=0;s<SOURCE_COUNT;s++){
		if (plane_source_id[id]==src[s]->id){
			plane->source = src[s];
			break;
		}
	}
}

unsigned int inter_sum(struct float4 *vec, unsigned int q1, unsigned int q2, unsigned int q3, unsigned int q4 ){
	unsigned int temp;

	unsigned char r1,g1,b1,a1;
	unsigned char r2,g2,b2,a2;
	unsigned char r3,g3,b3,a3;
	unsigned char r4,g4,b4,a4;
	float r,g,b,a;
	r1 = q1>>24;
	r2 = q2>>24;
	r3 = q3>>24;
	r4 = q4>>24;

	g1 = (q1 & 0x00FF0000)>>16;
	g2 = (q2 & 0x00FF0000)>>16;
	g3 = (q3 & 0x00FF0000)>>16;
	g4 = (q4 & 0x00FF0000)>>16;

	b1 = (q1 & 0x0000FF00)>>8;
	b2 = (q2 & 0x0000FF00)>>8;
	b3 = (q3 & 0x0000FF00)>>8;
	b4 = (q4 & 0x0000FF00)>>8;

	r = vec->x*(float)r1 + vec->y*(float)r2 + vec->z*(float)r3 + vec->w*(float)r4;
	g = vec->x*(float)g1 + vec->y*(float)g2 + vec->z*(float)g3 + vec->w*(float)g4;
	b = vec->x*(float)b1 + vec->y*(float)b2 + vec->z*(float)b3 + vec->w*(float)b4;
//	printf("r: %f g: %f b: %f\n",r,g,b );

	//temp = b(0)*q1 + b()
	temp =0x80;
	unsigned int rr = (unsigned int)r;
	unsigned int gg = (unsigned int)g;
	unsigned int bb = (unsigned int)b;

	temp |= (rr<<24 | (gg<<16) | bb<<8);
	//printf("%x\n",temp );
	return temp;
}

int main(){
	int s,p,f,i,j,fp;
	double phi,theta;
	mat t = mat(3,3);
	vec v = vec(3);
	vec o = vec(3);
	vec in = vec(4);
	mat m = mat(4,4);
	vec crd;
	vec b= vec(4);
	mat bicubic = mat(4,4);
	vec vals = vec(4);

	FILE *ptr;
	
	double x,y,z;
	struct source *source[SOURCE_COUNT];

	for (s=0;s<SOURCE_COUNT;s++){
		source[s] = new struct source;
		defsource(s, source[s]);
	}

	struct plane *planes[PLANE_COUNT];
	

	for(p=0;p<PLANE_COUNT;p++){
		planes[p] = new struct plane;
		defplane(p, planes[p], source);
	}
	fp = -1;

	for (j=0;j<OUT_X;j++){
		for (i=0;i<OUT_Y;i++){
			
			test[i][j] = 0x00000000;
		}
	}
	#if 1
	ptr = fopen("out2.raw", "wb+");

	for (j=0,phi=-datum::pi/2;phi<datum::pi/2;phi+=((datum::pi)/OUT_Y),j++){
		for(i=0,theta=-datum::pi;theta<datum::pi;theta+=((2*datum::pi)/OUT_X),i++){
			
			x = cos(phi)*cos(theta);
			y = sin(phi); 
			z = cos(phi)*sin(theta);

			if (fp!=-1){
				//TODO: set only xyz deps
				t 	<< -x << planes[fp]->dots[1](0) - planes[fp]->dots[0](0) << planes[fp]->dots[2](0) - planes[fp]->dots[0](0) << endr
					<< -y << planes[fp]->dots[1](1) - planes[fp]->dots[0](1) << planes[fp]->dots[2](1) - planes[fp]->dots[0](1) << endr
					<< -z << planes[fp]->dots[1](2) - planes[fp]->dots[0](2) << planes[fp]->dots[2](2) - planes[fp]->dots[0](2) << endr;
				v 	<< -planes[fp]->dots[0](0) << endr
					<< -planes[fp]->dots[0](1) << endr
					<< -planes[fp]->dots[0](2) << endr;
			
				o = inv(t) * v;
				if ((o(1)>=0 && o(1)<=1) && (o(2)>=0 && o(2)<=1) && ((o(1) + o(2))<=1) && (o(0)>=0) && (o(0)<=1)){
					in 	<< x*o(0) << endr
						<< y*o(0) << endr
						<< z*o(0) << endr
						<< 1 << endr;
					if (planes[p]->source==NULL){
						printf("error!\n");
						return -1;
					}
					m =  planes[fp]->source_matrix * planes[fp]->source->world_matrix;
					
					crd = m * in;
					
					if ((i>=0) && (i<OUT_X) && (j>=0) && (j<OUT_Y)){
						dotarray[j][i].sourceid = p;

						// int ry = round(crd(1));
						// int rx = round(crd(0));
						// test[j][i]=*(planes[fp]->source->data+ry*1200 + rx);
						int intpart;
						
						dotarray[j][i].x = crd(0);
						dotarray[j][i].y = crd(1);
						if ((unsigned int)floor(crd(0))!=(unsigned int)ceil(crd(0)) && (unsigned int)floor(crd(1))!=(unsigned int)ceil(crd(1))){
							bicubic << 1 << floor(crd(0)) << floor(crd(1)) << floor(crd(0)) * floor(crd(1)) << endr
									<< 1 << floor(crd(0)) << ceil(crd(1)) << floor(crd(0)) * ceil(crd(1)) << endr
									<< 1 << ceil(crd(0)) << floor(crd(1)) << ceil(crd(0)) * floor(crd(1)) << endr
									<< 1 << ceil(crd(0)) << ceil(crd(1)) << ceil(crd(0)) * ceil(crd(1)) << endr;
							vals 	<< 1 << endr
									<< crd(0) << endr
									<< crd(1) << endr
									<< crd(0)*crd(1) << endr;
							
							b = trans(inv(bicubic)) * vals;// * vals;// * vals;		
							// b.print();
							// printf("\n");
							
							bmap[j][i].x = b(0);
							bmap[j][i].y = b(1);
							bmap[j][i].z = b(2);
							bmap[j][i].w = b(3);
						}else{
							bmap[j][i].x = 1.0f;
							bmap[j][i].y = 0;
							bmap[j][i].z = 0;
							bmap[j][i].w = 0;
						}
						int x1 = (int)floor(crd(0));
						int y1 = (int)floor(crd(1));
						int x2 = (int)ceil(crd(0));
						int y2 = ceil(crd(1));

						if (x1<0){
							printf("Warning x1: %d at j:%d i:%d for plane %d\n",x1,j,i,fp);
							x1=0;
						}
						if (x2<0){
							printf("Warning x2: %d at j:%d i:%d for plane %d\n",x2,j,i,fp);
							x2=0;
						}
						if (y1<0){
							printf("Warning y1: %d at j:%d i:%d for plane %d\n",y1,j,i,fp);
							y1=0;
						}
						if (y2<0){
							printf("Warning y2: %d at j:%d i:%d for plane %d\n",y2,j,i,fp);
							y2=0;
						}
						
						xymap[j][i].x = (fp<<16) | x1; //x1
						xymap[j][i].y = x2; //y1
						xymap[j][i].z = y1; //x2
						xymap[j][i].w = y2; //y2

						test[j][i] = inter_sum(&bmap[j][i]	, *(planes[fp]->source->data+(unsigned int)floor(crd(1))*1200 + (unsigned int)floor(crd(0)))
															, *(planes[fp]->source->data+(unsigned int)floor(crd(1))*1200 + (unsigned int)ceil(crd(0)))
															, *(planes[fp]->source->data+(unsigned int)ceil(crd(0))*1200 + (unsigned int)floor(crd(1)))
															, *(planes[fp]->source->data+(unsigned int)ceil(crd(0))*1200 + (unsigned int)ceil(crd(1))));
					}else{
						//printf("warning x: %d y: %d\n", i,j);
					}
					continue;
				}
			}
			
			for (p=0;p<PLANE_COUNT;p++){
				t 	<< -x << planes[p]->dots[1](0) - planes[p]->dots[0](0) << planes[p]->dots[2](0) - planes[p]->dots[0](0) << endr
					<< -y << planes[p]->dots[1](1) - planes[p]->dots[0](1) << planes[p]->dots[2](1) - planes[p]->dots[0](1) << endr
					<< -z << planes[p]->dots[1](2) - planes[p]->dots[0](2) << planes[p]->dots[2](2) - planes[p]->dots[0](2) << endr;
				v 	<< -planes[p]->dots[0](0) << endr
					<< -planes[p]->dots[0](1) << endr
					<< -planes[p]->dots[0](2) << endr;
			
				o = inv(t) * v;
				if ((o(1)>=0 && o(1)<=1) && (o(2)>=0 && o(2)<=1) && ((o(1) + o(2))<=1) && (o(0)>=0) && (o(0)<=1)){
					in 	<< x*o(0) << endr
						<< y*o(0) << endr
						<< z*o(0) << endr
						<< 1 << endr;
					if (planes[p]->source==NULL){
						printf("error!\n");
						return -1;
					}
					m =  planes[p]->source_matrix * planes[p]->source->world_matrix;
					
					crd = m * in;
					
					if ((i>=0) && (i<OUT_X) && (j>=0) && (j<OUT_Y)){

						// printf("%d in: %f %f  %f out: %f %f\n", p, in(0),in(1),in(2), crd(0), crd(1));
				//		test[j][i]=*(planes[p]->source->data + (int(crd(1))*1200) + int(crd(0)));

						dotarray[j][i].sourceid = p;
						dotarray[j][i].x = crd(0);
						dotarray[j][i].y = crd(1);
						if ((unsigned int)floor(crd(0))!=(unsigned int)ceil(crd(0)) && (unsigned int)floor(crd(1))!=(unsigned int)ceil(crd(1))){
							bicubic << 1 << floor(crd(0)) << floor(crd(1)) << floor(crd(0)) * floor(crd(1)) << endr
									<< 1 << floor(crd(0)) << ceil(crd(1)) << floor(crd(0)) * ceil(crd(1)) << endr
									<< 1 << ceil(crd(0)) << floor(crd(1)) << ceil(crd(0)) * floor(crd(1)) << endr
									<< 1 << ceil(crd(0)) << ceil(crd(1)) << ceil(crd(0)) * ceil(crd(1)) << endr;
							vals 	<< 1 << endr
									<< crd(0) << endr
									<< crd(1) << endr
									<< crd(0)*crd(1) << endr;
							
							b = trans(inv(bicubic)) * vals;// * vals;// * vals;		
							// b.print();
							// printf("\n");
							
							bmap[j][i].x = b(0);
							bmap[j][i].y = b(1);
							bmap[j][i].z = b(2);
							bmap[j][i].w = b(3);
						}else{
							bmap[j][i].x = 1.0f;
							bmap[j][i].y = 0;
							bmap[j][i].z = 0;
							bmap[j][i].w = 0;
						}
						int x1 = (int)floor(crd(0));
						int y1 = (int)floor(crd(1));
						int x2 = (int)ceil(crd(0));
						int y2 = ceil(crd(1));
						if (x1<0){
							printf("Warning x1: %d at j:%d i:%d for plane %d\n",x1,j,i,p);
							x1=0;
						}
						if (x2<0){
							printf("Warning x2: %d at j:%d i:%d for plane %d\n",x2,j,i,p);
							x2=0;
						}
						if (y1<0){
							printf("Warning y1: %d at j:%d i:%d for plane %d\n",y1,j,i,p);
							y1=0;
						}
						if (y2<0){
							printf("Warning y2: %d at j:%d i:%d for plane %d\n",y2,j,i,p);
							y2=0;
						}

						xymap[j][i].x = (p<<16) | x1; //x1
						xymap[j][i].y = y1; //y1
						xymap[j][i].z = x2; //x2
						xymap[j][i].w = y2; //y2

						test[j][i] = inter_sum(&bmap[j][i]	, *(planes[p]->source->data+(unsigned int)floor(crd(1))*1200 + (unsigned int)floor(crd(0)))
															, *(planes[p]->source->data+(unsigned int)floor(crd(1))*1200 + (unsigned int)ceil(crd(0)))
															, *(planes[p]->source->data+(unsigned int)ceil(crd(0))*1200 + (unsigned int)floor(crd(1)))
															, *(planes[p]->source->data+(unsigned int)ceil(crd(0))*1200 + (unsigned int)ceil(crd(1))));

					}	
					fp = p;
					break;
				}
			}
			
		}
		
		if (!(j%24)){
			printf("%d%%...\n", j/24);
		}
	}
	//LUT is done
	fwrite(test, OUT_X*OUT_Y,4, ptr);
	fflush(ptr);
	fclose(ptr);

	FILE *xyp, *bmapp;
	xyp = fopen("xymap.bin","wb+");
	bmapp = fopen("bmap.bin", "wb+");
	fwrite(xymap, OUT_Y*OUT_X, sizeof(struct int4), xyp);
	fwrite(bmap, OUT_Y*OUT_X, sizeof(struct float4), bmapp);
	fflush(xyp);
	fflush(bmapp);
	fclose(xyp);
	fclose(bmapp);
	
	//for y
	//x1 = -datum::pi/2
	//x2 = datum::pi/2
	//y1 = 0
	//y2 = OUT_Y


	//for x
	//x1 = -datum::pi
	//x2 = datum::pi
	//y1 = 0
	//y2 = OUT_X


#endif
#if 1
	mat all;
	all = mat(4,4);
	all = planes[0]->source_matrix * source[0]->world_matrix;
	//cout<<"all"<<endl;
	all.print();

	vec in2 = vec(4);

	in2  << -0.5 << endr 
		<< -0.5 << endr
		<< 1 << endr
		<< 1 << endr;

	vec out = vec(4);

	out = all*in2;
	out.print();
#endif 

	return 0;
}





















