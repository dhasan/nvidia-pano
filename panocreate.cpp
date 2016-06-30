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


void defsource(int id, struct source *source){

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

bool sameside(vec *p1, vec *p2, vec *a, vec *b){
	vec cp1,cp2;
	cp1 = cross(*b-*a, *p1-*a);
	cp2 = cross(*b-*a, *p2-*a);
	if (dot(cp1, cp2)>=0) return true;
	else return false;
}

bool pointintriangle(vec *p, vec *a, vec *b, vec *c){
	if (sameside(p, a, b, c) && sameside(p,b, a,c) && sameside(p,c, a,b)){
		return true;
	}else
		return false;
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
	ptr = fopen("out.raw", "wb+");

	for (j=0,phi=-datum::pi/2;phi<datum::pi/2;phi+=((datum::pi)/OUT_Y),j++){
		for(i=0,theta=0.00000001;theta<2*datum::pi;theta+=((2*datum::pi)/OUT_X),i++){
			
			x = 2*cos(phi)*cos(theta);
			y = 2*sin(phi); 
			z = 2*cos(phi)*sin(theta);

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
					//m =  planes[p]->source->world_matrix * planes[p]->source_matrix;
					
					crd = m * in;
					
					if ((i>=0) && (i<OUT_X) && (j>=0) && (j<OUT_Y)){
						dotarray[j][i].sourceid = p;
						if ((fp==0) || (fp==1))
							test[j][i] = 0xFF000080;
						else if ((fp==2) || (fp==3))
							test[j][i] = 0x00FF0080;
						else if ((fp==4) || (fp==5))
							test[j][i] = 0x0000FF80;
						else if ((fp==6) || (fp==7))
							test[j][i] = 0xFFFF0080;
						else if ((fp==8) || (fp==9))
							test[j][i] = 0xFF00FF80;
						else
							test[j][i] = 0x00FFFF80;
						dotarray[j][i].x = crd(0);
						dotarray[j][i].y = crd(1);
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
					m =  planes[p]->source->world_matrix * planes[p]->source_matrix;
					
					crd = m * in;
					
					if ((i>=0) && (i<OUT_X) && (j>=0) && (j<OUT_Y)){
						dotarray[j][i].sourceid = p;
						if ((p==0) || (p==1))
							test[j][i] = 0xFF000080;
						else if ((p==2) || (p==3))
							test[j][i] = 0x00FF0080;
						else if ((p==4) || (p==5))
							test[j][i] = 0x0000FF80;
						else if ((p==6) || (p==7))
							test[j][i] = 0xFFFF0080;
						else if ((p==8) || (p==9))
							test[j][i] = 0xFF00FF80;
						else
							test[j][i] = 0x00FFFF80;
						dotarray[j][i].x = crd(0);
						dotarray[j][i].y = crd(1);
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
	//fflush(ptr);
	//fclose(ptr);
	


#if 1
	mat all;
	all = mat(4,4);
	all = planes[0]->source_matrix * source[0]->world_matrix;
	//cout<<"all"<<endl;
	//all.print();

	vec in2 = vec(4);

	in2  << 0 << endr 
		<< 0 << endr
		<< 1 << endr
		<< 1 << endr;

	vec out = vec(4);

	out = all*in2;
	//out.print();
#endif 

	return 0;
}





















