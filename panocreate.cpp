#include <iostream>
#include <armadillo>

using namespace arma;

#define 	OUT_X				3600
#define 	OUT_Y				2400


#define 	SOURCE_COUNT		4

#define		SOURCE_0_ID			0
#define 	SOURCE_0_NAME		"front"
#define 	SOURCE_0_DEVNAME	"/dev/video0"
#define		SOURCE_0_X			1920
#define 	SOURCE_0_Y 			1080
#define		SOURCE_0_TRANSP_X	0.5
#define		SOURCE_0_TRANSP_Y	-0.5
#define 	SOURCE_0_TRANSP_Z	-0.5
#define 	SOURCE_0_ROTATE_X	datum::pi
#define 	SOURCE_0_ROTATE_Y	0
#define 	SOURCE_0_ROTATE_Z	0

#define		SOURCE_1_ID			1
#define 	SOURCE_1_NAME		"left"
#define 	SOURCE_1_DEVNAME	"/dev/video1"
#define		SOURCE_1_X			1920
#define 	SOURCE_1_Y 			1080
#define		SOURCE_1_TRANSP_X	0.5
#define		SOURCE_1_TRANSP_Y	-0.5
#define 	SOURCE_1_TRANSP_Z	0.5
#define 	SOURCE_1_ROTATE_X	datum::pi
#define 	SOURCE_1_ROTATE_Y	datum::pi/2
#define 	SOURCE_1_ROTATE_Z	0

#define		SOURCE_2_ID			2
#define 	SOURCE_2_NAME		"back"
#define 	SOURCE_2_DEVNAME	"/dev/video2"
#define		SOURCE_2_X			1920
#define 	SOURCE_2_Y 			1080
#define		SOURCE_2_TRANSP_X	-0.5
#define		SOURCE_2_TRANSP_Y	-0.5
#define 	SOURCE_2_TRANSP_Z	0.5
#define 	SOURCE_2_ROTATE_X	datum::pi
#define 	SOURCE_2_ROTATE_Y	0
#define 	SOURCE_2_ROTATE_Z	0

#define		SOURCE_3_ID			3
#define 	SOURCE_3_NAME		"right"
#define 	SOURCE_3_DEVNAME	"/dev/video1"
#define		SOURCE_3_X			1920
#define 	SOURCE_3_Y 			1080
#define		SOURCE_3_TRANSP_X	-0.5
#define		SOURCE_3_TRANSP_Y	-0.5
#define 	SOURCE_3_TRANSP_Z	-0.5
#define 	SOURCE_3_ROTATE_X	datum::pi
#define 	SOURCE_3_ROTATE_Y	-datum::pi/2
#define 	SOURCE_3_ROTATE_Z	0


#define		PLANE_COUNT			4

#define		PLANE_0_ID			0
#define		PLANE_0_SOURCEID	0
#define 	PLANE_0_DOT_0_X		-0.5
#define 	PLANE_0_DOT_0_Y		0.5
#define 	PLANE_0_DOT_0_Z		0.5
#define 	PLANE_0_DOT_1_X		0.5
#define 	PLANE_0_DOT_1_Y		0.5
#define 	PLANE_0_DOT_1_Z		0.5
#define 	PLANE_0_DOT_2_X		0.5
#define 	PLANE_0_DOT_2_Y		-0.5
#define 	PLANE_0_DOT_2_Z		0.5
#define 	PLANE_0_SCALE_X		1000
#define 	PLANE_0_SCALE_Y		1000
#define 	PLANE_0_OFFSET_X	100
#define 	PLANE_0_OFFSET_Y	100

#define		PLANE_1_ID			1
#define		PLANE_1_SOURCEID	0
#define 	PLANE_1_DOT_0_X		-0.5
#define 	PLANE_1_DOT_0_Y		0.5
#define 	PLANE_1_DOT_0_Z		0.5
#define 	PLANE_1_DOT_1_X		-0.5
#define 	PLANE_1_DOT_1_Y		-0.5
#define 	PLANE_1_DOT_1_Z		0.5
#define 	PLANE_1_DOT_2_X		0.5
#define 	PLANE_1_DOT_2_Y		-0.5
#define 	PLANE_1_DOT_2_Z		0.5
#define 	PLANE_1_SCALE_X		1000
#define 	PLANE_1_SCALE_Y		1000
#define 	PLANE_1_OFFSET_X	100
#define 	PLANE_1_OFFSET_Y	100

#define		PLANE_2_ID			2
#define		PLANE_2_SOURCEID	1
#define 	PLANE_2_DOT_0_X		-0.5
#define 	PLANE_2_DOT_0_Y		0.5
#define 	PLANE_2_DOT_0_Z		-0.5
#define 	PLANE_2_DOT_1_X		-0.5
#define 	PLANE_2_DOT_1_Y		0.5
#define 	PLANE_2_DOT_1_Z		0.5
#define 	PLANE_2_DOT_2_X		-0.5
#define 	PLANE_2_DOT_2_Y		-0.5
#define 	PLANE_2_DOT_2_Z		0.5
#define 	PLANE_2_SCALE_X		1000
#define 	PLANE_2_SCALE_Y		1000
#define 	PLANE_2_OFFSET_X	100
#define 	PLANE_2_OFFSET_Y	100

#define		PLANE_3_ID			3
#define		PLANE_3_SOURCEID	1
#define 	PLANE_3_DOT_0_X		-0.5
#define 	PLANE_3_DOT_0_Y		0.5
#define 	PLANE_3_DOT_0_Z		-0.5
#define 	PLANE_3_DOT_1_X		-0.5
#define 	PLANE_3_DOT_1_Y		-0.5
#define 	PLANE_3_DOT_1_Z		-0.5
#define 	PLANE_3_DOT_2_X		-0.5
#define 	PLANE_3_DOT_2_Y		-0.5
#define 	PLANE_3_DOT_2_Z		0.5
#define 	PLANE_3_SCALE_X		1000
#define 	PLANE_3_SCALE_Y		1000
#define 	PLANE_3_OFFSET_X	100
#define 	PLANE_3_OFFSET_Y	100


int source_ids[SOURCE_COUNT]={
	SOURCE_0_ID,
	SOURCE_1_ID,
	SOURCE_2_ID,
	SOURCE_3_ID
};

const char* source_names[SOURCE_COUNT]={
	SOURCE_0_NAME,
	SOURCE_1_NAME,
	SOURCE_2_NAME,
	SOURCE_3_NAME
};

const char* source_devnames[SOURCE_COUNT]={
	SOURCE_0_DEVNAME,
	SOURCE_1_DEVNAME,
	SOURCE_2_DEVNAME,
	SOURCE_3_DEVNAME
};

unsigned int source_x[SOURCE_COUNT]={
	SOURCE_0_X,
	SOURCE_1_X,
	SOURCE_2_X,
	SOURCE_3_X
};

unsigned int source_y[SOURCE_COUNT]={
	SOURCE_0_Y,
	SOURCE_1_Y,
	SOURCE_2_Y,
	SOURCE_3_Y
};

vec source_transp[SOURCE_COUNT]={
	colvec({SOURCE_0_TRANSP_X, SOURCE_0_TRANSP_Y, SOURCE_0_TRANSP_Z}),
	colvec({SOURCE_1_TRANSP_X, SOURCE_1_TRANSP_Y, SOURCE_1_TRANSP_Z}),
	colvec({SOURCE_2_TRANSP_X, SOURCE_2_TRANSP_Y, SOURCE_2_TRANSP_Z}),
	colvec({SOURCE_3_TRANSP_X, SOURCE_3_TRANSP_Y, SOURCE_3_TRANSP_Z})
};

vec source_rotate[SOURCE_COUNT]={
	colvec({SOURCE_0_ROTATE_X, SOURCE_0_ROTATE_Y, SOURCE_0_ROTATE_Z}),
	colvec({SOURCE_1_ROTATE_X, SOURCE_1_ROTATE_Y, SOURCE_1_ROTATE_Z}),
	colvec({SOURCE_2_ROTATE_X, SOURCE_2_ROTATE_Y, SOURCE_2_ROTATE_Z}),
	colvec({SOURCE_3_ROTATE_X, SOURCE_3_ROTATE_Y, SOURCE_3_ROTATE_Z})
};

/*Planes*/
int plane_ids[PLANE_COUNT]={
	PLANE_0_ID,
	PLANE_1_ID,
	PLANE_2_ID,
	PLANE_3_ID
};

int plane_source_id[PLANE_COUNT]={
	PLANE_0_SOURCEID,
	PLANE_1_SOURCEID,
	PLANE_2_SOURCEID,
	PLANE_3_SOURCEID
};

vec plane_dots[PLANE_COUNT][3]={
	{
		{colvec({PLANE_0_DOT_0_X, PLANE_0_DOT_0_Y, PLANE_0_DOT_0_Z})},
		{colvec({PLANE_0_DOT_1_X, PLANE_0_DOT_1_Y, PLANE_0_DOT_1_Z})},
		{colvec({PLANE_0_DOT_2_X, PLANE_0_DOT_2_Y, PLANE_0_DOT_2_Z})},
	},
	{
		{colvec({PLANE_1_DOT_0_X, PLANE_1_DOT_0_Y, PLANE_1_DOT_0_Z})},
		{colvec({PLANE_1_DOT_1_X, PLANE_1_DOT_1_Y, PLANE_1_DOT_1_Z})},
		{colvec({PLANE_1_DOT_2_X, PLANE_1_DOT_2_Y, PLANE_1_DOT_2_Z})},
	},
	{
		{colvec({PLANE_2_DOT_0_X, PLANE_2_DOT_0_Y, PLANE_2_DOT_0_Z})},
		{colvec({PLANE_2_DOT_1_X, PLANE_2_DOT_1_Y, PLANE_2_DOT_1_Z})},
		{colvec({PLANE_2_DOT_2_X, PLANE_2_DOT_2_Y, PLANE_2_DOT_2_Z})},
	},
	{
		{colvec({PLANE_3_DOT_0_X, PLANE_3_DOT_0_Y, PLANE_3_DOT_0_Z})},
		{colvec({PLANE_3_DOT_1_X, PLANE_3_DOT_1_Y, PLANE_3_DOT_1_Z})},
		{colvec({PLANE_3_DOT_2_X, PLANE_3_DOT_2_Y, PLANE_3_DOT_2_Z})},
	}
};

vec plane_scale[PLANE_COUNT]={
	colvec({PLANE_0_SCALE_X,PLANE_0_SCALE_Y}),
	colvec({PLANE_1_SCALE_X,PLANE_1_SCALE_Y}),
	colvec({PLANE_2_SCALE_X,PLANE_2_SCALE_Y}),
	colvec({PLANE_3_SCALE_X,PLANE_3_SCALE_Y})
};

vec plane_offset[PLANE_COUNT]={
	colvec({PLANE_0_OFFSET_X,PLANE_0_OFFSET_Y}),
	colvec({PLANE_1_OFFSET_X,PLANE_1_OFFSET_Y}),
	colvec({PLANE_2_OFFSET_X,PLANE_2_OFFSET_Y}),
	colvec({PLANE_3_OFFSET_X,PLANE_3_OFFSET_Y})
};


#define 	GET_SOURCE(id,prop)	SOURCE_##id##_##prop
#define 	GET_PLANE(id,prop)	PLANE_##id##_##prop
#define		GET_PLANE_DOTS(id,dotid,prop)	PLANE_##id##_DOT_##dotid##_##prop

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
	j=0;i=0;
	// for (j=0;j<OUT_Y;j++){
	// 	phi = (j * (datum::pi)/OUT_Y) - datum::pi/2;
	// 	for(i=0;i<OUT_X;i++){
	// 		theta = (i * ((2*datum::pi)/OUT_X)) - datum::pi;


	for (j=0,phi=-datum::pi/2;phi<datum::pi/2;phi+=((datum::pi)/OUT_Y),j++){
		///i=0;
		for(i=0,theta=0.00000001;theta<2*datum::pi;theta+=((2*datum::pi)/OUT_X),i++){
			
			x = 2*cos(phi)*cos(theta);
			y = 2*sin(phi); 
			z = 2*cos(phi)*sin(theta);
			
			for (p=0;p<PLANE_COUNT;p++){
				if (1){
				//if (p!=fp){
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
						// if (((i<0) || (i>OUT_X)) || (j>OUT_Y) || (j<0))
						// 	printf("%d %d\n",i,j );
						//printf("%d\n",p );
						//printf("%d %d %d\n",p,i,j);

						if ((i>=0) && (i<OUT_X) && (j>=0) && (j<OUT_Y)){
							dotarray[j][i].sourceid = p;
							if (p==0)
								test[j][i] = 0xFF000080;
							else if (p==1)
								test[j][i] = 0x00FF0080;
							else if (p==2)
								test[j][i] = 0x0000FF80;
							else 
								test[j][i] = 0xFFFF0080;
							dotarray[j][i].x = crd(0);
							dotarray[j][i].y = crd(1);
						}	

					}
				
				}
				 
			}
			//i++;
		}
		//printf("phi: %f\n", phi);
		//j++;
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





















