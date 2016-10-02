#include 	<fcntl.h>
#include 	<stdio.h>
#define 	OUT_X				(3600)
#define 	OUT_Y				(1800)

#define 	SOURCE_X 			(1920)
#define 	SOURCE_Y 			(1080)

#define DEST_X 	(640)
#define DEST_Y	(480)

#define DEST_RATIO ((float)DEST_X)/((float)DEST_Y)

#define ANGLE_PHI	(0)
#define ANGLE_THETA	(0)

// #define FOV_X		(90)
// #define FOV_Y		(90)

#define RADIUS		(1)
//#define RADIUS		(OUT_X/(2*datum::pi))

//[1200][1200];

#define MAX(a,b) (a>b)?a:b
#define MIN(a,b) (a<b)?a:b

struct dev {
	float *matrix;
	uint4 *xymap;
	float4 *bmap;
	unsigned int *sdata[6];
	unsigned int *panodata;
};

struct pano {

	unsigned int pano_width;
	unsigned int pano_height;

	float theta;
	float phi;
	float fov;
	float matrix[9];

	unsigned int source_width;
	unsigned int source_height;

	FILE *sdatafd[6];
	unsigned int *sdata[6];

	FILE *xymapfd;
	uint4 *xymap;

	FILE *bmapfd;
	float4 *bmap;

	unsigned int dest_width;
	unsigned int dest_height;
	//float dest_ratio;

	unsigned int *panodata;

	struct dev dev;

	//////////////////
	cudaTextureObject_t xytext;// = 0;	
	cudaTextureObject_t btext;// = 0;	


};

#ifndef __CUDACC__ 
struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct float3 {
	float x;
	float y;
	float z;

};

struct int4 {
	 unsigned int x;
	 unsigned int y;
	 unsigned int z;
	 unsigned int w;
};
#else
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif

# define M_PI           3.14159265358979323846

#ifdef CUDA_PANO_LIB
struct pano gstpano;
cudaStream_t stream[8];
#endif

static void mul4x4x4(float *a, float *b, float *out){
	out[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8] + a[3]*b[12];
	out[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9] + a[3]*b[13];
	out[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14];
	out[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15];


	out[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8] + a[7]*b[12];
	out[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9] + a[7]*b[13];
	out[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14];
	out[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15];

	out[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8] + a[11]*b[12];
	out[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9] + a[11]*b[13];
	out[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10] + a[11]*b[14];
	out[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]*b[15];

	out[12] = a[12]*b[0] + a[13]*b[4] + a[14]*b[8] + a[15]*b[12];
	out[13] = a[12]*b[1] + a[13]*b[5] + a[14]*b[9] + a[15]*b[13];
	out[14] = a[12]*b[2] + a[13]*b[6] + a[14]*b[10] + a[15]*b[14];
	out[15] = a[12]*b[3] + a[13]*b[7] + a[14]*b[11] + a[15]*b[15];

}



__device__ void mul4x4x1(double *a, double *b, double *out){
	out[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
	out[1] = a[4]*b[0] + a[5]*b[1] + a[6]*b[2] + a[7]*b[3];
	out[2] = a[8]*b[0] + a[9]*b[1] + a[10]*b[2] + a[11]*b[3];
	out[3] = a[12]*b[0] + a[13]*b[1] + a[14]*b[2] + a[15]*b[3];
}



__device__ void trans4x4(double *data, double *out){
	out[0] = data[0];
	out[1] = data[4];
	out[2] = data[8];
	out[3] = data[12];

	out[4] = data[1];
	out[5] = data[5];
	out[6] = data[9];
	out[7] = data[13];

	out[8] = data[2];
	out[9] = data[6];
	out[10]= data[10];
	out[11] = data[14];

	out[12] = data[3];
	out[13] = data[7];
	out[14] = data[11];
	out[15] = data[15];

}

__device__
double det4x4(double *a){
	double det;

	det = a[0]*a[5]*a[10]*a[15] + a[0]*a[6]*a[11]*a[13] + a[0]*a[7]*a[9]*a[14]
		   +a[1]*a[4]*a[11]*a[14] + a[1]*a[6]*a[8]*a[15] + a[1]*a[7]*a[10]*a[12]
		   +a[2]*a[4]*a[9]*a[15] + a[2]*a[5]*a[11]*a[12] + a[2]*a[7]*a[8]*a[13]
		   +a[3]*a[4]*a[10]*a[13] + a[3]*a[5]*a[8]*a[14] + a[3]*a[6]*a[9]*a[12]
		   -a[0]*a[5]*a[11]*a[14] - a[0]*a[6]*a[9]*a[15] - a[0]*a[7]*a[10]*a[13]
		   -a[1]*a[4]*a[10]*a[15] - a[1]*a[6]*a[11]*a[12] - a[1]*a[7]*a[8]*a[14]
		   -a[2]*a[4]*a[11]*a[13] - a[2]*a[5]*a[8]*a[15] - a[2]*a[7]*a[9]*a[12]
		   -a[3]*a[4]*a[9]*a[14] - a[3]*a[5]*a[10]*a[12] - a[3]*a[6]*a[8]*a[13];
	return det;
}

__device__ bool inverse4x4(double *a, double *b){
	double det = det4x4(a);

	if (det==0.0f){
		return false;
	}

	b[0] = (a[5]*a[10]*a[15] + a[6]*a[11]*a[13] + a[7]*a[9]*a[14]  -
			a[5]*a[11]*a[14] - a[6]*a[9]*a[15]  - a[7]*a[10]*a[13])/det;
	b[1] = (a[1]*a[11]*a[14] + a[2]*a[9]*a[15]  + a[3]*a[10]*a[13] -
			a[1]*a[10]*a[15] - a[2]*a[11]*a[13] - a[3]*a[9]*a[14])/det;
	b[2] = (a[1]*a[6]*a[15]  + a[2]*a[7]*a[13]  + a[3]*a[5]*a[14]  -
			a[1]*a[7]*a[14]  - a[2]*a[5]*a[15]  - a[3]*a[6]*a[13])/det;
	b[3] = (a[1]*a[7]*a[10]  + a[2]*a[5]*a[11]  + a[3]*a[6]*a[9]   -
			a[1]*a[6]*a[11]  - a[2]*a[7]*a[9]   - a[3]*a[5]*a[10])/det;
	b[4] = (a[4]*a[11]*a[14] + a[6]*a[8]*a[15]  + a[7]*a[10]*a[12] -
			a[4]*a[10]*a[15] - a[6]*a[11]*a[12] - a[7]*a[8]*a[14])/det;
	b[5] = (a[0]*a[10]*a[15] + a[2]*a[11]*a[12] + a[3]*a[8]*a[14]  -
			a[0]*a[11]*a[14] - a[2]*a[8]*a[15]  - a[3]*a[10]*a[12])/det;
	b[6] = (a[0]*a[7]*a[14]  + a[2]*a[4]*a[15]  + a[3]*a[6]*a[12]  -
			a[0]*a[6]*a[15]  - a[2]*a[7]*a[12]  - a[3]*a[4]*a[14])/det;
	b[7] = (a[0]*a[6]*a[11]  + a[2]*a[7]*a[8]   + a[3]*a[4]*a[10]  -
			a[0]*a[7]*a[10]  - a[2]*a[4]*a[11]  - a[3]*a[6]*a[8])/det;
	b[8] = (a[4]*a[9]*a[15]  + a[5]*a[11]*a[12] + a[7]*a[8]*a[13] -
			a[4]*a[11]*a[13] - a[5]*a[8]*a[15]  - a[7]*a[9]*a[12])/det;
	b[9] = (a[0]*a[11]*a[13] + a[1]*a[8]*a[15]  + a[3]*a[9]*a[12] -
			a[0]*a[9]*a[15]  - a[1]*a[11]*a[12] - a[3]*a[8]*a[13])/det;
	b[10]= (a[0]*a[5]*a[15]  + a[1]*a[7]*a[12]  + a[3]*a[4]*a[13] -
			a[0]*a[7]*a[13]  - a[1]*a[4]*a[15]  - a[3]*a[5]*a[12])/det;
	b[11]= (a[0]*a[7]*a[9]   + a[1]*a[4]*a[11]  + a[3]*a[5]*a[8]  -
			a[0]*a[5]*a[11]  - a[1]*a[7]*a[8]   - a[3]*a[4]*a[9])/det;
	b[12]= (a[4]*a[10]*a[13] + a[5]*a[8]*a[14]  + a[6]*a[9]*a[12] -
			a[4]*a[9]*a[14]  - a[5]*a[10]*a[12] - a[6]*a[8]*a[13])/det;
	b[13]= (a[0]*a[9]*a[14]  + a[1]*a[10]*a[12] + a[2]*a[8]*a[13] -
			a[0]*a[10]*a[13] - a[1]*a[8]*a[14]  - a[2]*a[9]*a[12])/det;
	b[14]= (a[0]*a[6]*a[13]  + a[1]*a[4]*a[14]  + a[2]*a[5]*a[12] -
			a[0]*a[5]*a[14]  - a[1]*a[6]*a[12]  - a[2]*a[4]*a[13])/det;
	b[15]= (a[0]*a[5]*a[10]  + a[1]*a[6]*a[8]   + a[2]*a[4]*a[9]  -
			a[0]*a[6]*a[9]   - a[1]*a[4]*a[10]  - a[2]*a[5]*a[8])/det;
	return true;
}

static float det3x3(float *data){

	float p1 = *(data + 0*3 + 0) * *(data + 1*3 + 1) * *(data + 2*3 + 2);
	float p2 = *(data + 0*3 + 1) * *(data + 1*3 + 2) * *(data + 2*3 + 0);
	float p3 = *(data + 1*3 + 0) * *(data + 2*3 + 1) * *(data + 0*3 + 2);

	float n1 = *(data + 0*3 + 2) * *(data + 1*3 + 1) * *(data + 2*3 + 0);
	float n2 = *(data + 1*3 + 0) * *(data + 0*3 + 1) * *(data + 2*3 + 2);
	float n3 = *(data + 2*3 + 1) * *(data + 1*3 + 2) * *(data + 0*3 + 0);

	return p1+p2+p3-n1-n2-n3;

}

static float det2x2(float *data){
	float p1 = *(data + 0*2 + 0) * *(data + 1*2 + 1);
	float n1 = *(data + 0*2 + 2) * *(data + 1*2 + 1);

	return p1-n1;
}


static float det2x2args(float a11, float a12, float a21, float a22){
	return a11*a22 - a12*a21;
}

static float inverse3x3(float *data, float *out){
	float det = det3x3(data);
	if (det==0)
		return false;
	out[0] =  1*det2x2args(data[4], data[5], data[7], data[8])/det;
	out[1] = -1*det2x2args(data[1], data[2], data[7], data[8])/det;
	out[2] =  1*det2x2args(data[1], data[2], data[4], data[5])/det;
	out[3] = -1*det2x2args(data[3], data[5], data[6], data[8])/det;
	out[4] =  1*det2x2args(data[0], data[2], data[6], data[8])/det;
	out[5] = -1*det2x2args(data[0], data[2], data[3], data[5])/det;
	out[6] =  1*det2x2args(data[3], data[4], data[6], data[7])/det;
	out[7] = -1*det2x2args(data[0], data[1], data[6], data[7])/det;
	out[8] =  1*det2x2args(data[0], data[1], data[3], data[4])/det;

	return true;

}

static void mul3x3x3(float *a, float *b, float *out){
	 out[0] = a[0]*b[0] + a[1]*b[3] + a[2]*b[6];
	 out[1] = a[0]*b[1] + a[1]*b[4] + a[2]*b[7];
	 out[2] = a[0]*b[2] + a[1]*b[5] + a[2]*b[8];

	 out[3] = a[3]*b[0] + a[4]*b[3] + a[5]*b[6];
	 out[4] = a[3]*b[1] + a[4]*b[4] + a[5]*b[7];
	 out[5] = a[3]*b[2] + a[4]*b[5] + a[5]*b[8];

	 out[6] = a[6]*b[0] + a[7]*b[3] + a[8]*b[6];
	 out[7] = a[6]*b[1] + a[7]*b[4] + a[8]*b[7];
	 out[8] = a[6]*b[2] + a[7]*b[5] + a[8]*b[8];
}

__device__ void mul3x3x1(float *a, float *b, float *out){

	out[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
	out[1] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2];
	out[2] = a[6]*b[0] + a[7]*b[1] + a[8]*b[2];

}

static void mul3x3x1h(float *a, float *b, float *out){

	out[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
	out[1] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2];
	out[2] = a[6]*b[0] + a[7]*b[1] + a[8]*b[2];

}

void trans3x3(float *data, float *out){
	out[0] = data[0];
	out[1] = data[3];
	out[2] = data[6];
	out[3] = data[1];
	out[4] = data[4];
	out[5] = data[7];
	out[6] = data[2];
	out[7] = data[5];
	out[8] = data[8];

}


__device__ float phi_to_j(float phi){
	float j = (float)(((OUT_Y-1)*phi  +  (OUT_Y-1)*0)/M_PI);
	return j;
}

__device__ float theta_to_i(float theta){
	float i = (float)(((((OUT_X-1)*theta)) + ((OUT_X-1)*0))/(2*M_PI));
	return i;
}

static float deg_to_rad(float deg){
	float rad = deg*M_PI/180;
	return rad;	
}

static float rad_to_deg(float rad){
	float deg = rad*180/M_PI;
	return deg;	
}


static void sphere_to_cart(float3 *sph, float3 *cart){

	float x = sph->z*cos(sph->x)*sin(sph->y);
	float y = sph->z*sin(sph->x)*sin(sph->y);
	float z = sph->z*cos(sph->y);

	cart->x = x;
	cart->y = y;
	cart->z = z;
}

__device__ void cart_to_sphere(float3 *cart, float3 *sph){
	float theta;
	float phi;
	float r = sqrt((cart->x*cart->x) + (cart->y*cart->y) + (cart->z*cart->z));
	
	if (cart->x==0) 
		if (cart->y < 0)
			theta = -M_PI/2;
		else
			theta = M_PI/2;
	else
		theta = atan(cart->y/cart->x);

	phi = acos(cart->z/r);

	if (cart->x<0)
		phi*=-1;


	sph->x = theta;
	sph->y = phi;
	sph->z = r;
}

float distance(float3 *p1, float3 *p2){

	float x = p2->x - p1->x;
	float y = p2->y - p1->y;
	float z = p2->z - p1->z;

	float r = sqrt((x*x) + (y*y) + (z*z));

	return r;
}


struct plane {


	float3 dots[4];

};

int4 *xymap;
unsigned int *plane;
float4 *bmapg;

static void create_out_plane(float *coord, float fov, float ratio){

	float3 cart_c;

	float3 cart_1,cart_2,cart_3,cart_4;
	float3 sph_t;

	float phi_c = deg_to_rad(ANGLE_PHI);
	float theta_c = deg_to_rad(ANGLE_THETA);

	float x,y,x2,y2;

	y = fov/sqrt((ratio*ratio) + 1);
	x = ratio*y;
	x2 = x/2.0;
	y2 = y/2.0;

	//printf("x2: %f y2 %f\n",x2,y2 );

	//float fov2 = fov/2.0;

	float phi_1 =		phi_c 	- 	y2;
	float theta_1 = 	theta_c	+	x2;

	float phi_2 = phi_c - y2;
	float theta_2 = theta_c - x2;

	float phi_3 = phi_c + y2;
	float theta_3 = theta_c - x2;

	float phi_4 = phi_c + y2;
	float theta_4 = theta_c + x2;

	sph_t.x = theta_1;
	sph_t.y = phi_1;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	if (phi_1<0){
		phi_1 *= -1;
		if (theta_1<M_PI)
			theta_1 +=M_PI;
		else
			theta_1 -=M_PI;
	}

	if (theta_1<0){
		theta_1 = 2*M_PI + theta_1;
	}

	sphere_to_cart(&sph_t, &cart_1);
	//printf("theta1: %f phi1 %f\n",rad_to_deg(theta_1), rad_to_deg(phi_1) );

	sph_t.x = theta_2;
	sph_t.y = phi_2;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	if (phi_2<0){
		phi_2 *= -1;
		if (theta_2<M_PI)
			theta_2 +=M_PI;
		else
			theta_2 -=M_PI;
	}

	if (theta_2<0){
		theta_2 = 2*M_PI + theta_2;
	}


	sphere_to_cart(&sph_t, &cart_2);
	//printf("theta2: %f phi2 %f\n",rad_to_deg(theta_2), rad_to_deg(phi_2));

	sph_t.x = theta_3;
	sph_t.y = phi_3;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);

	if (phi_3<0){
		phi_3 *= -1;
		if (theta_3<M_PI)
			theta_3 +=M_PI;
		else
			theta_3 -=M_PI;
	}

	if (theta_3<0){
		theta_3 = 2*M_PI + theta_3;
	}


	sphere_to_cart(&sph_t, &cart_3);
	//printf("theta3: %f phi3 %f\n",rad_to_deg(theta_3), rad_to_deg(phi_3) );

	sph_t.x = theta_4;
	sph_t.y = phi_4;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);

	if (phi_4<0){
		phi_4 *= -1;
		if (theta_4<M_PI)
			theta_4 +=M_PI;
		else
			theta_4 -=M_PI;
	}

	if (theta_4<0){
		theta_4 = 2*M_PI + theta_4;
	}

	sphere_to_cart(&sph_t, &cart_4);
//	printf("theta4: %f phi4 %f\n",rad_to_deg(theta_4), rad_to_deg(phi_4) );

	cart_c.x = (cart_1.x + cart_3.x)/2;
	cart_c.y = (cart_1.y + cart_3.y)/2;
	cart_c.z = (cart_1.z + cart_3.z)/2;

	// printf("p1 x: %f, y: %f, z: %f\n",cart_1.x,cart_1.y,cart_1.z );
	// printf("p2 x: %f, y: %f, z: %f\n",cart_2.x,cart_2.y,cart_2.z );
	// printf("p3 x: %f, y: %f, z: %f\n",cart_3.x,cart_3.y,cart_3.z );
	// printf("p4 x: %f, y: %f, z: %f\n",cart_4.x,cart_4.y,cart_4.z );
	// printf("center x: %f y: %f z: %f\n",cart_c.x, cart_c.y, cart_c.z );
	coord[0]=cart_2.x;coord[1]=cart_2.y;coord[2]=cart_2.z;
	coord[3]=cart_4.x;coord[4]=cart_4.y;coord[5]=cart_4.z;
	coord[6]=cart_3.x;coord[7]=cart_3.y;coord[8]=cart_3.z;
}

static void create_project_matrix(float *outplane, float *inputplane, float *pmatrix){
//	int i,j;
	
    float pa[9];// = mat(3,3);
    float p1[3];
   
    pa[0] = inputplane[0]; 					pa[1] = inputplane[1]; 					pa[2] = 1;
    pa[3] = inputplane[0] + inputplane[2]; 	pa[4] = inputplane[1]; 					pa[5] = 1;
    pa[6] = inputplane[0] + inputplane[2]; 	pa[7] = inputplane[1] + inputplane[3]; 	pa[8] = 1;
  
    p1[0] = outplane[0];
    p1[1] = outplane[3];
    p1[2] = outplane[6];

    float l1[3];
    float invpa[9];
    inverse3x3(pa, invpa);
    mul3x3x1h(invpa, p1, l1);

    pmatrix[0] = l1[0];
    pmatrix[1] = l1[1];
    pmatrix[2] = l1[2];

    p1[0] = outplane[1];
    p1[1] = outplane[4];
    p1[2] = outplane[7];

    mul3x3x1h(invpa, p1, l1);

    pmatrix[3] = l1[0];
    pmatrix[4] = l1[1];
    pmatrix[5] = l1[2];

    p1[0] = outplane[2];
    p1[1] = outplane[5];
    p1[2] = outplane[8];

    mul3x3x1h(invpa, p1, l1);

    pmatrix[6] = l1[0];
    pmatrix[7] = l1[1];
    pmatrix[8] = l1[2];
}

static void create_rotate_matrix(float theta, float phi, float *rmatrix){

    float fa[9];
    float fb[9];
   

    fb[0] = 1; fb[1] = 0; 			fb[2] = 0;
    fb[3] = 0; fb[4] = cos(phi);	fb[5] = -sin(phi);
    fb[6] = 0; fb[7] = sin(phi);	fb[8] = cos(phi);

    fa[0] = cos(theta); fa[1] = -sin(theta); 	fa[2] = 0;
    fa[3] = sin(theta); fa[4] = cos(theta); 	fa[5] = 0;
    fa[6] = 0; 			fa[7] = 0;				fa[8] = 1;

    
 	//rmatrix = fa * fb;
    mul3x3x3(fa,fb,rmatrix);
}


__device__ unsigned int argb_interpolate(struct float4 vec, unsigned int q1, unsigned int q2, unsigned int q3, unsigned int q4 ){

	// float4 vec;
	// vec.x = gvec.x;
	// vec.y = gvec.y;
	// vec.z = gvec.z;
	// vec.w = gvec.w;

	unsigned char a1 = (unsigned char)((q1 & 0xFF000000) >> 24);
	unsigned char a2 = (unsigned char)((q2 & 0xFF000000) >> 24);
	unsigned char a3 = (unsigned char)((q3 & 0xFF000000) >> 24);
	unsigned char a4 = (unsigned char)((q4 & 0xFF000000) >> 24);

	unsigned char r1 = (unsigned char)((q1 & 0x00FF0000) >> 16);
	unsigned char r2 = (unsigned char)((q2 & 0x00FF0000) >> 16);
	unsigned char r3 = (unsigned char)((q3 & 0x00FF0000) >> 16);
	unsigned char r4 = (unsigned char)((q4 & 0x00FF0000) >> 16);

	unsigned char g1 = (unsigned char)((q1 & 0x0000FF00) >> 8);
	unsigned char g2 = (unsigned char)((q2 & 0x0000FF00) >> 8);
	unsigned char g3 = (unsigned char)((q3 & 0x0000FF00) >> 8);
	unsigned char g4 = (unsigned char)((q4 & 0x0000FF00) >> 8);

	unsigned char b1 = (unsigned char)((q1 & 0x000000FF) >> 0);
	unsigned char b2 = (unsigned char)((q2 & 0x000000FF) >> 0);
	unsigned char b3 = (unsigned char)((q3 & 0x000000FF) >> 0);
	unsigned char b4 = (unsigned char)((q4 & 0x000000FF) >> 0);

	unsigned char a = 	floor((vec.x * (float)a1 +
						vec.y * (float)a2 +
						vec.z * (float)a3 +
						vec.w * (float)a4));

	unsigned char r = 	floor((vec.x * (float)r1 +
						vec.y * (float)r2 +
						vec.z * (float)r3 +
						vec.w * (float)r4));

	unsigned char g = 	floor((vec.x * (float)g1 +
						vec.y * (float)g2 +
						vec.z * (float)g3 +
						vec.w * (float)g4));

	unsigned char b = 	floor((vec.x * (float)b1 +
						vec.y * (float)b2 +
						vec.z * (float)b3 +
						vec.w * (float)b4));

	unsigned int ret = (a << 24) | (r << 16) | (g << 8) | (b << 0);
	return ret;
}

__device__ unsigned int interpolate(float x, float y, unsigned int q1, unsigned int q2, unsigned int q3, unsigned int q4){
	double nv_bicubic[16];
	double nv_vals[4], nv_b[4];

	double nv_inv[16];
	double nv_transp[16];

	float4 bmap;
	unsigned int val;//,x1,y1,x2,y2;

	nv_bicubic[0] = 1.0f;
	nv_bicubic[1] = floor(x);
	nv_bicubic[2] = floor(y);
	nv_bicubic[3] = floor(x) * floor(y);
	nv_bicubic[4] = 1.0f;
	nv_bicubic[5] = floor(x);
	nv_bicubic[6] = ceil(y);
	nv_bicubic[7] = floor(x) * ceil(y);
	nv_bicubic[8] = 1.0f;
	nv_bicubic[9] = ceil(x);
	nv_bicubic[10] = floor(y);
	nv_bicubic[11] = ceil(x) * floor(y);
	nv_bicubic[12] = 1.0f;
	nv_bicubic[13] = ceil(x);
	nv_bicubic[14] = ceil(y);
	nv_bicubic[15] = ceil(x) * ceil(y);

	nv_vals[0] = 1.0f;
	nv_vals[1] = x;
	nv_vals[2] = y;
	nv_vals[3] = x*y;

	if(inverse4x4(nv_bicubic, nv_inv)){
		trans4x4(nv_inv, nv_transp);
		mul4x4x1(nv_transp, nv_vals, nv_b);
		bmap.x = nv_b[0];
		bmap.y = nv_b[1];
		bmap.z = nv_b[2];
		bmap.w = nv_b[3];
	}else{
		bmap.x = 0.25f;
		bmap.y = 0.25f;
		bmap.z = 0.25f;
		bmap.w = 0.25f;
	}

	val =  argb_interpolate(bmap, q1,q2,q3,q4); 

	return val;
}	
__device__ unsigned int dotsmultiply(cudaTextureObject_t xytext, /*float4 *bmappt*/cudaTextureObject_t btext, unsigned int **sources, int y, int x){

	uint4 xymap = tex2D<uint4>(xytext, x,y);
	float4 bmap = tex2D<float4>(btext, x,y);
	unsigned int sid = xymap.x >> 16;

	unsigned int *sdatapt = sources[sid];//&sdata[sid][0][0];

	unsigned int x1 = xymap.x & 0x0000FFFF;
	unsigned int x2 = xymap.y;
	unsigned int y1 = xymap.z;
	unsigned int y2 = xymap.w;
	//printf("%d %d %d %d\n", x1,x2,y1,y2 );


	if ((x1>SOURCE_X) || (x2>SOURCE_X) || (y1>SOURCE_Y) || (y2>SOURCE_Y))
		return 0x0;
	
	//unsigned int t=0;

	unsigned int q =argb_interpolate(bmap, 	
									 *(sdatapt + SOURCE_X*y1 + x1),
									 *(sdatapt + SOURCE_X*y2 + x1), 
									 *(sdatapt + SOURCE_X*y1 + x2),
									 *(sdatapt + SOURCE_X*y2 + x2)	); 
	//printf("%d %d %d %d\n",x1, x2, y1,y2 );
	//printf("%x\n",*(sdatapt + SOURCE_X*y1 + x1) );
	return q;
}

__global__ void create_pano(float *dev_wm, /*uint4 *dev_xymap*/ cudaTextureObject_t xytext, cudaTextureObject_t btext/*float4 *dev_bmap*/, 	unsigned int *dev_source0,
														unsigned int *dev_source1,
														unsigned int *dev_source2,
														unsigned int *dev_source3,
														unsigned int *dev_source4,
														unsigned int *dev_source5,
														unsigned int *dev_plane){

	float nv_invec[3];
	float nv_outvec[3];
	unsigned int *sources[6];
	//float dev_wm_s[9];
	//int i;
	float3 cr,sp;
	float jff, iff;

	sources[0] = dev_source0;
	sources[1] = dev_source1;
	sources[2] = dev_source2;
	sources[3] = dev_source3;
	sources[4] = dev_source4;
	sources[5] = dev_source5;

	int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

	nv_invec[0] = (float)ii;
	nv_invec[1] = (float)jj;
	nv_invec[2] = (float)1;
	// for (i=0;i<9;i++)
	// 	dev_wm_s[i] = dev_wm[9];
	
	mul3x3x1(dev_wm, nv_invec, nv_outvec);

	cr.x = nv_outvec[0];
	cr.y = nv_outvec[1];
	cr.z = nv_outvec[2];

	cart_to_sphere(&cr, &sp);
	if (sp.y<0){
		sp.y *= -1;
		if (sp.x<M_PI)
			sp.x +=M_PI;
		else
			sp.x -=M_PI;
	}else if (sp.y>M_PI){
		sp.y = M_PI - (sp.y - M_PI);
		if (sp.x<M_PI)
			sp.x +=M_PI;
		else
			sp.x -=M_PI;
	}
	if (sp.x<0){
		sp.x = (2*M_PI) + sp.x;
	}else if (sp.x>(2*M_PI))
		sp.x = sp.x - (2*M_PI);
		jff = phi_to_j(sp.y);
		iff = theta_to_i(sp.x);



		unsigned int q1 = dotsmultiply(xytext, btext, sources, floor(jff), floor(iff));
		unsigned int q2 = dotsmultiply(xytext, btext, sources, ceil(jff), floor(iff));
		unsigned int q3 = dotsmultiply(xytext, btext, sources, floor(jff), ceil(iff));
		unsigned int q4 = dotsmultiply(xytext, btext, sources, ceil(jff), ceil(iff));
#ifndef CUDA_PANO_LIB
		*(dev_plane + jj*DEST_X + ii) = interpolate(iff, jff, q1,q2,q3,q4 );
#else
		*(dev_plane + jj*DEST_X + ii) = interpolate(iff, jff, q1,q2,q3,q4 );
		
#endif
}

#ifdef CUDA_PANO_LIB

extern "C" void gstcuda_process(){
	cudaSetDevice(0);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 grid(gstpano.dest_width/16,gstpano.dest_height/16);
    dim3 block(16,16);

    cudaEventRecord(start);

	create_pano<<<grid,block>>>(	gstpano.dev.matrix,
    								gstpano.xytext,
    							gstpano.btext,
    							gstpano.dev.sdata[0],
    							gstpano.dev.sdata[1],
    							gstpano.dev.sdata[2],
    							gstpano.dev.sdata[3],
    							gstpano.dev.sdata[4],
    							gstpano.dev.sdata[5],
    							gstpano.dev.panodata
    							);	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("kernel execution: %fms\n", milliseconds);
}

extern "C" void gstcuda_get_output(void *out){
	cudaSetDevice(0);
	HANDLE_ERROR(cudaMemcpy( out, gstpano.dev.panodata, 4*gstpano.dest_width*gstpano.dest_height, cudaMemcpyDeviceToHost)); 

}

#endif

#ifdef CUDA_PANO_LIB
extern "C" void gstcuda_update_matrix(float fov, float phi, float theta){
	static int once = 0;
	
	cudaSetDevice(0);
//	printf("theta: %f\n",theta );
	float outplane[9];
	float pmatrix[9];
	float rmatrix[9];
	//float nv_wm[9];
	float inputplane[4];

	gstpano.phi = phi;
	gstpano.theta = theta;
	gstpano.fov = fov;

	create_out_plane(outplane, fov, gstpano.dest_width / gstpano.dest_height);

	inputplane[0] = 0;
	inputplane[1] = 0;
	inputplane[2] = gstpano.dest_width;
	inputplane[3] = gstpano.dest_height;

	create_project_matrix(outplane, inputplane, pmatrix);
								//theta 		//phi
	create_rotate_matrix(theta, phi, rmatrix);

	mul3x3x3(rmatrix,pmatrix, gstpano.matrix);
	if (once==0){
		//allocate matrix on device
		HANDLE_ERROR( cudaMalloc( (void**)&gstpano.dev.matrix, sizeof(gstpano.matrix) ) );

		once = 1;
	}
	HANDLE_ERROR( cudaMemcpy( gstpano.dev.matrix, gstpano.matrix, sizeof(gstpano.matrix), cudaMemcpyHostToDevice ) );
}
#else
void update_matrix(struct pano *pano){
	static int once = 0;
	float outplane[9];
	float pmatrix[9];
	float rmatrix[9];
	//float nv_wm[9];
	float inputplane[4];
	
	create_out_plane(outplane, pano->fov, pano->dest_width / pano->dest_height);

	inputplane[0] = 0;
	inputplane[1] = 0;
	inputplane[2] = DEST_X;
	inputplane[3] = DEST_Y;

	create_project_matrix(outplane, inputplane, pmatrix);
								//theta 		//phi
	create_rotate_matrix(pano->theta, pano->phi, rmatrix);

	mul3x3x3(rmatrix,pmatrix, pano->matrix);
	if (once==0){
		//allocate matrix on device
		HANDLE_ERROR( cudaMalloc( (void**)&pano->dev.matrix, sizeof(pano->matrix) ) );

		once = 1;
	}
	HANDLE_ERROR( cudaMemcpy( pano->dev.matrix, pano->matrix, sizeof(pano->matrix), cudaMemcpyHostToDevice ) );
}
#endif
#ifdef CUDA_PANO_LIB
extern "C" void gstcuda_bmap_config(const char *bmapname){
	cudaSetDevice(0);
	gstpano.bmapfd = fopen(bmapname, "rb");
    if (gstpano.bmapfd==NULL){
    	printf("can't open bmap\n");
    	exit(1);
    }

    HANDLE_ERROR(cudaHostAlloc((void**) &gstpano.bmap, sizeof(float4) * gstpano.pano_width*gstpano.pano_height,cudaHostAllocDefault));
    HANDLE_ERROR( cudaMalloc( (void**)&gstpano.dev.bmap,  sizeof(float4)*gstpano.pano_width*gstpano.pano_height ) );

    fread(gstpano.bmap, sizeof(float4), gstpano.pano_width*gstpano.pano_height, gstpano.bmapfd);
    fflush(gstpano.bmapfd);

    HANDLE_ERROR( cudaMemcpy( gstpano.dev.bmap, gstpano.bmap, sizeof(float4)*gstpano.pano_width*gstpano.pano_height, cudaMemcpyHostToDevice ) );
    fclose(gstpano.bmapfd);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
  	resDesc.resType = cudaResourceTypePitch2D;
  	resDesc.res.pitch2D.devPtr = gstpano.dev.bmap;
  	resDesc.res.pitch2D.pitchInBytes =  sizeof(float4)*gstpano.pano_width;
  	resDesc.res.pitch2D.width = gstpano.pano_width;
	resDesc.res.pitch2D.height = gstpano.pano_height;
	resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
  	resDesc.res.pitch2D.desc.x = 32; // bits per channel
  	resDesc.res.pitch2D.desc.y = 32; // bits per channel
  	resDesc.res.pitch2D.desc.z = 32; // bits per channel
  	resDesc.res.pitch2D.desc.w = 32; // bits per channel
  	

  	cudaTextureDesc textDesc;	
  	memset(&textDesc, 0, sizeof(textDesc));
  	textDesc.readMode = cudaReadModeElementType;

  	gstpano.btext = 0;
  	cudaCreateTextureObject(&gstpano.btext, &resDesc, &textDesc, NULL);
}
extern "C" void gstcuda_xymap_config(const char *xymapname){
	cudaSetDevice(0);
	gstpano.xymapfd = fopen(xymapname, "rb");
    if (gstpano.xymapfd == NULL){
    	printf("can't open xymap\n");
    	exit(1);
    }

    HANDLE_ERROR(cudaHostAlloc((void**) &gstpano.xymap, sizeof(uint4) * gstpano.pano_width*gstpano.pano_height,cudaHostAllocDefault));
    //#error create out allocator function
	//HANDLE_ERROR( cudaMalloc( (void**)&gstpano.dev.panodata, sizeof(uint4)*gstpano.pano_width*gstpano.pano_height ) );
	HANDLE_ERROR( cudaMalloc( (void**)&gstpano.dev.xymap, sizeof(uint4)*gstpano.pano_width*gstpano.pano_height ) );
    fread(gstpano.xymap, sizeof(int4), gstpano.pano_width*gstpano.pano_height, gstpano.xymapfd);
	fflush(gstpano.xymapfd);
    HANDLE_ERROR( cudaMemcpy( gstpano.dev.xymap, gstpano.xymap, sizeof(uint4)*gstpano.pano_width*gstpano.pano_height, cudaMemcpyHostToDevice ) );
    fclose(gstpano.xymapfd);

  	cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
  	resDesc.resType = cudaResourceTypePitch2D;
  	resDesc.res.pitch2D.devPtr = gstpano.dev.xymap;
  	resDesc.res.pitch2D.pitchInBytes =  sizeof(uint4)*gstpano.pano_width;
  	resDesc.res.pitch2D.width = gstpano.pano_width;
	resDesc.res.pitch2D.height = gstpano.pano_height;
	resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
  	resDesc.res.pitch2D.desc.x = 32; // bits per channel
  	resDesc.res.pitch2D.desc.y = 32; // bits per channel
  	resDesc.res.pitch2D.desc.z = 32; // bits per channel
  	resDesc.res.pitch2D.desc.w = 32; // bits per channel
  	

  	cudaTextureDesc textDesc;	
  	memset(&textDesc, 0, sizeof(textDesc));
  	textDesc.readMode = cudaReadModeElementType;

  	gstpano.xytext = 0;
  	cudaCreateTextureObject(&gstpano.xytext, &resDesc, &textDesc, NULL);

  	// 			cudaDestroyTextureObject(texSrc);
			// cudaFree(d_pSrc);

}

extern "C" void *gstcuda_host_alloc(size_t size){
	cudaSetDevice(0);
	void *mem;
	printf("cuda alloc size: %lu...........................................\n",size );
	HANDLE_ERROR(cudaHostAlloc((void**) &mem, size,cudaHostAllocDefault));
	return mem;
}

extern "C" void gstcuda_host_free(void *mem){
	cudaSetDevice(0);
	HANDLE_ERROR(cudaFreeHost(mem));
}

extern "C" void gstcuda_set_dims(
						int pano_width,
						int pano_height,
						int source_width,
						int source_height,
						int dest_width,
						int dest_height
					){
	int i;
	cudaSetDevice(0);
	gstpano.pano_width = pano_width;
    gstpano.pano_height = pano_height;

    gstpano.source_width = source_width;
    gstpano.source_height = source_height;

    gstpano.dest_width = dest_width;
    gstpano.dest_height = dest_height;

	HANDLE_ERROR( cudaMalloc( (void**)&gstpano.dev.panodata, 4*gstpano.dest_width*gstpano.dest_height ) );
	for(i=0;i<6;i++){
		//HANDLE_ERROR(cudaHostAlloc((void**) &pano->sdata[i], 4 * pano->source_width*pano->source_height,cudaHostAllocDefault));
		//fread(pano->sdata[i], 4, pano->source_width*pano->source_height, pano->sdatafd[i]);
		//fflush(pano->sdatafd[i]);
		HANDLE_ERROR( cudaMalloc( (void**)&gstpano.dev.sdata[i], 4 * (gstpano.source_width*gstpano.source_height) ) );

       	//HANDLE_ERROR( cudaMemcpy( pano->dev.sdata[i], pano->sdata[i], 4*pano->source_width*pano->source_height, cudaMemcpyHostToDevice ) );
    	//HANDLE_ERROR(cudaStreamCreate(&stream[i]));
	}
//	HANDLE_ERROR(cudaStreamCreate(&stream[6]));
//	HANDLE_ERROR(cudaStreamCreate(&stream[7]));

}
#else
void open_static_config(struct pano *pano){
	pano->xymapfd = fopen("xymap.bin", "rb");
    if (pano->xymapfd == NULL){
    	printf("can't open xymap\n");
    	exit(1);
    }

    pano->bmapfd = fopen("bmap.bin", "rb");
    if (pano->bmapfd==NULL){
    	printf("can't open bmap\n");
    	exit(1);
    }

    HANDLE_ERROR(cudaHostAlloc((void**) &pano->xymap, sizeof(uint4) * pano->pano_width*pano->pano_height,cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**) &pano->bmap, sizeof(float4) * pano->pano_width*pano->pano_height,cudaHostAllocDefault));

	HANDLE_ERROR(cudaHostAlloc((void**) &pano->panodata, 4 * DEST_X*DEST_Y,cudaHostAllocDefault));
	HANDLE_ERROR( cudaMalloc( (void**)&pano->dev.panodata, 4 * DEST_X*DEST_Y ) );

	HANDLE_ERROR( cudaMalloc( (void**)&pano->dev.xymap, sizeof(uint4)*pano->pano_width*pano->pano_height ) );
    HANDLE_ERROR( cudaMalloc( (void**)&pano->dev.bmap,  sizeof(float4)*pano->pano_width*pano->pano_height ) );

	fread(pano->xymap, sizeof(int4), pano->pano_width*pano->pano_height, pano->xymapfd);
	fread(pano->bmap, sizeof(float4), pano->pano_width*pano->pano_height, pano->bmapfd);

	fflush(pano->xymapfd);
	fflush(pano->bmapfd);

	HANDLE_ERROR( cudaMemcpy( pano->dev.xymap, pano->xymap, sizeof(uint4)*pano->pano_width*pano->pano_height, cudaMemcpyHostToDevice ) );
    //printf("size of bmapg: %lu\n", sizeof(bmapg) );
    HANDLE_ERROR( cudaMemcpy( pano->dev.bmap, pano->bmap, sizeof(float4)*pano->pano_width*pano->pano_height, cudaMemcpyHostToDevice ) );

    fclose(pano->xymapfd);
    fclose(pano->bmapfd);
}
#endif

void open_sources(struct pano *pano){
	int i;
	pano->sdatafd[0] = fopen("./1.rgba", "rb");
    if (pano->sdatafd[0]==NULL){
    	printf("can't open front\n");
    	exit(1);
    }

    pano->sdatafd[1] = fopen("./1.rgba", "rb");
    if (pano->sdatafd[1]==NULL){
    	printf("can't open left\n");
    	exit(1);
    }

    pano->sdatafd[2] = fopen("./1.rgba", "rb");
    if (pano->sdatafd[2]==NULL){
    	printf("can't open right\n");
    	exit(1);
    }

    pano->sdatafd[3] = fopen("./1.rgba", "rb");
    if (pano->sdatafd[3]==NULL){
    	printf("can't open back\n");
    	exit(1);
    }

    pano->sdatafd[4] = fopen("./1.rgba", "rb");
    if (pano->sdatafd[4]==NULL){
    	printf("can't open top\n");
    	exit(1);
    }

    pano->sdatafd[5] = fopen("./1.rgba", "rb");
    if (pano->sdatafd[5]==NULL){
    	printf("can't open back\n");
    	exit(1);
    }

    for(i=0;i<6;i++){
		HANDLE_ERROR(cudaHostAlloc((void**) &pano->sdata[i], 4 * pano->source_width*pano->source_height,cudaHostAllocDefault));
		fread(pano->sdata[i], 4, pano->source_width*pano->source_height, pano->sdatafd[i]);
		fflush(pano->sdatafd[i]);
		HANDLE_ERROR( cudaMalloc( (void**)&pano->dev.sdata[i], 4 * (pano->source_width*pano->source_height) ) );

       	HANDLE_ERROR( cudaMemcpy( pano->dev.sdata[i], pano->sdata[i], 4*pano->source_width*pano->source_height, cudaMemcpyHostToDevice ) );
    
	}
}

#ifdef CUDA_PANO_LIB
extern "C" void gstcuda_update_source(int sourceid, unsigned int *data){
	cudaSetDevice(0);
//	printf("dest: %p src: %p size: %d\n",gstpano.dev.sdata[sourceid], data,  4*gstpano.source_width*gstpano.source_height);

	HANDLE_ERROR( cudaMemcpy( gstpano.dev.sdata[sourceid], (void *)data, 4*gstpano.source_width*gstpano.source_height, cudaMemcpyHostToDevice ) );	
}

extern "C" void gstcuda_sync_all(){
	cudaSetDevice(0);
	cudaDeviceSynchronize();
}

extern "C" void gstcuda_sync_stream(int id){
	cudaSetDevice(0);
}

#else

void update_sources(struct pano *pano){
	 int i;
	 for(i=0;i<6;i++){
		fread(pano->sdata[i], 4, pano->source_width*pano->source_height, pano->sdatafd[i]);
		fflush(pano->sdatafd[i]);
		
       	HANDLE_ERROR( cudaMemcpy( pano->dev.sdata[i], pano->sdata[i], 4*pano->source_width*pano->source_height, cudaMemcpyHostToDevice ) );
    
	}
}
#endif

#ifndef CUDA_PANO_LIB
int main(){

//    int i;

    struct pano panorama;
    panorama.pano_width = 3600;
    panorama.pano_height = 1800;

    panorama.source_width = 1920;
    panorama.source_height = 1080;

    panorama.dest_width = DEST_X;
    panorama.dest_height = DEST_Y;

    panorama.theta = deg_to_rad(90);
    panorama.phi = deg_to_rad(125);
    panorama.fov = deg_to_rad(120);

    update_matrix(&panorama);
    open_static_config(&panorama);
    open_sources(&panorama);
    
    dim3 grid(panorama.dest_width/8,panorama.dest_height/8);
    dim3 block(8,8);

    FILE *planefd = fopen("ooo.rgba", "wb+");
    create_pano<<<grid,block>>>(	panorama.dev.matrix,
    							panorama.dev.xymap,
    							panorama.dev.bmap,
    							panorama.dev.sdata[0],
    							panorama.dev.sdata[1],
    							panorama.dev.sdata[2],
    							panorama.dev.sdata[3],
    							panorama.dev.sdata[4],
    							panorama.dev.sdata[5],
    							panorama.dev.panodata
    							);

    HANDLE_ERROR(cudaMemcpy( panorama.panodata, panorama.dev.panodata, 4*DEST_Y*DEST_X, cudaMemcpyDeviceToHost )); 


	fwrite(panorama.panodata, DEST_Y*DEST_X,4,planefd);
	fflush(planefd);
	fclose(planefd);


}
#endif

