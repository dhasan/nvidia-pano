
#include	<armadillo>
#include 	<fcntl.h>

#define 	OUT_X				(3600)
#define 	OUT_Y				(1800)

#define 	SOURCE_Y 			(1200)
#define 	SOURCE_X 			(1200)

#define DEST_X 	(640)
#define DEST_Y	(640)

#define ANGLE_PHI	(0)
#define ANGLE_THETA	(0)

#define FOV_X		(90)
#define FOV_Y		(90)

#define RADIUS		(1)
//#define RADIUS		(OUT_X/(2*datum::pi))

unsigned int sdata[6][1200][1200];

#define MAX(a,b) (a>b)?a:b
#define MIN(a,b) (a<b)?a:b

using namespace arma;

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

void mul4x4x4(float *a, float *b, float *out){
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



void mul4x4x1(double *a, double *b, double *out){
	out[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
	out[1] = a[4]*b[0] + a[5]*b[1] + a[6]*b[2] + a[7]*b[3];
	out[2] = a[8]*b[0] + a[9]*b[1] + a[10]*b[2] + a[11]*b[3];
	out[3] = a[12]*b[0] + a[13]*b[1] + a[14]*b[2] + a[15]*b[3];
}



void trans4x4(double *data, double *out){
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

bool inverse4x4(double *a, double *b){
	double det = det4x4(a);
//	static int s=0;
	if (det==0.0f){
//		printf("null%d\n",s++);

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

float det3x3(float *data){

	float p1 = *(data + 0*3 + 0) * *(data + 1*3 + 1) * *(data + 2*3 + 2);
	float p2 = *(data + 0*3 + 1) * *(data + 1*3 + 2) * *(data + 2*3 + 0);
	float p3 = *(data + 1*3 + 0) * *(data + 2*3 + 1) * *(data + 0*3 + 2);

	float n1 = *(data + 0*3 + 2) * *(data + 1*3 + 1) * *(data + 2*3 + 0);
	float n2 = *(data + 1*3 + 0) * *(data + 0*3 + 1) * *(data + 2*3 + 2);
	float n3 = *(data + 2*3 + 1) * *(data + 1*3 + 2) * *(data + 0*3 + 0);

	return p1+p2+p3-n1-n2-n3;

}

float det2x2(float *data){
	float p1 = *(data + 0*2 + 0) * *(data + 1*2 + 1);
	float n1 = *(data + 0*2 + 2) * *(data + 1*2 + 1);

	return p1-n1;
}


float det2x2args(float a11, float a12, float a21, float a22){
	return a11*a22 - a12*a21;
}

float inverse3x3(float *data, float *out){
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

void mul3x3x3(float *a, float *b, float *out){
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

void mul3x3x1(float *a, float *b, float *out){

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


float phi_to_j(float phi){
	//float j = (float)((((OUT_Y-1)*phi) + (((OUT_Y-1)*datum::pi)/2))/datum::pi);
	float pi2 = datum::pi/2;
	float j = (float)(((OUT_Y-1)*phi  +  (OUT_Y-1)*0)/datum::pi);
	return j;
}

float theta_to_i(float theta){
	float i = (float)(((((OUT_X-1)*theta)) + ((OUT_X-1)*0))/(2*datum::pi));
	return i;
}

float deg_to_rad(float deg){
	float rad = deg*datum::pi/180;
	return rad;	
}

float rad_to_deg(float rad){
	float deg = rad*180/datum::pi;
	return deg;	
}


void sphere_to_cart(float3 *sph, float3 *cart){

	float x = sph->z*cos(sph->x)*sin(sph->y);
	float y = sph->z*sin(sph->x)*sin(sph->y);
	float z = sph->z*cos(sph->y);

	cart->x = x;
	cart->y = y;
	cart->z = z;
}

void cart_to_sphere(float3 *cart, float3 *sph){
	float theta;
	float phi;
	float r = sqrt((cart->x*cart->x) + (cart->y*cart->y) + (cart->z*cart->z));
	
	if (cart->x==0) 
		if (cart->y < 0)
			theta = -datum::pi/2;
		else
			theta = datum::pi/2;
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

struct int4 {
	 unsigned int x;
	 unsigned int y;
	 unsigned int z;
	 unsigned int w;
};

unsigned int pano[OUT_Y][OUT_X];
struct int4 xymap[OUT_Y][OUT_X];
//unsigned int bmap[OUT_Y][OUT_X];

unsigned int plane[DEST_Y][DEST_X];
struct float4 bmapg[OUT_Y][OUT_X];

void create_out_plane(mat& coord, float fov){

	float3 cart_c;

	float3 cart_1,cart_2,cart_3,cart_4;
	float3 sph_t;

	float phi_c = deg_to_rad(ANGLE_PHI);
	float theta_c = deg_to_rad(ANGLE_THETA);

	float fov2 = fov/2.0;

	float phi_1 =		phi_c 	- 	fov2;
	float theta_1 = 	theta_c	+	fov2;

	float phi_2 = phi_c - fov2;
	float theta_2 = theta_c - fov2;

	float phi_3 = phi_c + fov2;
	float theta_3 = theta_c - fov2;

	float phi_4 = phi_c + fov2;
	float theta_4 = theta_c + fov2;

	sph_t.x = theta_1;
	sph_t.y = phi_1;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	if (phi_1<0){
		phi_1 *= -1;
		if (theta_1<datum::pi)
			theta_1 +=datum::pi;
		else
			theta_1 -=datum::pi;
	}

	if (theta_1<0){
		theta_1 = 2*datum::pi + theta_1;
	}

	sphere_to_cart(&sph_t, &cart_1);
	printf("theta1: %f phi1 %f\n",rad_to_deg(theta_1), rad_to_deg(phi_1) );

	sph_t.x = theta_2;
	sph_t.y = phi_2;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	if (phi_2<0){
		phi_2 *= -1;
		if (theta_2<datum::pi)
			theta_2 +=datum::pi;
		else
			theta_2 -=datum::pi;
	}

	if (theta_2<0){
		theta_2 = 2*datum::pi + theta_2;
	}


	sphere_to_cart(&sph_t, &cart_2);
	printf("theta2: %f phi2 %f\n",rad_to_deg(theta_2), rad_to_deg(phi_2));

	sph_t.x = theta_3;
	sph_t.y = phi_3;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);

	if (phi_3<0){
		phi_3 *= -1;
		if (theta_3<datum::pi)
			theta_3 +=datum::pi;
		else
			theta_3 -=datum::pi;
	}

	if (theta_3<0){
		theta_3 = 2*datum::pi + theta_3;
	}


	sphere_to_cart(&sph_t, &cart_3);
	printf("theta3: %f phi3 %f\n",rad_to_deg(theta_3), rad_to_deg(phi_3) );

	sph_t.x = theta_4;
	sph_t.y = phi_4;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);

	if (phi_4<0){
		phi_4 *= -1;
		if (theta_4<datum::pi)
			theta_4 +=datum::pi;
		else
			theta_4 -=datum::pi;
	}

	if (theta_4<0){
		theta_4 = 2*datum::pi + theta_4;
	}

	sphere_to_cart(&sph_t, &cart_4);
	printf("theta4: %f phi4 %f\n",rad_to_deg(theta_4), rad_to_deg(phi_4) );

	cart_c.x = (cart_1.x + cart_3.x)/2;
	cart_c.y = (cart_1.y + cart_3.y)/2;
	cart_c.z = (cart_1.z + cart_3.z)/2;

	printf("p1 x: %f, y: %f, z: %f\n",cart_1.x,cart_1.y,cart_1.z );
	printf("p2 x: %f, y: %f, z: %f\n",cart_2.x,cart_2.y,cart_2.z );
	printf("p3 x: %f, y: %f, z: %f\n",cart_3.x,cart_3.y,cart_3.z );
	printf("p4 x: %f, y: %f, z: %f\n",cart_4.x,cart_4.y,cart_4.z );
	printf("center x: %f y: %f z: %f\n",cart_c.x, cart_c.y, cart_c.z );
	coord(0,0)=cart_2.x;coord(0,1)=cart_2.y;coord(0,2)=cart_2.z;
	coord(1,0)=cart_4.x;coord(1,1)=cart_4.y;coord(1,2)=cart_4.z;
	coord(2,0)=cart_3.x;coord(2,1)=cart_3.y;coord(2,2)=cart_3.z;
}

void create_project_matrix(mat& outplane, mat& inputplane, mat& pmatrix){
	int i,j;
	
    mat pa = mat(3,3);
   
    pa 	<< inputplane(0) 					<<	inputplane(1) 					<< 1 << endr
    	<< inputplane(0) + inputplane(2) 	<< 	inputplane(1) 					<< 1 << endr
    	<< inputplane(0) + inputplane(2) 	<< inputplane(1) + inputplane(3) 	<< 1 << endr;
  
    printf("outplane:\n");
    outplane.print();
    printf("inputplane:\n");
    inputplane.print();

    vec p1;
    p1  << outplane(0,0) << endr
    	<< outplane(1,0) << endr
    	<< outplane(2,0) << endr;
    	

    vec l1;
 
    l1 =inv(pa) * p1;

    pmatrix(0,0) = l1(0);
    pmatrix(0,1) = l1(1);
    pmatrix(0,2) = l1(2);


    p1  << outplane(0,1) << endr
    	<< outplane(1,1)  << endr
    	<< outplane(2,1)  << endr;

    l1 = inv(pa) * p1;


    pmatrix(1,0) = l1(0);
    pmatrix(1,1) = l1(1);
    pmatrix(1,2) = l1(2);

    p1  << outplane(0,2) << endr
    	<< outplane(1,2)  << endr
    	<< outplane(2,2)  << endr;

    l1 = inv(pa)*p1;

    pmatrix(2,0) = l1(0);
    pmatrix(2,1) = l1(1);
    pmatrix(2,2) = l1(2);
}

void create_rotate_matrix(float theta, float phi, mat& rmatrix){

    mat fa = mat(3,3);
    mat fb = mat(3,3);

    fb 	<< 1 		<< 0 			<< 0 			<< endr 
    	<< 0 		<< cos(phi) 	<< -sin(phi) 	<< endr
    	<< 0 		<< sin(phi)		<< cos(phi)		<< endr;

    fa 	<< cos(theta) 	<< -sin(theta) 	<< 0 <<endr
    	<< sin(theta) 	<<  cos(theta) 	<< 0 <<endr
    	<< 0 			<< 	0 			<< 1 <<endr;

 	
 	rmatrix = fa * fb;

}


unsigned int argb_interpolate(struct float4 *vec, unsigned int q1, unsigned int q2, unsigned int q3, unsigned int q4 ){

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

	unsigned char a = 	floor((vec->x * (float)a1 +
						vec->y * (float)a2 +
						vec->z * (float)a3 +
						vec->w * (float)a4));

	unsigned char r = 	floor((vec->x * (float)r1 +
						vec->y * (float)r2 +
						vec->z * (float)r3 +
						vec->w * (float)r4));

	unsigned char g = 	floor((vec->x * (float)g1 +
						vec->y * (float)g2 +
						vec->z * (float)g3 +
						vec->w * (float)g4));

	unsigned char b = 	floor((vec->x * (float)b1 +
						vec->y * (float)b2 +
						vec->z * (float)b3 +
						vec->w * (float)b4));

	unsigned int ret = (a << 24) | (r << 16) | (g << 8) | (b << 0);
	return ret;
}

unsigned int interpolate(float x, float y, unsigned int q1, unsigned int q2, unsigned int q3, unsigned int q4){
	double nv_bicubic[16];
	double nv_vals[4], nv_b[4];

	double nv_inv[16];
	double nv_transp[16];

	float4 bmap;
	unsigned int val,x1,y1,x2,y2;

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
		bmap.x = 1.0f;
		bmap.y = 0;
		bmap.z = 0;
		bmap.w = 0;
	}

	val =  argb_interpolate(&bmap, q1,q2,q3,q4); 

	return val;
}	
unsigned int dotsmultiply(int y, int x){
	int4 *xymappt = &xymap[0][0];
	struct float4 *bmappt = &bmapg[0][0];
	unsigned int *sdatapt = &sdata[sid][0][0];
	
	xymappt += (y*OUT_X + x);

	unsigned int sid = xymappt->x >> 16;
	unsigned int x1 = xymappt->x & 0x0000FFFF;
	unsigned int x2 = xymappt->y;
	unsigned int y1 = xymappt->z;
	unsigned int y2 = xymappt->w;

	unsigned int q =argb_interpolate(bmappt + y*OUT_X + x, 	
									 *(sdatapt + SOURCE_X*y1 + x1),
									 *(sdatapt + SOURCE_X*y2 + x1), 
									 *(sdatapt + SOURCE_X*y1 + x2),
									 *(sdatapt + SOURCE_X*y2 + x2)	); 

	return q;
}

int main(){
	int ii,jj;
    int i,j;
    vec invec,outvec;
	float3 cr,sp;

	mat outplane = mat(4,3);
	mat pmatrix = mat(3,3);
	mat rmatrix = mat(3,3);
	mat wm = mat(3,3);
	vec inputplane = vec(4);
	
	create_out_plane(outplane, deg_to_rad(120));

	inputplane	<< 0 << endr //x1 - upper left corner
				<< 0 << endr //y1 - upper left corner
				<< DEST_X << endr //x size
				<< DEST_Y << endr; //y size

	create_project_matrix(outplane, inputplane, pmatrix);
								//theta 		//phi
	create_rotate_matrix(deg_to_rad(90), deg_to_rad(90), rmatrix);
	wm =  rmatrix * pmatrix;

    wm.print();


    FILE *xymapfd = fopen("xymap.bin", "rb");
    if (xymapfd == NULL){
    	printf("can't open xymap\n");
    	exit(1);
    }

    FILE *bmapfd = fopen("bmap.bin", "rb");
    if (bmapfd==NULL){
    	printf("can't open bmap\n");
    	exit(1);
    }
// Sources
    FILE *frfd = fopen("./cube/front.rgba", "rb");
    if (frfd==NULL){
    	printf("can't open front\n");
    	exit(1);
    }

    FILE *leftfd = fopen("./cube/left.rgba", "rb");
    if (leftfd==NULL){
    	printf("can't open left\n");
    	exit(1);
    }

    FILE *rightfd = fopen("./cube/right.rgba", "rb");
    if (rightfd==NULL){
    	printf("can't open right\n");
    	exit(1);
    }

    FILE *backfd = fopen("./cube/back.rgba", "rb");
    if (backfd==NULL){
    	printf("can't open back\n");
    	exit(1);
    }

    FILE *topfd = fopen("./cube/top.rgba", "rb");
    if (topfd==NULL){
    	printf("can't open top\n");
    	exit(1);
    }

    FILE *bottomfd = fopen("./cube/bottom.rgba", "rb");
    if (bottomfd==NULL){
    	printf("can't open back\n");
    	exit(1);
    }


	FILE *planefd = fopen("plane.rgb", "wb+");
	if (planefd==NULL){
		printf("cant create output file\n");
		exit(1);
	}

	//fread(pano, 4, OUT_X*OUT_Y,panofd);
	fread(xymap, sizeof(struct int4), OUT_X*OUT_Y, xymapfd);
	fread(bmapg, sizeof(struct float4), OUT_Y*OUT_X, bmapfd);
	fread(&sdata[0], 4, 1200*1200, rightfd);
	fread(&sdata[1], 4, 1200*1200, frfd);
	fread(&sdata[2], 4, 1200*1200, leftfd);
	fread(&sdata[3], 4, 1200*1200, backfd);
	fread(&sdata[4], 4, 1200*1200, topfd);
	fread(&sdata[5], 4, 1200*1200, bottomfd);
	
	fflush(rightfd);
	fflush(frfd);
	fflush(leftfd);
	fflush(backfd);
	fflush(topfd);
	fflush(bottomfd);

	fclose(rightfd);
	fclose(frfd);
	fclose(leftfd);
	fclose(backfd);
	fclose(topfd);
	fclose(bottomfd);

	float nv_wm[9];
	float nv_invec[3], nv_outvec[3];

	nv_wm[0] = wm(0,0);
	nv_wm[1] = wm(0,1);
	nv_wm[2] = wm(0,2);
	nv_wm[3] = wm(1,0);
	nv_wm[4] = wm(1,1);
	nv_wm[5] = wm(1,2);
	nv_wm[6] = wm(2,0);
	nv_wm[7] = wm(2,1);
	nv_wm[8] = wm(2,2);

	float jff,iff;
	unsigned int *planept = &plane[0][0];
	for (jj=0;jj<DEST_Y;jj++){
		for(ii=0;ii<DEST_X;ii++){

			nv_invec[0] = (float)ii;
			nv_invec[1] = (float)jj;
			nv_invec[2] = (float)1;
			
			mul3x3x1(nv_wm, nv_invec, nv_outvec);

			cr.x = nv_outvec[0];
			cr.y = nv_outvec[1];
			cr.z = nv_outvec[2];

			cart_to_sphere(&cr, &sp);

			if (sp.y<0){
				sp.y *= -1;
				if (sp.x<datum::pi)
					sp.x +=datum::pi;
				else
					sp.x -=datum::pi;
			}else if (sp.y>datum::pi){
				sp.y = datum::pi - (sp.y - datum::pi);
				if (sp.x<datum::pi)
					sp.x +=datum::pi;
				else
					sp.x -=datum::pi;
			}

			if (sp.x<0){
				sp.x = (2*datum::pi) + sp.x;
			}else if (sp.x>(2*datum::pi))
				sp.x = sp.x - (2*datum::pi);

			jff = phi_to_j(sp.y);
			iff = theta_to_i(sp.x);

			unsigned int q1 = dotsmultiply(floor(jff), floor(iff));
			unsigned int q2 = dotsmultiply(ceil(jff), floor(iff));
			unsigned int q3 = dotsmultiply(floor(jff), ceil(iff));
			unsigned int q4 = dotsmultiply(ceil(jff), ceil(iff));

			*(planept + jj*DEST_X + ii) = interpolate(iff, jff, q1,q2,q3,q4 );
		}
	}

	fwrite(plane, DEST_Y*DEST_X,4,planefd);
	fflush(planefd);
	fflush(xymapfd);
	fflush(bmapfd);
	fclose(planefd);
	fclose(xymapfd);
	fclose(bmapfd);

}


