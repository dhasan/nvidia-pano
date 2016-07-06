
#include 	<iostream>
#include 	<armadillo>

#define 	OUT_X				3600
#define 	OUT_Y				2400

#define DEST_X 	640
#define DEST_Y	480

#define ANGLE_PHI	90
#define ANGLE_THETA	0

#define FOV_X		60
#define FOV_Y		40

using namespace arma;

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

struct float3 {
	float x;
	float y;
	float z;

};


unsigned int pano[OUT_Y][OUT_X];

float phi_to_j(float phi){
	float j = (float)((((OUT_Y-1)*phi) + (((OUT_Y-1)*datum::pi)/2))/datum::pi);
	return j;
}

float theta_to_i(float theta){
	float i = (float)(((((OUT_X-1)*theta)) + ((OUT_X-1)*datum::pi))/2*datum::pi);
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
	float x = cos(sph->y)*cos(sph->x);
	float y = sin(sph->y); 
	float z = cos(sph->y)*sin(sph->x);

	cart->x = x;
	cart->y = y;
	cart->z = z;
}

void cart_to_sphere(float3 *cart, float3 *sph){

	float r = sqrt(cart->x*cart->x + cart->y*cart->y + cart->z*cart->z);
	float theta = atan(cart->x/cart->z);
	float phi = acos(cart->y/r);

	sph->x = theta;
	sph->y = phi;
	sph->z = r;
}

float distance(float3 *p1, float3 *p2){

	float x = p2->x - p1->x;
	float y = p2->y - p1->y;
	float z = p2->z - p1->z;

	float r = sqrt(x*x + y*y + z*z);

	return r;
}


struct plane {


	float3 dots[4];

};

struct dest {

	mat transp;
	float3 rota;
	float3 scale;

	mat world_matrix;

};


int main(){
	int ii,jj;

	struct dest *dest;

	dest = new struct dest;

	float3 cart_1,cart_2,cart_3,cart_4;
	float3 sph_t;

	float phi_c = deg_to_rad(ANGLE_PHI);
	float theta_c = deg_to_rad(ANGLE_THETA);

	//point 1
	float phi_1 = deg_to_rad((float)ANGLE_PHI + (float)(FOV_Y/2));
	float theta_1 = deg_to_rad((float)ANGLE_THETA + (float)(FOV_X/2));

	//point 2
	float phi_2 = deg_to_rad((float)ANGLE_PHI + (float)(FOV_Y/2));
	float theta_2 = deg_to_rad((float)ANGLE_THETA - (float)(FOV_X/2));

	//point 3
	float phi_3 = deg_to_rad((float)ANGLE_PHI - (float)(FOV_Y/2));
	float theta_3 = deg_to_rad((float)ANGLE_THETA - (float)(FOV_X/2));	

	//point 4
	float phi_4 = deg_to_rad((float)ANGLE_PHI - (float)(FOV_Y/2));
	float theta_4 = deg_to_rad((float)ANGLE_THETA + (float)(FOV_X/2));	

	sph_t.x = theta_1;
	sph_t.y = phi_1;
	sph_t.z = 1;
	sphere_to_cart(&sph_t, &cart_1);

	sph_t.x = theta_2;
	sph_t.y = phi_2;
	sph_t.z = 1;
	sphere_to_cart(&sph_t, &cart_2);

	sph_t.x = theta_3;
	sph_t.y = phi_3;
	sph_t.z = 1;
	sphere_to_cart(&sph_t, &cart_3);

	sph_t.x = theta_4;
	sph_t.y = phi_4;
	sph_t.z = 1;
	sphere_to_cart(&sph_t, &cart_4);

	dest->world_matrix = mat(4,4);
	dest->world_matrix.eye();

	float sx = distance(&cart_1, &cart_2);
	float sy = distance(&cart_1, &cart_4);
	printf("sx: %f sy: %f\n",sx,sy );
	mat scale = mat(4,4);
	scale.eye();
	scale(0,0) = sx/DEST_X;
	scale(1,1) = sy/DEST_Y;
	dest->world_matrix = scale * dest->world_matrix;


	dest->transp = mat(4,4);
	dest->transp.eye();
	dest->transp(0,3) = -1*cart_1.x;
	dest->transp(1,3) = -1*cart_1.y;
	dest->transp(2,3) = -1*cart_1.z;

	

	dest->world_matrix = dest->transp * dest->world_matrix;
	printf("transp\n");
	dest->world_matrix.print();
	vec p1vec_x;

	p1vec_x << 	cart_2.x - cart_1.x << endr
			<<	cart_2.y - cart_1.y << endr
			<< 	cart_2.z - cart_1.z << endr;

	vec p1vec_y;
	p1vec_y << 	cart_4.x - cart_1.x << endr
			<<	cart_4.y - cart_1.y << endr
			<< 	cart_4.z - cart_1.z << endr;
	
	vec x_vec;
	x_vec << 1 << endr
		  << 0 << endr
		  << 0 << endr;

	vec y_vec;
	y_vec << 0 << endr
		  << 1 << endr
		  << 0 << endr;

	float ang_x = acos(norm_dot(p1vec_x, x_vec));

	float ang_y = acos(norm_dot(p1vec_y, y_vec));

	// ang_x *= -1;
	// ang_y *= -1;
	printf("ang_x: %f ang_y: %f\n", rad_to_deg(ang_x), rad_to_deg(ang_y));
	mat rot_x = mat(4,4);
	rot_x.eye();
	rot_x(1,1) = cos(ang_x);
	rot_x(1,2) = -sin(ang_x);
	rot_x(2,1) = sin(ang_x);
	rot_x(2,2) = cos(ang_x);
	dest->world_matrix = rot_x * dest->world_matrix;
	
	printf("rotx\n");
	dest->world_matrix.print();
	
	mat rot_y = mat(4,4);
	rot_y.eye();
	rot_y(0,0) = cos(ang_y);
	rot_y(0,2) = -sin(ang_y);
	rot_y(2,0) = sin(ang_y);
	rot_y(2,2) = cos(ang_y);
	dest->world_matrix = rot_y * dest->world_matrix;

	printf("roty\n");
	dest->world_matrix.print();

	
	dest->world_matrix.print();
	/////1. Convert to cartasian coordinate system
	/////2. Transponse origin to point 1
	/////3. Find angle between origin vectors x,y and vectors p12 and p13, rotate origin  
	/////4. Find distance from point 1 to point 2 -> x scale factor
	/////5. Find distance from point 1 to point 3 - > y scale factor

	

	vec input = colvec(4);
	vec output = colvec(4);
	float phimax=0, phimin=0, thetamin=0, thetamax=0;
	float3 ct;
	float3 st;
	float theta, phi;
#if 1
	for (jj=0;jj<DEST_Y;jj++){
		for(ii=0;ii<DEST_X;ii++){
			input 	<< (float)ii/(float)1 << endr
					<< (float)jj/(float)1 << endr
					<< 1 << endr
					<< 1 << endr;	
			output = dest->world_matrix * input;
			ct.x = output(0);
			ct.y = output(1);
			ct.z = output(2);
			cart_to_sphere(&ct, &st);
			//printf("theta: %f phi: %f\n", rad_to_deg(st.x), rad_to_deg(st.y) );
			theta = rad_to_deg(st.x);
			phi = rad_to_deg(st.y);
			if ((jj==0) && (ii==0)){
				phimax=phi;phimin=phi;
				thetamin=theta;thetamax=theta;
			}
			if (theta<thetamin) thetamin = theta;
			if (theta>thetamax) thetamax = theta;

			if (phi<phimin) phimin = phi;
			if (phi>phimax) phimax = phi;

			//output.print();
		
			//printf("x: %f y: %f\n", ct.x, ct.y );


		}
	}

	printf("thetamin %f, thetamax %f, phimin %f, phimax %f\n", thetamin, thetamax, phimin, phimax);
#endif

}