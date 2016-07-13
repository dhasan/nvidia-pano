
#include	<armadillo>
#include 	<fcntl.h>

#define 	OUT_X				(3600)
#define 	OUT_Y				(1800)

#define DEST_X 	(640)
#define DEST_Y	(640)

#define ANGLE_PHI	(-90)
#define ANGLE_THETA	(0)

#define FOV_X		(90)
#define FOV_Y		(90)

#define RADIUS		(1)
//#define RADIUS		(OUT_X/(2*datum::pi))


#define MAX(a,b) (a>b)?a:b
#define MIN(a,b) (a<b)?a:b

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



float phi_to_j(float phi){
	//float j = (float)((((OUT_Y-1)*phi) + (((OUT_Y-1)*datum::pi)/2))/datum::pi);
	float pi2 = datum::pi/2;
	float j = (float)(((OUT_Y-1)*phi  +  (OUT_Y-1)*pi2)/datum::pi);
	return j;
}

float theta_to_i(float theta){
	float i = (float)(((((OUT_X-1)*theta)) + ((OUT_X-1)*datum::pi))/(2*datum::pi));
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
	float x = sph->z*cos(sph->y)*cos(sph->x);
	float y = sph->z*sin(sph->y); 
	float z = sph->z*cos(sph->y)*sin(sph->x);

	cart->x = x;
	cart->y = y;
	cart->z = z;
}

void cart_to_sphere(float3 *cart, float3 *sph){
	float theta;
	float r = sqrt((cart->x*cart->x) + (cart->y*cart->y) + (cart->z*cart->z));
	if (cart->x==0){
		if (cart->z<0)
			theta = -1*datum::pi;
		else
			theta = datum::pi;
	}else
		theta = atan(cart->z/cart->x);



	float phi = acos(cart->y/r);

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

// struct dest {

// 	mat transp;
// 	float3 rota;
// 	float3 scale;

// 	mat world_matrix;

// };


unsigned int pano[OUT_Y][OUT_X];
unsigned int plane[DEST_Y][DEST_X];

void create_out_plane(mat& coord, float fov){


//	struct dest *dest;
	float3 cart_c;

//	dest = new struct dest;

	float3 cart_1,cart_2,cart_3,cart_4;
	float3 sph_t;

	float phi_c = deg_to_rad(ANGLE_PHI -90);
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
	sphere_to_cart(&sph_t, &cart_1);

	sph_t.x = theta_2;
	sph_t.y = phi_2;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	sphere_to_cart(&sph_t, &cart_2);

	sph_t.x = theta_3;
	sph_t.y = phi_3;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	sphere_to_cart(&sph_t, &cart_3);

	sph_t.x = theta_4;
	sph_t.y = phi_4;
	sph_t.z = RADIUS;//OUT_X/(2*datum::pi);
	sphere_to_cart(&sph_t, &cart_4);


	cart_c.x = (cart_1.x + cart_3.x)/2;
	cart_c.y = (cart_1.y + cart_3.y)/2;
	cart_c.z = (cart_1.z + cart_3.z)/2;

	printf("p1 x: %f, y: %f, z: %f\n",cart_1.x,cart_1.y,cart_1.z );
	printf("p2 x: %f, y: %f, z: %f\n",cart_2.x,cart_2.y,cart_2.z );
	printf("p3 x: %f, y: %f, z: %f\n",cart_3.x,cart_3.y,cart_3.z );
	printf("p4 x: %f, y: %f, z: %f\n",cart_4.x,cart_4.y,cart_4.z );
	printf("center x: %f y: %f z: %f\n",cart_c.x, cart_c.y, cart_c.z );
	coord(0,0)=cart_1.x;coord(0,1)=cart_1.y;coord(0,2)=cart_1.z;
	coord(1,0)=cart_2.x;coord(1,1)=cart_2.y;coord(1,2)=cart_2.z;
	coord(2,0)=cart_3.x;coord(2,1)=cart_3.y;coord(2,2)=cart_3.z;
	coord(3,0)=cart_4.x;coord(3,1)=cart_4.y;coord(3,2)=cart_4.z;

}

void create_project_matrix(mat& outplane, mat& inputplane, mat& pmatrix){
	int i,j;
	
    mat pa = mat(3,3);
   
    pa 	<< inputplane(0) 					<<	inputplane(1) 					<< 1 << endr
    	<< inputplane(0) + inputplane(2) 	<< 	inputplane(1) 					<< 1 << endr
    	<< inputplane(0) + inputplane(2) 	<< inputplane(1) + inputplane(3) 	<< 1 << endr;
  

    vec p1;
    p1  << outplane(0,0) << endr
    	<< outplane(1,0) << endr
    	<< outplane(2,0) << endr;
    	

    vec l1;
 
    l1 =inv(pa) * p1;

//    mat wm = mat(3,3);
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
    // float fan = deg_to_rad(0);
    // float fbn = deg_to_rad(-45); //phi
    fa 	<< cos(theta) << 0 << -sin(theta) << endr
    	<< 0 << 1 << 0 << endr
    	<< sin(theta) << 0 << cos(theta) << endr;

    fb << cos(phi) << -sin(phi) << 0 <<endr
    	<< sin(phi) << cos(phi) << 0 <<endr
    	<< 0 << 0 << 1 <<endr;

 	
 	rmatrix = fa * fb;
	//pmatrix =  fa * pmatrix;

	// wm = fb * pmatrix;
	// wm = fa * pmatrix;
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
	
	create_out_plane(outplane, deg_to_rad(90));

	inputplane	<< 0 << endr //x1 - upper left corner
				<< 0 << endr //y1 - upper left corner
				<< DEST_X << endr //x size
				<< DEST_Y << endr; //y size

	create_project_matrix(outplane, inputplane, pmatrix);
								//theta 		//phi
	create_rotate_matrix(deg_to_rad(135), deg_to_rad(45), rmatrix);
	wm =  rmatrix * pmatrix;

    wm.print();
   

	FILE *panofd = fopen("out3.raw","rb");
	if (panofd==NULL){
		printf("can't open pano file\n");
		//exit(1);
	}
	FILE *planefd = fopen("plane.rgb", "wb+");
	if (planefd==NULL){
		printf("cant create output file\n");
	}

	fread(pano, 4, OUT_X*OUT_Y,panofd);

	for (jj=0;jj<DEST_Y;jj++){
		for(ii=0;ii<DEST_X;ii++){

			invec 	<< ii << endr
					<< jj << endr
					<< 1 << endr;

			outvec = wm * invec;
			//ou.print();

			cr.x = outvec(0);
			cr.y = outvec(1);
			cr.z = outvec(2);
			cart_to_sphere(&cr, &sp);

			//////////
			j = (int)phi_to_j(sp.y-datum::pi/2);
		
			i = (int)theta_to_i(sp.x);
			if (i>OUT_X || i<0)
				printf("warn: i %d",i);
			if (j>OUT_Y || j<0)
				printf("warn: j %d",j);
		//	printf("jj: %d ii: %d j: %d i:: %d sp.x %f\n",jj,ii,j,i,sp.x );
			plane[jj][ii] = pano[j][i];

		}
	}

	fwrite(plane, DEST_Y*DEST_X,4,planefd);
	fflush(panofd);
	fflush(planefd);
	fclose(planefd);
	fclose(panofd);

}


