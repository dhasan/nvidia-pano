
#include	<armadillo>
#include 	<fcntl.h>

#define 	OUT_X				(3600)
#define 	OUT_Y				(1800)

#define DEST_X 	(640)
#define DEST_Y	(640)

#define ANGLE_PHI	(0)
#define ANGLE_THETA	(0)

#define FOV_X		(90)
#define FOV_Y		(90)

//#define RADIUS		(1)
#define RADIUS		(OUT_X/(2*datum::pi))


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
	// float x = sph->z*cos(sph->y)*cos(sph->x);
	// float y = sph->z*sin(sph->y); 
	// float z = sph->z*cos(sph->y)*sin(sph->x);

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
	// if (cart->x==0){
	// 	if (cart->z<0)
	// 		theta = -1*datum::pi;
	// 	else
	// 		theta = datum::pi;
	// }else
	// 	theta = atan(cart->z/cart->x);


	// //float phi = atan(r/cart->y);
	// if (cart->x>0)
	// 	phi = acos(cart->y/r);
	// else 
	// 	phi = acos(cart->y/r)*-1;

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

// struct dest {

// 	mat transp;
// 	float3 rota;
// 	float3 scale;

// 	mat world_matrix;

// };


unsigned int pano[OUT_Y][OUT_X];
unsigned int plane[DEST_Y][DEST_X];
struct float4 bmap[OUT_Y][OUT_X];

void create_out_plane(mat& coord, float fov){


//	struct dest *dest;
	float3 cart_c;

//	dest = new struct dest;

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
	//coord(3,0)=cart_3.x;coord(3,1)=cart_3.y;coord(3,2)=cart_3.z;

	// 1 			2
	// 4 			3



	// 4 			1
	// 3 			2




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
    // fa 	<< cos(theta) << 0 << -sin(theta) << endr
    // 	<< 0 << 1 << 0 << endr
    // 	<< sin(theta) << 0 << cos(theta) << endr;

    // fb << cos(phi) << -sin(phi) << 0 <<endr
    // 	<< sin(phi) << cos(phi) << 0 <<endr
    // 	<< 0 << 0 << 1 <<endr;
    fb 	<< 1 		<< 0 			<< 0 			<< endr 
    	<< 0 		<< cos(phi) 	<< -sin(phi) 	<< endr
    	<< 0 		<< sin(phi)		<< cos(phi)		<< endr;

     fa << cos(theta) << -sin(theta) << 0 <<endr
    	<< sin(theta) << cos(theta) << 0 <<endr
    	<< 0 << 0 << 1 <<endr;

 	
 	rmatrix = fa * fb;
	//pmatrix =  fa * pmatrix;

	// wm = fb * pmatrix;
	// wm = fa * pmatrix;
}


unsigned int argb_interpolate(struct float4 *vec, unsigned int q1, unsigned int q2, unsigned int q3, unsigned int q4 ){
	unsigned int temp;
	static int s=0;
	unsigned char r1,g1,b1,a1;
	unsigned char r2,g2,b2,a2;
	unsigned char r3,g3,b3,a3;
	unsigned char r4,g4,b4,a4;
	float r,g,b,a;
	r1 = (q1 & 0x00FF0000)>>16;
	r2 = (q2 & 0x00FF0000)>>16;
	r3 = (q3 & 0x00FF0000)>>16;
	r4 = (q4 & 0x00FF0000)>>16;

	g1 = (q1 & 0x0000FF00)>>8;
	g2 = (q2 & 0x0000FF00)>>8;
	g3 = (q3 & 0x0000FF00)>>8;
	g4 = (q4 & 0x0000FF00)>>8;

	b1 = (q1 & 0x000000FF)>>0;
	b2 = (q2 & 0x000000FF)>>0;
	b3 = (q3 & 0x000000FF)>>0;
	b4 = (q4 & 0x000000FF)>>0;

	r = vec->x*(float)r1 + vec->y*(float)r2 + vec->z*(float)r3 + vec->w*(float)r4;
	g = vec->x*(float)g1 + vec->y*(float)g2 + vec->z*(float)g3 + vec->w*(float)g4;
	b = vec->x*(float)b1 + vec->y*(float)b2 + vec->z*(float)b3 + vec->w*(float)b4;
//	printf("r: %f g: %f b: %f\n",r,g,b );

	//temp = b(0)*q1 + b()
	temp =0x80000000;
	unsigned int rr = (unsigned int)r;
	unsigned int gg = (unsigned int)g;
	unsigned int bb = (unsigned int)b;

	//printf("%x %x %x\n", rr<<24,gg<<16, bb<<8 );
	//printf("%x %x %x %x\n",q1,q2,q3,q4 );

	temp |= ((rr<<16) | (gg<<8) | (bb<<0));
	//printf("%x\n",temp );
	s++;
	return temp;
}


unsigned int interpolate(float x, float y, unsigned int *data, unsigned int pstride){

	mat bicubic = mat(4,4);
	vec vals = vec(4);
	vec b= vec(4);
	float4 bmap;
	unsigned int val;

	if ((unsigned int)floor(x)!=(unsigned int)ceil(x) && (unsigned int)floor(y)!=(unsigned int)ceil(y)){
			bicubic << 1 << floor(x) << floor(y) << floor(x) * floor(y) << endr
					<< 1 << floor(x) << ceil(y) << floor(x) * ceil(y) << endr
					<< 1 << ceil(x) << floor(y) << ceil(x) * floor(y) << endr
					<< 1 << ceil(x) << ceil(y) << ceil(x) * ceil(y) << endr;
			
			vals 	<< 1 << endr
					<< x << endr
					<< y << endr
					<< x*y << endr;
						
			b = trans(inv(bicubic)) * vals;// * vals;// * vals;		
				// b.print();
				// printf("\n");
			bmap.x = b(0);
			bmap.y = b(1);
			bmap.z = b(2);
			bmap.w = b(3);
		}else{
			bmap.x = 1.0f;
			bmap.y = 0;
			bmap.z = 0;
			bmap.w = 0;
		}
		int x1 = (int)floor(x);
		int y1 = (int)floor(y);
		int x2 = (int)ceil(x);
		int y2 = (int)ceil(y);
		// printf("q1: %x", *(data +pstride*y1 + x1));
		// printf("q1: %x", *(data +pstride*y2 + x1));
		// printf("q1: %x", *(data +pstride*y1 + x2));
		// printf("q1: %x", *(data +pstride*y2 + x2));

		//plane[jj][ii] = argb_interpolate(&bmap[jj][ii], pano[y1][x1], pano[y2][x1], pano[y1][x2], pano[y2][x2]); 
		val =  argb_interpolate(&bmap, *(data +pstride*y1 + x1), *(data + pstride*y2 + x1), *(data + pstride*y1 +x2), *(data + (pstride*y2) + x2)); 
	//	printf("%x\n", val);
		return val;
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
	create_rotate_matrix(deg_to_rad(0), deg_to_rad(90), rmatrix);
	wm =  rmatrix * pmatrix;

    wm.print();
   

	FILE *panofd = fopen("e1s.rgba","rb");
	if (panofd==NULL){
		printf("can't open pano file\n");
		exit(1);
	}
	FILE *planefd = fopen("plane.rgb", "wb+");
	if (planefd==NULL){
		printf("cant create output file\n");
		exit(1);
	}

	fread(pano, 4, OUT_X*OUT_Y,panofd);
	 
	float jff,iff;
	for (jj=0;jj<DEST_Y;jj++){
		for(ii=0;ii<DEST_X;ii++){

			invec 	<< ii << endr
					<< jj << endr
					<< 1 << endr;

			outvec = wm * invec;


			cr.x = outvec(0);
			cr.y = outvec(1);
			cr.z = outvec(2);
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

			plane[jj][ii] = interpolate(iff, jff, &pano[0][0],OUT_X );
		}
	
	}

	fwrite(plane, DEST_Y*DEST_X,4,planefd);
	fflush(panofd);
	fflush(planefd);
	fclose(planefd);
	fclose(panofd);

}


