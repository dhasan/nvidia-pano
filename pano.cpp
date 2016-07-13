
//#include 	<iostream>
#include 	<armadillo>
#include <fcntl.h>

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

struct dest {

	mat transp;
	float3 rota;
	float3 scale;

	mat world_matrix;

};


#define FIX(n)					\
if (phi_##n<0){					\
		phi_##n*=-1;			\
		theta_##n+=datum::pi;	\
								\
	}else if (phi_##n>datum::pi){	\
		phi_##n = datum::pi - (phi_##n - datum::pi);	\
		theta_##n+=datum::pi;							\
	}													\
	if (theta_##n>(2*datum::pi))						\
			theta_##n -= 2*datum::pi; 					\
	else if (theta_##n<0)								\
		theta_##n = 2*datum::pi - theta_##n;			


unsigned int pano[OUT_Y][OUT_X];
unsigned int plane[DEST_Y][DEST_X];

int main(){
	int ii,jj;

	struct dest *dest;
	float3 cart_c;

	dest = new struct dest;

	float3 cart_1,cart_2,cart_3,cart_4;
	float3 sph_t;

	float phi_c = deg_to_rad(ANGLE_PHI -90);
	float theta_c = deg_to_rad(ANGLE_THETA);

	float fov = (float)deg_to_rad(FOV_X/2.0);

	// //point 1
	// float phi_1 = deg_to_rad((float)ANGLE_PHI - (float)(FOV_Y/2));
	// float theta_1 = deg_to_rad((float)ANGLE_THETA + (float)(FOV_X/2));
	float phi_1 =		phi_c 	- 	fov;
	float theta_1 = 	theta_c	+	fov;
//	FIX(1);

	// if (phi_1<0){
	// 	phi_1*=-1;
	// 	theta_1+=datum::pi;
		
	// }else if (phi_1>datum::pi){
	// 	phi_1 = datum::pi - (phi_1 - datum::pi);
	// 	theta_1+=datum::pi;
	// }

	// if (theta_1>(2*datum::pi))
	// 		theta_1 -= 2*datum::pi; 
	// else if (theta_1<0)
	// 	theta_1 = 2*datum::pi - theta_1;
	// //point 2
	// float phi_2 = deg_to_rad((float)ANGLE_PHI - (float)(FOV_Y/2));
	// float theta_2 = deg_to_rad((float)ANGLE_THETA - (float)(FOV_X/2));
	float phi_2 = phi_c - fov;
	float theta_2 = theta_c - fov;
//FIX(2);
	// //point 3
	// float phi_3 = deg_to_rad((float)ANGLE_PHI + (float)(FOV_Y/2));
	// float theta_3 = deg_to_rad((float)ANGLE_THETA - (float)(FOV_X/2));	
	float phi_3 = phi_c + fov;
	float theta_3 = theta_c - fov;
//	FIX(3);
	// //point 4
	// float phi_4 = deg_to_rad((float)ANGLE_PHI + (float)(FOV_Y/2));
	// float theta_4 = deg_to_rad((float)ANGLE_THETA + (float)(FOV_X/2));	
	float phi_4 = phi_c + fov;
	float theta_4 = theta_c + fov;
//	FIX(4);


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



	// cart_1.z = 0.5;
	// cart_2.z = -0.5;
	// cart_3.z = 0.5;
	// cart_4.z = 0.5;



	cart_c.x = (cart_1.x + cart_3.x)/2;
	cart_c.y = (cart_1.y + cart_3.y)/2;
	cart_c.z = (cart_1.z + cart_3.z)/2;

	printf("p1 x: %f, y: %f, z: %f\n",cart_1.x,cart_1.y,cart_1.z );
	printf("p2 x: %f, y: %f, z: %f\n",cart_2.x,cart_2.y,cart_2.z );
	printf("p3 x: %f, y: %f, z: %f\n",cart_3.x,cart_3.y,cart_3.z );
	printf("p4 x: %f, y: %f, z: %f\n",cart_4.x,cart_4.y,cart_4.z );
	printf("center x: %f y: %f z: %f\n",cart_c.x, cart_c.y, cart_c.z );
	vec p1vec_x;

	p1vec_x << 	cart_2.x - cart_1.x << endr
			<<	cart_2.y - cart_1.y << endr
			<< 	cart_2.z - cart_1.z << endr;

	vec p1vec_y;
	p1vec_y << 	cart_4.x - cart_1.x << endr
			<<	cart_4.y - cart_1.y << endr
			<< 	cart_4.z - cart_1.z << endr;
	

	vec normv = cross(p1vec_x, p1vec_y);
	float3 cartnorm,spnorm;
	cartnorm.x = normv(0);
	cartnorm.y = normv(1);
	cartnorm.z = normv(2);

	cart_to_sphere(&cartnorm, &spnorm);
	printf("nnnnnnnnnnnnnnnorm theta: %f phi %f\n", rad_to_deg(spnorm.x), rad_to_deg(spnorm.y) );

	vec x_vec,y_vec,z_vec;

	x_vec 	<< 1 << endr	
			<< 0 << endr
			<< 0 << endr;
	y_vec 	<< 0 << endr
			<< 1 << endr
			<< 0 << endr;

	z_vec 	<< 0 << endr
			<< 0 << endr
			<< 1 << endr;

	printf("p1 vecs: \n");
	p1vec_x.print();
	p1vec_y.print();

	vec nvec = cross(p1vec_x, p1vec_y);
	float d = -1*(nvec(0)*cart_1.x + nvec(1)*cart_1.y + nvec(2)*cart_1.z);

	printf("nvec: d: %f\n",d);
	nvec.print();

	dest->world_matrix = mat(4,4);
	dest->world_matrix.eye();

	
	// float ang_y1 = acos(norm_dot(p1vec_y, y_vec));
	// ry = mat(4,4);
	// ry.eye();
	// ry(0,0) = cos(ang_y1);
	// ry(0,2) = -sin(ang_y1);
	// ry(2,0) = sin(ang_y1);
	// ry(2,2) = cos(ang_y1);
	// dest->world_matrix = ry * dest->world_matrix;

	


		float sx = distance(&cart_1, &cart_2);
		float sy = distance(&cart_1, &cart_4);
		printf("sx: %f sy: %f\n", sx/DEST_X, sy/DEST_Y );
		mat scale = mat(4,4);
		scale.eye();
		scale(0,0) = (1*sx)/DEST_X;
		scale(1,1) = (1*sx)/DEST_Y;
		scale(2,2) = 1;
		dest->world_matrix = scale * dest->world_matrix;

	 

	float ang_x = deg_to_rad(ANGLE_PHI);// -1*acos(norm_dot(p1vec_y, y_vec));
	float ang_y = deg_to_rad(ANGLE_THETA-90);// -1*acos(norm_dot(normv, y_vec));
	float ang_z = 0;//datum::pi/4;//acos(norm_dot(normv, z_vec));
	printf("ang_x %f ang_y %f ang_z %f\n",ang_x, ang_y, ang_z );
	// ang_x*=-1;
	// ang_y*=-1;
	// ang_z*=-1;





	mat ry = mat(4,4);
	ry.eye();
	ry(0,0) = cos(ang_y);
	ry(0,2) = -sin(ang_y);
	ry(2,0) = sin(ang_y);
	ry(2,2) = cos(ang_y);
	dest->world_matrix = ry * dest->world_matrix;



	

	mat rx = mat(4,4);
	rx.eye();
	rx(1,1) = cos(ang_x);
	rx(1,2) = -sin(ang_x);
	rx(2,1) = sin(ang_x);
	rx(2,2) = cos(ang_x);
	dest->world_matrix = rx * dest->world_matrix;


	



	dest->transp = mat(4,4);
	dest->transp.eye();
	dest->transp(0,3) = 1*cart_c.x;
	dest->transp(1,3) = 1*cart_c.y;
	dest->transp(2,3) = 1*cart_c.z;
	dest->world_matrix = dest->transp * dest->world_matrix;

	// mat rz = mat(4,4);
	// rz.eye();
	// rz(0,0) = cos(datum::pi/2);
	// rz(0,1) = -sin(datum::pi/2);
	// rz(1,0) = sin(datum::pi/2);
	// rz(1,1) = cos(datum::pi/2);
	// dest->world_matrix = rz * dest->world_matrix;



	



dest->world_matrix.print();





	
	// dest->transp.eye();
	// dest->transp(0,3) = 0;
	// dest->transp(1,3) = -1*sy;
	// dest->transp(2,3) = 0;
	// dest->world_matrix = dest->transp * dest->world_matrix;

	 // mat ry = mat(4,4);
	// ry.eye();
	// ry(0,0) = cos(datum::pi);
	// ry(0,2) = -sin(datum::pi);
	// ry(2,0) = sin(datum::pi);
	// ry(2,2) = cos(datum::pi);
	// dest->world_matrix = ry * dest->world_matrix;


//	mat rx = mat(4,4);
	// rx.eye();
	// rx(1,1) = cos(datum::pi/2);
	// rx(1,2) = -sin(datum::pi/2);
	// rx(2,1) = sin(datum::pi/2);
	// rx(2,2) = cos(datum::pi/2);
	// dest->world_matrix = rx * dest->world_matrix;

	

	

	// 	mat rz = mat(4,4);
	// 	rz.eye();
	// 	rz(0,0) = cos(ang_z);
	// 	rz(0,1) = -sin(ang_z);
	// 	rz(1,0) = sin(ang_z);
	// 	rz(1,1) = cos(ang_z);
	// 	dest->world_matrix = rz * dest->world_matrix;


	

	

	


	// dest->transp.eye();
	// dest->transp(0,3) = 1*cart_1.x;
	// dest->transp(1,3) = 1*cart_1.y;
	// dest->transp(2,3) = 1*cart_1.z;
	// dest->world_matrix = dest->transp * dest->world_matrix;

	// printf("ang_x: %f ang_y: %f\n",rad_to_deg(ang_x), rad_to_deg(ang_y) );

	// printf("p1vecx:\n");
	// p1vec_x.print();
	// printf("p1vecy:\n");
	// p1vec_y.print();

	// dest->world_matrix = mat(4,4);
	// dest->world_matrix.eye();



	//float sx = distance(&cart_3, &cart_4);
	//float sy = distance(&cart_1, &cart_4);
	// printf("sx: %f sy: %f\n", (sx), (sy) );
	// mat scale = mat(4,4);
	// scale.eye();
	// scale(0,0) = sx/DEST_X;
	// scale(1,1) = sy/DEST_Y;
	// //scale(2,2) = sy/DEST_Y;
	// dest->world_matrix = scale * dest->world_matrix;	


	// mat rx = mat(4,4);
	// rx.eye();
	// rx(1,1) = cos(ang_x);
	// rx(1,2) = -sin(ang_x);
	// rx(2,1) = sin(ang_x);
	// rx(2,2) = cos(ang_x);


	// dest->world_matrix = rx * dest->world_matrix;

	// mat ry = mat(4,4);
	// ry.eye();
	// ry(0,0) = cos(ang_y);
	// ry(0,2) = -sin(ang_y);
	// ry(2,0) = sin(ang_y);
	// ry(2,2) = cos(ang_y);
	// dest->world_matrix = ry * dest->world_matrix;




	//  dest->transp = mat(4,4);
	// dest->transp.eye();
	// dest->transp(0,3) = -sx/2;
	// dest->transp(1,3) = -sy/2;
	// dest->transp(2,3) = 0;
	// dest->world_matrix = dest->transp * dest->world_matrix;

	// mat rx1 = mat(4,4);
	// rx1.eye();
	// rx1(1,1) = cos(datum::pi);
	// rx1(1,2) = -sin(datum::pi);
	// rx1(2,1) = sin(datum::pi);
	// rx1(2,2) = cos(datum::pi);

	// dest->world_matrix = rx1 * dest->world_matrix;

	 
	// dest->transp.eye();
	// dest->transp(0,3) = 0;
	// dest->transp(1,3) = 1*sy;
	// dest->transp(2,3) = 0;
	// dest->world_matrix = dest->transp * dest->world_matrix;

	// dest->transp.eye();
	// dest->transp(0,3) = 1*cart_4.x;
	// dest->transp(1,3) = 1*cart_4.y;
	// dest->transp(2,3) = 1*cart_4.z;
	// dest->world_matrix = dest->transp * dest->world_matrix;


	
	

	

// dest->transp.eye();
// 	dest->transp(0,3) = 0;
// 	dest->transp(1,3) = 0;
// 	dest->transp(2,3) = 0.75;
// 	dest->world_matrix = dest->transp * dest->world_matrix;

	
// 	dest->world_matrix.print();
	/////1. Convert to cartasian coordinate system
	/////2. Transponse origin to point 1
	/////3. Find angle between origin vectors x,y and vectors p12 and p13, rotate origin  
	/////4. Find distance from point 1 to point 2 -> x scale factor
	/////5. Find distance from point 1 to point 3 - > y scale factor


	// vec in2,in3;

	// in2 << 0 << endr
	// 	<< 0 << endr
	// 	<< 0 << endr
	// 	<< 1 << endr;

	// in3 << 640 << endr
	// 	<< 640 << endr
	// 	<< 0 << endr
	// 	<< 1 << endr;

	// vec out2 = dest->world_matrix * in2;
	// vec out3 = dest->world_matrix * in3;

	// out2.print();
	// printf("\n");
	// out3.print();

	// vec input = colvec(4);
	// vec output = colvec(4);
	// float phimax=0, phimin=0, thetamin=0, thetamax=0;
	// float3 ct;
	// float3 st;
	// float theta, phi;



	FILE *panofd = fopen("out3.raw","rb");
	if (panofd==NULL){
		printf("eeeeeee\n");
		//exit(1);
	}
	FILE *planefd = fopen("plane.rgb", "wb+");
	if (planefd==NULL){
		printf("iiiiiiiiiiiiiii\n");
	}

	fread(&pano[0][0], 4, OUT_X*OUT_Y,panofd);

	float phi_rad, theta_rad;

#if 1
	float x,y,z,r;
	float dx,dy,dz;
	float a,b,c,x1,y1,z1;
	float det,t,det0;
	float3 cr,sp;
	vec v;
	//for (jj=0,y=cart_4.y;y<(cart_1.y);y+=(sy/(float)DEST_Y),jj++){
	//	for(ii=0,x=cart_4.x;x<(cart_3.x);x+=(sx/(float)DEST_X),ii++){
	
	float start_x = cart_4.x;
	float start_y = cart_4.y;
	float end_x = cart_2.x;
	float end_y = cart_2.y;

	

	float maxx,maxy,minx,miny;

	maxx=MAX(cart_1.x,cart_2.x);
	maxx=MAX(maxx,cart_3.x);
	maxx=MAX(maxx,cart_4.x);

	maxy=MAX(cart_1.y,cart_2.y);
	maxy=MAX(maxy,cart_3.y);
	maxy=MAX(maxy,cart_4.y);


	minx=MIN(cart_1.x,cart_2.x);
	minx=MIN(minx,cart_3.x);
	minx=MIN(minx,cart_4.x);

	miny=MIN(cart_1.y,cart_2.y);
	miny=MIN(miny,cart_3.y);
	miny=MIN(miny,cart_4.y);

	start_x=minx;
	start_y=miny;
	end_x=maxx;
	end_y=maxy;

	float diffx = end_x - start_x;
	float diffy = end_y - start_y;
	float stepx = diffx/((float)DEST_X);
	float stepy = diffy/((float)DEST_Y);


	printf("maxx: %f maxy %f minx: %f miny %f\n",maxx, maxy, minx,miny );
	x= start_x;
	y=start_y;
	jj=0;
	ii=0;

	#if 0
	do{
		
		do{
	#if 0		
			if (nvec(2)==0)
				z = 0;
			else
				z = -1*((x*nvec(0))+(y*nvec(1))+d)/nvec(2);
		
			v 	<< x + (1*RADIUS*nvec(0)) << endr
				<< y + (1*RADIUS*nvec(1)) << endr
				<< z + (1*RADIUS*nvec(2)) << endr;


			// r = sqrt(x*x + y*y + z*z);
			// if (r<(OUT_X/(2*datum::pi))){
			// 	printf("inside\n");
			// }else
			// 	printf("outside\n");

			dx = v(0) - x;
			dy = v(1) - y;
			dz = v(2) - z;
			r = RADIUS;
			a = (dx*dx) + (dy*dy) + (dz*dz);
			b = (2*dx*x) + (2*dy*y) + (2*dz*z);
			c = (x*x) + (y*y) + (z*z) -(r*r);
			det0 = (b*b) - (4*a*c);
			if (det0<0){
				printf("wanr det<0\n");
				continue;

			}
			det = sqrt(det0);

			if (a==0)
				t=0;
			else
				t = ((-1*b)-det)/(2*a);
			x1 = x+(t*dx);
			y1 = y+(t*dy);
			z1 = z+(t*dz);
			cr.x = x1;cr.y=y1;cr.z=z1;
			cart_to_sphere(&cr, &sp);

			// printf("v0: %f v1: %f v2: %f\n",v(0),v(1),v(2) );
			// printf("x: %f y: %f z: %f\n",x,y,z );
			// printf("dx: %f dy: %f dz: %f\n",dx,dy,dz );
			// printf("a: %f b: %f c: %f\n",a,b,c );
			// printf("det0: %f t: %f\n",det0,t );
			// printf("x1: %f y1: %f z1: %f\n",x1,y1,z1 );
			// printf("phi: %f theta: %f\n",rad_to_deg(sp.y), rad_to_deg(sp.x));

			// if (sp.y>(datum::pi/2)){
			// 	sp.y = datum::pi/2 - (sp.y - (datum::pi/2));
			// 	if (sp.x>0)
			// 		sp.x-=datum::pi;
			// 	else
			// 		sp.x+=datum::pi;
			// }

			if (phi_to_j(sp.y)>2*OUT_Y || phi_to_j(sp.y)<0)
				printf("warn: j: %f\n",phi_to_j(sp.y));
			if (theta_to_i(sp.x)>OUT_X || theta_to_i(sp.x)<0)
				printf("warn: i: %f\n",theta_to_i(sp.x));

#endif

			//////////

			cr.x = x;
			cr.y = y;
			cr.z = z;
			cart_to_sphere(&cr, &sp);

			//////////

			plane[jj][ii] = pano[(int)phi_to_j(sp.y-datum::pi/2)][(int)theta_to_i(sp.x)];

			x+=stepx;
			ii++;

		}while(x>minx && x<maxx);
		ii=0;
		jj++;
		y+=stepy;
		x=start_x;
	}while(y>miny && y<maxy	);
	#endif
	vec ou,inv1,inv2,inv3,inv4;

	float ax = deg_to_rad(ANGLE_THETA+90);
	float bx = deg_to_rad(ANGLE_PHI);

	// //ax*=-1;
	mat axmat = mat(4,4);
	axmat.eye();
	axmat(0,0) = cos(ax);
	axmat(0,2) = -sin(ax);
	axmat(2,0) = sin(ax);
	axmat(2,2) = cos(ax);

	mat bxmat = mat(4,4);
	bxmat.eye();
	bxmat(1,1) = cos(bx);
	bxmat(1,2) = -sin(bx);
	bxmat(2,1) = sin(bx);
	bxmat(2,2) = cos(bx);
	axmat  = bxmat * axmat;
	int i,j;

	mat tsp = mat(4,4);

	inv1 << -DEST_X << endr
		 << DEST_Y << endr
		 << 1 << endr
		 << 1 << endr;
	inv2 << DEST_X << endr
		 << DEST_Y << endr
		 << 1 << endr
		 << 1 << endr;
	inv3 << DEST_X << endr
		 << -DEST_Y << endr
		 << 1 << endr
		 << 1 << endr;
	inv4 << -DEST_X << endr
		 << -DEST_Y << endr
		 << 1 << endr
		 << 1 << endr;

	ou = dest->world_matrix * inv1;
	printf("p1\n");
	ou.print();
	ou = dest->world_matrix * inv2;
	printf("p2\n");
	ou.print();
	ou = dest->world_matrix * inv3;
	printf("p3\n");
	ou.print();
	ou = dest->world_matrix * inv4;
	printf("p4\n");
	ou.print();


	inv1 << 0 << endr
		 << 0 << endr
		 << 0 << endr
		 << 1 << endr;
	ou = dest->world_matrix * inv1;
	printf("center\n");
	ou.print();


	vec ti;
	ti << -320 
		<<	320 
		<<	1 <<endr;
	vec to;
	to << 0.433013  
		<<	0.500000  
		<<	0.750000 << endr;

	mat inv5 = solve(ti,to);
	printf("----\n");
	inv5.print();
	printf("----\n");
	printf("world\n");
	dest->world_matrix.print();




	vec out1;
	out1 << cart_1.x << endr
		 << cart_1.y << endr
		 << cart_1.z << endr
		 << 1 << endr;
	
	vec out2;
	out2 << cart_2.x << endr
		 << cart_2.y << endr
		 << cart_2.z << endr
		 << 1 << endr;
	
	vec out3;
	out3 << cart_3.x << endr
		 << cart_3.y << endr
		 << cart_3.z << endr
		 << 1 << endr;
	
	vec out4;
	out4 << cart_4.x << endr
		 << cart_4.y << endr
		 << cart_4.z << endr
		 << 1 << endr;

	vec in1 = vec(4);
	in1  << 0 << endr
		 << 0 << endr
		 << 1 << endr
		 << 1 << endr;

	vec in2 =vec(4);
	in2  << DEST_X << endr
		 << 0 << endr
		 << 1 << endr
		 << 1 << endr;
		
	vec in3=vec(4);
	in3  << DEST_X << endr
		 << DEST_Y << endr
		 << 1 << endr
		 << 1 << endr;

	vec in4=vec(4);
	in4  << 0<< endr
		 << DEST_Y << endr
		 << 1 << endr
		 << 1 << endr;
    
    mat pa = mat(4,4);
   
    pa 	<< in1(0) << in1(1) << in1(2) << /*in1(3) << */endr
    	<< in2(0) << in2(1) << in2(2) << /*in2(3) <<*/ endr
    	<< in3(0) << in3(1) << in3(2) << /*in3(3) <<*/ endr;
    //	<< in4(0) << in4(1) << in4(2) << in4(3) << endr;

    vec p1;
    p1  << out1(0) << endr
    	<< out2(0) << endr
    	<< out3(0) << endr;
    	//<< out4(0) << endr;

    vec l1;//=vec(4);
    printf("pa:\n");
    pa.print();
  


    l1 =inv(pa) * p1;

    mat wm = mat(3,3);
    wm(0,0) = l1(0);
    wm(0,1) = l1(1);
    wm(0,2) = l1(2);
    //wm(0,4) = l1(3);

    ///////////////

     p1  << out1(1) << endr
    	<< out2(1) << endr
    	<< out3(1) << endr;
    //	<< out4(1) << endr;

    l1 = inv(pa) * p1;

    //wm = mat(4,4);
    wm(1,0) = l1(0);
    wm(1,1) = l1(1);
    wm(1,2) = l1(2);
    //wm(1,4) = l1(3);

    //////////////////

     ///////////////

     p1  << out1(2) << endr
    	<< out2(2) << endr
    	<< out3(2) << endr;
    //	<< out4(2) << endr;

    l1 = inv(pa)*p1;

    //wm = mat(4,4);
    wm(2,0) = l1(0);
    wm(2,1) = l1(1);
    wm(2,2) = l1(2);
    //wm(2,4) = l1(3);

    //  ///////////////

    //  p1  << out1(3) << endr
    // 	<< out2(3) << endr
    // 	<< out3(3) << endr;
    // 	//<< out4(3) << endr;

    // l1 = inv(pa) * p1;

    // //wm = mat(4,4);
    // wm(3,0) = l1(0);
    // wm(3,1) = l1(1);
    // wm(3,2) = l1(2);
    // wm(3,3) = l1(3);

    mat fa = mat(3,3);
    mat fb = mat(3,3);
    float fan = deg_to_rad(45);
    float fbn = deg_to_rad(-45);
    fa 	<< cos(fan) << 0 << -sin(fan) << endr
    	<< 0 << 1 << 0 << endr
    	<< sin(fan) << 0 << cos(fan) << endr;

    fb << cos(fbn) << -sin(fbn) << 0 <<endr
    	<< sin(fbn) << cos(fbn) << 0 <<endr
    	<< 0 << 0 << 1 <<endr;

 	
 	wm = fb * wm;
wm =  fa * wm;
    wm.print();
#if 1
	for (jj=0;jj<DEST_Y;jj++){
		for(ii=0;ii<DEST_X;ii++){




			inv1 << ii << endr
				<< jj << endr
				<<1 << endr;
		// 	//	<< 1 << endr;

		// 	// inv2 = axmat * inv1; 
		// 	// inv = bxmat * inv2;
		// 	if ((jj==(-DEST_Y/2)) && (ii==-(DEST_X/2)))
		// //	inv.print();
			
		// 	ou = dest->world_matrix * inv1;

		// 	//////////

			ou = wm * inv1;
			//ou.print();

			cr.x = ou(0);
			cr.y = ou(1);
			cr.z = ou(2);
			cart_to_sphere(&cr, &sp);

			//////////
			j = (int)phi_to_j(sp.y-datum::pi/2);
			// sp.x-=datum::pi;
			// if (sp.x<-datum::pi)
			// 	sp.x+=datum::pi;
			i = (int)theta_to_i(sp.x);
			if (i>OUT_X || i<0)
				printf("warn: i %d",i);
			if (j>OUT_Y || j<0)
				printf("warn: j %d",j);
		//	printf("jj: %d ii: %d j: %d i:: %d sp.x %f\n",jj,ii,j,i,sp.x );
			plane[jj][ii] = pano[j][i];

		}
	}
	#endif
	fwrite(plane, DEST_Y*DEST_X,4,planefd);
	fflush(panofd);
	fflush(planefd);
	fclose(planefd);
	fclose(panofd);

//	printf("thetamin %f, thetamax %f, phimin %f, phimax %f\n", thetamin, thetamax, phimin, phimax);
#endif

}


