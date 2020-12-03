#ifndef 	_CONSTANTS_H_
#define 	_CONSTANTS_H_

#include 	<iostream>
#include 	<armadillo>

#define 	OUT_X				3600
#define 	OUT_Y				1800


#define 	SOURCE_COUNT		6

#define		SOURCE_0_ID			0
#define 	SOURCE_0_NAME		"right"
#define 	SOURCE_0_DEVNAME	"./cube/right.rgba"
#define		SOURCE_0_X			1920
#define 	SOURCE_0_Y 			1080
#define		SOURCE_0_TRANSP_X	-0.5
#define		SOURCE_0_TRANSP_Y	-0.5
#define 	SOURCE_0_TRANSP_Z	-0.5
#define 	SOURCE_0_ROTATE_X	0
#define 	SOURCE_0_ROTATE_Y	arma::datum::pi/2
#define 	SOURCE_0_ROTATE_Z	arma::datum::pi/2

#define		SOURCE_1_ID			1
#define 	SOURCE_1_NAME		"front"
#define 	SOURCE_1_DEVNAME	"./cube/front.rgba"
#define		SOURCE_1_X			1920
#define 	SOURCE_1_Y 			1080
#define		SOURCE_1_TRANSP_X	0.5
#define		SOURCE_1_TRANSP_Y	-0.5
#define 	SOURCE_1_TRANSP_Z	0.5
#define 	SOURCE_1_ROTATE_X	arma::datum::pi
#define 	SOURCE_1_ROTATE_Y	0
#define 	SOURCE_1_ROTATE_Z	0

#define		SOURCE_2_ID			2
#define 	SOURCE_2_NAME		"left"
#define 	SOURCE_2_DEVNAME	"./cube/left.rgba"
#define		SOURCE_2_X			1920
#define 	SOURCE_2_Y 			1080
#define		SOURCE_2_TRANSP_X	0.5
#define		SOURCE_2_TRANSP_Y	-0.5
#define 	SOURCE_2_TRANSP_Z	0.5
#define 	SOURCE_2_ROTATE_X	arma::datum::pi
#define 	SOURCE_2_ROTATE_Y	0
#define 	SOURCE_2_ROTATE_Z	0

#define		SOURCE_3_ID			3
#define 	SOURCE_3_NAME		"back"
#define 	SOURCE_3_DEVNAME	"./cube/back.rgba"
#define		SOURCE_3_X			1920
#define 	SOURCE_3_Y 			1080
#define		SOURCE_3_TRANSP_X	-0.5
#define		SOURCE_3_TRANSP_Y	-0.5
#define 	SOURCE_3_TRANSP_Z	-0.5
#define 	SOURCE_3_ROTATE_X	arma::datum::pi
#define 	SOURCE_3_ROTATE_Y	0
#define 	SOURCE_3_ROTATE_Z	0


#define		SOURCE_4_ID			4
#define 	SOURCE_4_NAME		"top"
#define 	SOURCE_4_DEVNAME	"./cube/top.rgba"
#define		SOURCE_4_X			1920
#define 	SOURCE_4_Y 			1080
#define		SOURCE_4_TRANSP_X	0.5
#define		SOURCE_4_TRANSP_Y	-0.5
#define 	SOURCE_4_TRANSP_Z	0.5
#define 	SOURCE_4_ROTATE_X	(-arma::datum::pi/2)
#define 	SOURCE_4_ROTATE_Y	0
#define 	SOURCE_4_ROTATE_Z	0

#define		SOURCE_5_ID			5
#define 	SOURCE_5_NAME		"bottom"
#define 	SOURCE_5_DEVNAME	"./cube/bottom.rgba"
#define		SOURCE_5_X			1920
#define 	SOURCE_5_Y 			1080
#define		SOURCE_5_TRANSP_X	0.5
#define		SOURCE_5_TRANSP_Y	0.5
#define 	SOURCE_5_TRANSP_Z	-0.5
#define 	SOURCE_5_ROTATE_X	(arma::datum::pi/2)
#define 	SOURCE_5_ROTATE_Y	0
#define 	SOURCE_5_ROTATE_Z	0

#define		PLANE_COUNT			12

#define		PLANE_0_ID			0
#define		PLANE_0_SOURCEID	0
// 0.5,0.5,0.5 - top left(0,0,1)
#define 	PLANE_0_DOT_0_X		0.5
#define 	PLANE_0_DOT_0_Y		0.5
#define 	PLANE_0_DOT_0_Z		0.5
#define 	PLANE_0_SOURCE_DOT_0_X 	0
#define 	PLANE_0_SOURCE_DOT_0_Y 	0
//0.5,-0.5,0.5 - top right(1200,0,1)
#define 	PLANE_0_DOT_1_X		0.5
#define 	PLANE_0_DOT_1_Y		-0.5
#define 	PLANE_0_DOT_1_Z		0.5
#define 	PLANE_0_SOURCE_DOT_1_X 	1920
#define 	PLANE_0_SOURCE_DOT_1_Y 	0
//0.5,-0.5,-0.5 - down right(1200,1200,1)
#define 	PLANE_0_DOT_2_X		0.5
#define 	PLANE_0_DOT_2_Y		-0.5
#define 	PLANE_0_DOT_2_Z		-0.5
#define 	PLANE_0_SOURCE_DOT_2_X 	1920
#define 	PLANE_0_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_0_SCALE_X		1000
// #define 	PLANE_0_SCALE_Y		1000
// #define 	PLANE_0_OFFSET_X	100
// #define 	PLANE_0_OFFSET_Y	100

#define		PLANE_1_ID			1
#define		PLANE_1_SOURCEID	0
//0.5,0.5,0.5 - top left(0,0,1)
#define 	PLANE_1_DOT_0_X		0.5
#define 	PLANE_1_DOT_0_Y		0.5
#define 	PLANE_1_DOT_0_Z		0.5
#define 	PLANE_1_SOURCE_DOT_0_X 	0
#define 	PLANE_1_SOURCE_DOT_0_Y 	0
//0.5,0.5,-0.5 - down left(0,1200,1)
#define 	PLANE_1_DOT_1_X		0.5
#define 	PLANE_1_DOT_1_Y		0.5
#define 	PLANE_1_DOT_1_Z		-0.5
#define 	PLANE_1_SOURCE_DOT_1_X 	0
#define 	PLANE_1_SOURCE_DOT_1_Y 	1080
//0.5,-0.5,-0.5 - down right(1200,1200,1)
#define 	PLANE_1_DOT_2_X		0.5
#define 	PLANE_1_DOT_2_Y		-0.5
#define 	PLANE_1_DOT_2_Z		-0.5
#define 	PLANE_1_SOURCE_DOT_2_X 	1920
#define 	PLANE_1_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_1_SCALE_X		1000
// #define 	PLANE_1_SCALE_Y		1000
// #define 	PLANE_1_OFFSET_X	100
// #define 	PLANE_1_OFFSET_Y	100

#define		PLANE_2_ID			2
#define		PLANE_2_SOURCEID	1
//-0.5,0.5,0.5 - top left
#define 	PLANE_2_DOT_0_X		-0.5
#define 	PLANE_2_DOT_0_Y		0.5
#define 	PLANE_2_DOT_0_Z		0.5
#define 	PLANE_2_SOURCE_DOT_0_X 	0
#define 	PLANE_2_SOURCE_DOT_0_Y 	0
//0.5,0.5,0.5 - top right
#define 	PLANE_2_DOT_1_X		0.5
#define 	PLANE_2_DOT_1_Y		0.5
#define 	PLANE_2_DOT_1_Z		0.5
#define 	PLANE_2_SOURCE_DOT_1_X 	1920
#define 	PLANE_2_SOURCE_DOT_1_Y 	0
//0.5,0.5,-0.5 - down right
#define 	PLANE_2_DOT_2_X		0.5
#define 	PLANE_2_DOT_2_Y		0.5
#define 	PLANE_2_DOT_2_Z		-0.5
#define 	PLANE_2_SOURCE_DOT_2_X 	1920
#define 	PLANE_2_SOURCE_DOT_2_Y 	1080
// #define 	PLANE_2_SCALE_X		1000
// #define 	PLANE_2_SCALE_Y		1000
// #define 	PLANE_2_OFFSET_X	100
// #define 	PLANE_2_OFFSET_Y	100

#define		PLANE_3_ID			3
#define		PLANE_3_SOURCEID	1
//-0.5,0.5,0.5 - top left
#define 	PLANE_3_DOT_0_X		-0.5
#define 	PLANE_3_DOT_0_Y		0.5
#define 	PLANE_3_DOT_0_Z		0.5
#define 	PLANE_3_SOURCE_DOT_0_X 	0
#define 	PLANE_3_SOURCE_DOT_0_Y 	0
//-0.5,0.5,-0.5 - donw left
#define 	PLANE_3_DOT_1_X		-0.5
#define 	PLANE_3_DOT_1_Y		0.5
#define 	PLANE_3_DOT_1_Z		-0.5
#define 	PLANE_3_SOURCE_DOT_1_X 	0
#define 	PLANE_3_SOURCE_DOT_1_Y 	1080
//0.5,0.5,-0.5 - down right
#define 	PLANE_3_DOT_2_X		0.5
#define 	PLANE_3_DOT_2_Y		0.5
#define 	PLANE_3_DOT_2_Z		-0.5
#define 	PLANE_3_SOURCE_DOT_2_X 	1920
#define 	PLANE_3_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_3_SCALE_X		1000
// #define 	PLANE_3_SCALE_Y		1000
// #define 	PLANE_3_OFFSET_X	100
// #define 	PLANE_3_OFFSET_Y	100


#define		PLANE_4_ID			4
#define		PLANE_4_SOURCEID	2
//-0.5, -0.5, 0.5 - top left
#define 	PLANE_4_DOT_0_X		-0.5
#define 	PLANE_4_DOT_0_Y		-0.5
#define 	PLANE_4_DOT_0_Z		0.5
#define 	PLANE_4_SOURCE_DOT_0_X 	0
#define 	PLANE_4_SOURCE_DOT_0_Y 	0
//-0.5, 0.5, 0.5 - top right
#define 	PLANE_4_DOT_1_X		-0.5
#define 	PLANE_4_DOT_1_Y		0.5
#define 	PLANE_4_DOT_1_Z		0.5
#define 	PLANE_4_SOURCE_DOT_1_X 	1920
#define 	PLANE_4_SOURCE_DOT_1_Y 	0
//-0.5, 0.5, -0.5 -down right
#define 	PLANE_4_DOT_2_X		-0.5
#define 	PLANE_4_DOT_2_Y		0.5
#define 	PLANE_4_DOT_2_Z		-0.5
#define 	PLANE_4_SOURCE_DOT_2_X 	1920
#define 	PLANE_4_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_4_SCALE_X		1000
// #define 	PLANE_4_SCALE_Y		1000
// #define 	PLANE_4_OFFSET_X	100
// #define 	PLANE_4_OFFSET_Y	100

#define		PLANE_5_ID			5
#define		PLANE_5_SOURCEID	2
//-0.5, -0.5, 0.5 - top left
#define 	PLANE_5_DOT_0_X		-0.5
#define 	PLANE_5_DOT_0_Y		-0.5
#define 	PLANE_5_DOT_0_Z		0.5
#define 	PLANE_5_SOURCE_DOT_0_X 	0
#define 	PLANE_5_SOURCE_DOT_0_Y 	0
//-0.5, -0.5, -0.5 - down left
#define 	PLANE_5_DOT_1_X		-0.5
#define 	PLANE_5_DOT_1_Y		-0.5
#define 	PLANE_5_DOT_1_Z		-0.5
#define 	PLANE_5_SOURCE_DOT_1_X 	0
#define 	PLANE_5_SOURCE_DOT_1_Y 	1080
//-0.5, 0.5, -0.5 - down right
#define 	PLANE_5_DOT_2_X		-0.5
#define 	PLANE_5_DOT_2_Y		0.5
#define 	PLANE_5_DOT_2_Z		-0.5
#define 	PLANE_5_SOURCE_DOT_2_X 	1920
#define 	PLANE_5_SOURCE_DOT_2_Y 	1080
/*
#define 	PLANE_5_SCALE_X		1000
#define 	PLANE_5_SCALE_Y		1000
#define 	PLANE_5_OFFSET_X	100
#define 	PLANE_5_OFFSET_Y	100
*/

#define		PLANE_6_ID			6
#define		PLANE_6_SOURCEID	3
//0.5, -0.5, 0.5 - top left
#define 	PLANE_6_DOT_0_X		0.5
#define 	PLANE_6_DOT_0_Y		-0.5
#define 	PLANE_6_DOT_0_Z		0.5
#define 	PLANE_6_SOURCE_DOT_0_X 	0
#define 	PLANE_6_SOURCE_DOT_0_Y 	0
//-0.5, -0.5, 0.5 - top right
#define 	PLANE_6_DOT_1_X		-0.5
#define 	PLANE_6_DOT_1_Y		-0.5
#define 	PLANE_6_DOT_1_Z		0.5
#define 	PLANE_6_SOURCE_DOT_1_X 	1920
#define 	PLANE_6_SOURCE_DOT_1_Y 	0
//-0.5, -0.5, -0.5 - down right
#define 	PLANE_6_DOT_2_X		-0.5
#define 	PLANE_6_DOT_2_Y		-0.5
#define 	PLANE_6_DOT_2_Z		-0.5
#define 	PLANE_6_SOURCE_DOT_2_X 	1920
#define 	PLANE_6_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_6_SCALE_X		1000
// #define 	PLANE_6_SCALE_Y		1000
// #define 	PLANE_6_OFFSET_X	100
// #define 	PLANE_6_OFFSET_Y	100

#define		PLANE_7_ID			7
#define		PLANE_7_SOURCEID	3
//0.5, -0.5, 0.5 - top left
#define 	PLANE_7_DOT_0_X		0.5
#define 	PLANE_7_DOT_0_Y		-0.5
#define 	PLANE_7_DOT_0_Z		0.5
#define 	PLANE_7_SOURCE_DOT_0_X 	0
#define 	PLANE_7_SOURCE_DOT_0_Y 	0
//0.5, -0.5, -0.5 - down left
#define 	PLANE_7_DOT_1_X		0.5
#define 	PLANE_7_DOT_1_Y		-0.5
#define 	PLANE_7_DOT_1_Z		-0.5
#define 	PLANE_7_SOURCE_DOT_1_X 	0
#define 	PLANE_7_SOURCE_DOT_1_Y 	1080
//-0.5, -0.5, -0.5 - down right
#define 	PLANE_7_DOT_2_X		-0.5
#define 	PLANE_7_DOT_2_Y		-0.5
#define 	PLANE_7_DOT_2_Z		-0.5
#define 	PLANE_7_SOURCE_DOT_2_X 	1920
#define 	PLANE_7_SOURCE_DOT_2_Y 	1080
// #define 	PLANE_7_SCALE_X		1000
// #define 	PLANE_7_SCALE_Y		1000
// #define 	PLANE_7_OFFSET_X	100
// #define 	PLANE_7_OFFSET_Y	100

#define		PLANE_8_ID			8
#define		PLANE_8_SOURCEID	4
//-0.5, -0.5, 0.5 - top left
#define 	PLANE_8_DOT_0_X		-0.5
#define 	PLANE_8_DOT_0_Y		-0.5
#define 	PLANE_8_DOT_0_Z		0.5
#define 	PLANE_8_SOURCE_DOT_0_X 	0
#define 	PLANE_8_SOURCE_DOT_0_Y 	0
//0.5, -0.5, 0.5 - top right
#define 	PLANE_8_DOT_1_X		0.5
#define 	PLANE_8_DOT_1_Y		-0.5
#define 	PLANE_8_DOT_1_Z		0.5
#define 	PLANE_8_SOURCE_DOT_1_X 	1920
#define 	PLANE_8_SOURCE_DOT_1_Y 	0
//0.5, 0.5, 0.5 - down right
#define 	PLANE_8_DOT_2_X		0.5
#define 	PLANE_8_DOT_2_Y		0.5
#define 	PLANE_8_DOT_2_Z		0.5
#define 	PLANE_8_SOURCE_DOT_2_X 	1920
#define 	PLANE_8_SOURCE_DOT_2_Y 	1080
// #define 	PLANE_8_SCALE_X		1000
// #define 	PLANE_8_SCALE_Y		1000
// #define 	PLANE_8_OFFSET_X	100
// #define 	PLANE_8_OFFSET_Y	100

#define		PLANE_9_ID			9
#define		PLANE_9_SOURCEID	4
//-0.5, -0.5, 0.5 - top left
#define 	PLANE_9_DOT_0_X		-0.5
#define 	PLANE_9_DOT_0_Y		-0.5
#define 	PLANE_9_DOT_0_Z		0.5
#define 	PLANE_9_SOURCE_DOT_0_X 	0
#define 	PLANE_9_SOURCE_DOT_0_Y 	0
//-0.5, 0.5, 0.5 - down right
#define 	PLANE_9_DOT_1_X		-0.5
#define 	PLANE_9_DOT_1_Y		0.5
#define 	PLANE_9_DOT_1_Z		0.5
#define 	PLANE_9_SOURCE_DOT_1_X 	0
#define 	PLANE_9_SOURCE_DOT_1_Y 	1080
//0.5, 0.5, 0.5 - down left
#define 	PLANE_9_DOT_2_X		0.5
#define 	PLANE_9_DOT_2_Y		0.5
#define 	PLANE_9_DOT_2_Z		0.5
#define 	PLANE_9_SOURCE_DOT_2_X 	1920
#define 	PLANE_9_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_9_SCALE_X		1000
// #define 	PLANE_9_SCALE_Y		1000
// #define 	PLANE_9_OFFSET_X	100
// #define 	PLANE_9_OFFSET_Y	100

#define		PLANE_10_ID			10
#define		PLANE_10_SOURCEID	5
//-0.5, 0.5, -0.5 - top left
#define 	PLANE_10_DOT_0_X	-0.5
#define 	PLANE_10_DOT_0_Y	0.5
#define 	PLANE_10_DOT_0_Z	-0.5
#define 	PLANE_10_SOURCE_DOT_0_X 	0
#define 	PLANE_10_SOURCE_DOT_0_Y 	0
//0.5, 0.5, -0.5 - top right
#define 	PLANE_10_DOT_1_X	0.5
#define 	PLANE_10_DOT_1_Y	0.5
#define 	PLANE_10_DOT_1_Z	-0.5
#define 	PLANE_10_SOURCE_DOT_1_X 	1920
#define 	PLANE_10_SOURCE_DOT_1_Y 	0
//0.5, -0.5, -0.5 - down right
#define 	PLANE_10_DOT_2_X	0.5
#define 	PLANE_10_DOT_2_Y	-0.5
#define 	PLANE_10_DOT_2_Z	-0.5
#define 	PLANE_10_SOURCE_DOT_2_X 	1920
#define 	PLANE_10_SOURCE_DOT_2_Y 	1080

// #define 	PLANE_10_SCALE_X	1000
// #define 	PLANE_10_SCALE_Y	1000
// #define 	PLANE_10_OFFSET_X	100
// #define 	PLANE_10_OFFSET_Y	100

#define		PLANE_11_ID			11
#define		PLANE_11_SOURCEID	5
//-0.5, 0.5, -0.5 - top left
#define 	PLANE_11_DOT_0_X	-0.5
#define 	PLANE_11_DOT_0_Y	0.5
#define 	PLANE_11_DOT_0_Z	-0.5
#define 	PLANE_11_SOURCE_DOT_0_X 	0
#define 	PLANE_11_SOURCE_DOT_0_Y 	0
//-0.5, -0.5, -0.5 - down left
#define 	PLANE_11_DOT_1_X	-0.5
#define 	PLANE_11_DOT_1_Y	-0.5
#define 	PLANE_11_DOT_1_Z	-0.5
#define 	PLANE_11_SOURCE_DOT_1_X 	0
#define 	PLANE_11_SOURCE_DOT_1_Y 	1080
//0.5, -0.5, -0.5 - down right
#define 	PLANE_11_DOT_2_X	0.5
#define 	PLANE_11_DOT_2_Y	-0.5
#define 	PLANE_11_DOT_2_Z	-0.5
#define 	PLANE_11_SOURCE_DOT_2_X 	1920
#define 	PLANE_11_SOURCE_DOT_2_Y 	1080
// #define 	PLANE_11_SCALE_X	1000
// #define 	PLANE_11_SCALE_Y	1000
// #define 	PLANE_11_OFFSET_X	100
// #define 	PLANE_11_OFFSET_Y	100

#endif
