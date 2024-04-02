Merge "swept-wing.stp";



SetFactory("OpenCASCADE");


Dilate {{0, 0, 0}, {0.002, 0.002, 0.002}} {
    Volume{1}; 
  }
  



r = 4;

Point(101) = {0, -r, 0, 1.0};
Point(102) = {0, -r, -r, 1.0};
Point(103) = {r, -r, 0, 1.0};
Point(104) = {0, -r, r, 1.0};


Circle(101) = {102, 101, 103};
Circle(102) = {103, 101, 104};
Line(103) = {102, 104};

Curve Loop (101) = {101, 102, -103};
Plane Surface (101) = {101};

Extrude {0, 2.2*r, 0} {
    Surface{101}; 
}


BooleanDifference{ Volume{2}; Delete; }{ Volume{1}; Delete; }

Rotate {{1, 0, 0}, {0, 0, 0}, -Pi/2} {
  Volume{2}; 
}



Field[1] = Distance;
Field[1].SurfacesList = {6, 7 ,8, 9, 10, 11, 12};

Field[2] = MathEval;
Field[2].F = "max(0.005, F1*0.2)";

Background Field = 2;


Physical Surface("wing", 22) = {6, 7 ,8, 9, 10, 11, 12};
//+
Physical Surface("sides", 23) = {2};
//+
Physical Surface("farfield", 24) = {1, 3, 4, 5};
//+
Physical Volume("internal", 25) = {2};



//Mesh 3;

//Save "wing.su2";


