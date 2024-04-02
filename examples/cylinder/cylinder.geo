//+
SetFactory("OpenCASCADE");
//+
Point(1) = {-3.5, 0, 0, 1.0};
//+
Point(2) = {-0.5, 0, 0, 1.0};
//+
Point(3) = {0, 0.0, 0, 1.0};
//+
Point(4) = {0.5, -0, 0, 1.0};
//+
Point(5) = {5.0, 0, 0, 1.0};
//+
Point(6) = {5.0, 3.5, 0, 1.0};
//+
Point(7) = {-3.5, 3.5, 0, 1.0};
//+
Line(1) = {4, 5};
//+
Line(2) = {5, 6};
//+
Line(3) = {6, 7};
//+
Line(4) = {7, 1};
//+
Line(5) = {1, 2};
//+
Circle(6) = {2, 3, 4};
//+
//+
Curve Loop(1) = {6, 1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Field[1] = BoundaryLayer;
Field[1].AnisoMax = 1000;
Field[1].CurvesList = {6};
Field[1].PointsList = {2,4};
Field[1].Quads = 1;
Field[1].Ratio = 1.2;
Field[1].Thickness = 0.05;
Field[1].hwall_n = 0.005;


Field[2] = Distance;
Field[2].EdgesList = {6};

Field[3] = MathEval;
Field[3].F = "max(0.02, F2*0.01)";

BoundaryLayer Field = 1;
Background Field = 3;

//+
Extrude {0, 0, 0.1} {
  Surface{1}; Layers {1}; Recombine;
}
//+
Physical Surface("inlet", 444445) = {6};
//+
Physical Surface("outlet", 444446) = {4};
//+
Physical Surface("symmetry", 444447) = {7, 3};
//+
Physical Surface("sides", 444448) = {8, 1};
//+
Physical Surface("top", 444449) = {5};
//+
Physical Surface("cylinder", 444450) = {2};
//+
Physical Volume("internal", 444451) = {1};

Mesh 3;

Save "cylinder.su2";
