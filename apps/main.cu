#include "render/window.cuh"

#include "fvmops.cuh"
#include "fvmfields.cuh"
#include "meshio.cuh"
#include "fvmfieldsio.cuh"

#include "models/euler.cuh"

#include <math.h>
#include <time.h>



class MyModel : public ModelBase {
public:

    MyModel(int argc, char** argv) : ModelBase(argc, argv) {}

    void run() {
        printf("Starting simulation\n");
        PhysicsEuler* euler = (PhysicsEuler*)this->physics[0];

        iter = 0;
        int run = 1;
        while (run) {
            printf("Iter : %d\n", iter);
            if (this->stepBefore()) run = 0;

            if (this->do_compute() | (iter == 0)) {
                if (euler->step(iter)) run = 0;
                printf("- residuals : %lf\n", euler->residuals());
            }

            if (this->stepAfter()) break;
            iter ++;
        }
    }

};




int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Error, no file provided.\nusage: graphite <filename>\n");
        return 1;
    }

    std::string filename(argv[1]);
    PhysicsEuler physics(filename);

    MyModel model(argc, argv);
    model.add_physics(&physics);

    model.setup();
    model.run();
    model.finish();


    return 0;
}


