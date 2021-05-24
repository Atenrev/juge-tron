#include "sphere.hpp"
#include "wheels.hpp"
#include <assert.h>
#include <math.h>
#include <ode/ode.h>
#include <plugins/physics.h>
#include <stdlib.h>
#include <utility>
#include <windows.h>

static pthread_mutex_t mutex; // needed to run with multi-threaded version of ODE
static Sphere *sp;
static Wheels *wh;

void webots_physics_init() {
    pthread_mutex_init(&mutex, NULL);
    // AllocConsole();
    // freopen("CONOUT$", "w", stdout);
    wh = new Wheels();
    sp = new Sphere(wh);
    // sp->printData();
    // wh->printData();
}

void webots_physics_step() {
    // sp->printData();
    // wh->printData();
    sp->step();
}

int webots_physics_collide(dGeomID g1, dGeomID g2) {
    return 0;
}

void webots_physics_cleanup() {
    pthread_mutex_destroy(&mutex);
    dWebotsConsolePrintf("Clean up\n");
    // system("cls");
    delete sp;
    delete wh;
}