#pragma once
#include "wheels.hpp"
#include <GL\gl.h>
#include <ode/ode.h>
#include <plugins/physics.h>
#include <string>

// #define printf dWebotsConsolePrintf

#define NAME "Sphero"
#define SPHERE_R 0.5
#define WHEEL_R 0.03

class Sphere {
    dBodyID body_;
    dGeomID geo_;
    Wheels *wheels_;

  public:
    Sphere(Wheels *wheels) {

        if (_DEBUG)
            printf("Init Sphere\n");

        geo_ = dWebotsGetGeomFromDEF(NAME);
        body_ = dWebotsGetBodyFromDEF(NAME);
        wheels_ = wheels;

        if (_DEBUG)
            printf("%s %p body %p\n", NAME, geo_, body_);
    }

    const dReal *getPosition() const {
        return dBodyGetPosition(body_);
    }

    const dReal *getTorque() const {
        return dBodyGetTorque(body_);
    }

    void addForceRel(const dReal *t, const dReal *p) {
        dBodyAddForceAtRelPos(body_, t[0], t[1], t[2], p[0], p[1], p[2]);
    }

    void addTorque(const dReal *t) {
        dBodyAddRelTorque(body_, t[0], t[1], t[2]);
    }

    void setAngular(const dReal *t) {
        dBodySetAngularVel(body_, t[0], t[1], t[2]);
    }

    void printV(char *text, const dReal *p) {
        printf("%s %-10.3f %-10.3f %-10.3f\n", text, p[0], p[1], p[2]);
    }

    dReal getNorm(const dReal *w) {
        return sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    }

    void step() {
        dReal wf[3] = {0, 0, 0};

        auto wheels = wheels_->getON();
        for (int i = 0; i < N_WHEELS; i++) {
            if (wheels[i]) {
                auto w = wheels_->getWheelAngular(i);
                wf[0] += (w[0] * WHEEL_R) / SPHERE_R;
                wf[1] += (w[1] * WHEEL_R) / SPHERE_R;
                wf[2] += (w[2] * WHEEL_R) / SPHERE_R;
            }
        }
        setAngular(wf);
    }

    void printData() const {
        printf("Sphere\n");
        auto p = getPosition();
        printf("X %-10.3f %-10.3f %-10.3f\n", p[0], p[1], p[2]);
        auto t = getTorque();
        printf("T %-10.3f %-10.3f %-10.3f\n", t[0], t[1], t[2]);
        printf("\n");
    }
};