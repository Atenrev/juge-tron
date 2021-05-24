#pragma once
#include "util.hpp"
#include <assert.h>
#include <ode/ode.h>
#include <plugins/physics.h>
#include <string>
#define N_WHEELS 3
#define N_sWHEELS 12

class Wheels {
    dGeomID wheelGeo_[N_WHEELS];
    dBodyID smallWheelBody_[N_WHEELS][N_sWHEELS];

    dBodyID wheelBody_[N_WHEELS];
    dGeomID smallWheelGeo_[N_WHEELS][N_sWHEELS];
    float wheelVelocity_[N_WHEELS];

  public:
    Wheels() {
        static char name[64];
        for (int i = 0; i < N_WHEELS; i++) {
            wheelVelocity_[i] = 0;
            sprintf(name, "wheel%dSolid", i);
            wheelGeo_[i] = dWebotsGetGeomFromDEF(name);
            wheelBody_[i] = dWebotsGetBodyFromDEF(name);

            if (_DEBUG)
                printf("%s geo %p body %p\n", name, wheelGeo_[i], wheelBody_[i]);

            for (int j = 0; j < N_sWHEELS; j++) {
                sprintf(name, "sr%d%d", i, j);
                smallWheelGeo_[i][j] = dWebotsGetGeomFromDEF(name);
                smallWheelBody_[i][j] = dWebotsGetBodyFromDEF(name);

                if (_DEBUG)
                    printf("    %s geo %p body %p\n", name, smallWheelGeo_[i][j], smallWheelBody_[i][j]);
            }
        }
    }

    const dReal *getWheelPos(const int i) const {
        assert(i < N_WHEELS);
        return dBodyGetPosition(wheelBody_[i]);
    }

    const dReal *getSmallWheelPos(const int i, const int j) const {
        assert(i < N_WHEELS && j < N_sWHEELS);
        return dBodyGetPosition(smallWheelBody_[i][j]);
    }

    const dReal *getWheelTorque(const int i) const {
        assert(i < N_WHEELS);
        return dBodyGetTorque(wheelBody_[i]);
    }

    const dReal *getWheelAngular(const int i) const {
        assert(i < N_WHEELS);
        return dBodyGetAngularVel(wheelBody_[i]);
    }

    bool checkBigWheel(dGeomID g1) {
        for (int i = 0; i < N_WHEELS; i++)
            if (dAreGeomsSame(g1, wheelGeo_[i]))
                return true;
        return false;
    }

    void setAngular(const dReal *t1, const dReal *t2, const int i) {
        dReal t[3];
        for (int i = 0; i < N_WHEELS; i++) {
            t[i] = -(t1[i] + t2[i]);
        }
        dBodySetAngularVel(wheelBody_[i], t[0], t[1], t[2]);
    }

    const float *getON() {
        int dataSize;
        const int *data = (const int *)dWebotsReceive(&dataSize);
        if (dataSize > 0) {
            if (data[0] == VELOCITY_MSG) {
                // printf("Received velocity: \n");
                const float *vel = (const float *)&data[1];
                for (int i = 0; i < N_WHEELS; i++) {
                    wheelVelocity_[i] = vel[i];
                    // printf("%0.2f ", vel[i]);
                }
            }
        }
        return wheelVelocity_;
    }

    void printData(bool smallWheel = false) const {
        printf("Wheels\n");
        for (int i = 0; i < N_WHEELS; i++) {
            auto p = getWheelPos(i);
            auto t = getWheelTorque(i);
            printf("X %d %-10.3f %-10.3f %-10.3f\n", i, p[0], p[1], p[2]);
            printf("T %-10.3f %-10.3f %-10.3f\n", t[0], t[1], t[2]);
            if (smallWheel) {
                for (int j = 0; j < N_sWHEELS; j++) {
                    auto v = getSmallWheelPos(i, j);
                    printf("    %d %-10.3f %-10.3f %-10.3f\n", j, v[0], v[1], v[2]);
                }
            }
        }
        printf("\n");
    }
};