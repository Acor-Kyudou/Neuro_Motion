#include <mujoco/mujoco.h>
#include <mex.h>
#include <stdio.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mujoco_mex:invalidNumInputs", "One input required: predictions");
    }
    if (!mxIsDouble(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:mujoco_mex:notDouble", "Input must be double");
    }

    double *predictions = mxGetPr(prhs[0]);
    mwSize num_predictions = mxGetNumberOfElements(prhs[0]);

    // Load MuJoCo model
    const char *model_path = "C:\\Users\\USER\\Downloads\\sahan\\mujoco_menagerie-main\\mujoco_menagerie-main\\shadow_hand\\scene_right.xml";
    char error[1000] = "";
    mjModel *m = mj_loadXML(model_path, NULL, error, 1000);
    if (!m) {
        mexErrMsgIdAndTxt("MATLAB:mujoco_mex:modelLoadError", "Model load error: %s", error);
    }
    mjData *d = mj_makeData(m);

    // Keyframe positions
    double open_hand[20] = {0};
    double close_hand[20] = {0, 0, 0, 1.571, 0, 1.571, 0.17, 0, 1.571, 1.571, 0, 1.571, 1.571, 0, 1.571, 0, 0, 1.571, 1.2, 0};

    // Output joint positions
    plhs[0] = mxCreateDoubleMatrix(m->nq, num_predictions, mxREAL);
    double *qpos_out = mxGetPr(plhs[0]);

    // Simulate for each prediction
    for (mwSize i = 0; i < num_predictions; i++) {
        if (predictions[i] != 0 && predictions[i] != 1) {
            mj_deleteData(d);
            mj_deleteModel(m);
            mexErrMsgIdAndTxt("MATLAB:mujoco_mex:invalidPrediction", "Predictions must be 0 or 1");
        }

        // Set control based on prediction
        for (int j = 0; j < m->nu; j++) {
            d->ctrl[j] = (predictions[i] == 1) ? open_hand[j] : close_hand[j];
        }

        // Step simulation
        for (int step = 0; step < 100; step++) {
            mj_step(m, d);
        }

        // Store joint positions
        for (int j = 0; j < m->nq; j++) {
            qpos_out[j + i * m->nq] = d->qpos[j];
        }
    }

    mj_deleteData(d);
    mj_deleteModel(m);
}