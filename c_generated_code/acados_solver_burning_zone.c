/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> // memcpy
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "burning_zone_model/burning_zone_model.h"


#include "burning_zone_constraints/burning_zone_constraints.h"
#include "burning_zone_cost/burning_zone_cost.h"



#include "acados_solver_burning_zone.h"

#define NX     BURNING_ZONE_NX
#define NZ     BURNING_ZONE_NZ
#define NU     BURNING_ZONE_NU
#define NP     BURNING_ZONE_NP
#define NP_GLOBAL     BURNING_ZONE_NP_GLOBAL
#define NY0    BURNING_ZONE_NY0
#define NY     BURNING_ZONE_NY
#define NYN    BURNING_ZONE_NYN

#define NBX    BURNING_ZONE_NBX
#define NBX0   BURNING_ZONE_NBX0
#define NBU    BURNING_ZONE_NBU
#define NG     BURNING_ZONE_NG
#define NBXN   BURNING_ZONE_NBXN
#define NGN    BURNING_ZONE_NGN

#define NH     BURNING_ZONE_NH
#define NHN    BURNING_ZONE_NHN
#define NH0    BURNING_ZONE_NH0
#define NPHI   BURNING_ZONE_NPHI
#define NPHIN  BURNING_ZONE_NPHIN
#define NPHI0  BURNING_ZONE_NPHI0
#define NR     BURNING_ZONE_NR

#define NS     BURNING_ZONE_NS
#define NS0    BURNING_ZONE_NS0
#define NSN    BURNING_ZONE_NSN

#define NSBX   BURNING_ZONE_NSBX
#define NSBU   BURNING_ZONE_NSBU
#define NSH0   BURNING_ZONE_NSH0
#define NSH    BURNING_ZONE_NSH
#define NSHN   BURNING_ZONE_NSHN
#define NSG    BURNING_ZONE_NSG
#define NSPHI0 BURNING_ZONE_NSPHI0
#define NSPHI  BURNING_ZONE_NSPHI
#define NSPHIN BURNING_ZONE_NSPHIN
#define NSGN   BURNING_ZONE_NSGN
#define NSBXN  BURNING_ZONE_NSBXN





// ** solver data **

burning_zone_solver_capsule * burning_zone_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(burning_zone_solver_capsule));
    burning_zone_solver_capsule *capsule = (burning_zone_solver_capsule *) capsule_mem;

    return capsule;
}


int burning_zone_acados_free_capsule(burning_zone_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int burning_zone_acados_create(burning_zone_solver_capsule* capsule)
{
    int N_shooting_intervals = BURNING_ZONE_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return burning_zone_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int burning_zone_acados_update_time_steps(burning_zone_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "burning_zone_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for burning_zone_acados_create: step 1
 */
void burning_zone_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = EXTERNAL;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = EXTERNAL;

    nlp_solver_plan->nlp_cost[N] = EXTERNAL;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = IRK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* burning_zone_acados_create_setup_dimensions(burning_zone_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 15;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for burning_zone_acados_create: step 3
 */
void burning_zone_acados_create_setup_functions(burning_zone_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        // constraints.constr_type == "BGH" and dims.nh > 0
        capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac[i], burning_zone_constr_h_fun_jac_uxt_zt);
        }
        capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun[i], burning_zone_constr_h_fun);
        }
    
        // external cost
        MAP_CASADI_FNC(ext_cost_0_fun, burning_zone_cost_ext_cost_0_fun);
        MAP_CASADI_FNC(ext_cost_0_fun_jac, burning_zone_cost_ext_cost_0_fun_jac);
        MAP_CASADI_FNC(ext_cost_0_fun_jac_hess, burning_zone_cost_ext_cost_0_fun_jac_hess);



    
        // implicit dae
        capsule->impl_dae_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(impl_dae_fun[i], burning_zone_impl_dae_fun);
        }

        capsule->impl_dae_fun_jac_x_xdot_z = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(impl_dae_fun_jac_x_xdot_z[i], burning_zone_impl_dae_fun_jac_x_xdot_z);
        }

        capsule->impl_dae_jac_x_xdot_u_z = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(impl_dae_jac_x_xdot_u_z[i], burning_zone_impl_dae_jac_x_xdot_u_z);
        }

        
    
        // external cost
        capsule->ext_cost_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun[i], burning_zone_cost_ext_cost_fun);
        }

        capsule->ext_cost_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun_jac[i], burning_zone_cost_ext_cost_fun_jac);
        }

        capsule->ext_cost_fun_jac_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun_jac_hess[i], burning_zone_cost_ext_cost_fun_jac_hess);
        }

        

        
    } // N > 0
    // external cost - function
    MAP_CASADI_FNC(ext_cost_e_fun, burning_zone_cost_ext_cost_e_fun);

    // external cost - jacobian
    MAP_CASADI_FNC(ext_cost_e_fun_jac, burning_zone_cost_ext_cost_e_fun_jac);

    // external cost - hessian
    MAP_CASADI_FNC(ext_cost_e_fun_jac_hess, burning_zone_cost_ext_cost_e_fun_jac_hess);

    // external cost - jacobian wrt params
    

    

#undef MAP_CASADI_FNC
}


/**
 * Internal function for burning_zone_acados_create: step 5
 */
void burning_zone_acados_create_set_default_parameters(burning_zone_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;

    // initialize parameters to initial value
    
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        burning_zone_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for burning_zone_acados_create: step 5
 */
void burning_zone_acados_create_setup_nlp_in_numerical_values(burning_zone_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        burning_zone_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 5;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 5;
        cost_scaling[1] = 5;
        cost_scaling[2] = 5;
        cost_scaling[3] = 5;
        cost_scaling[4] = 5;
        cost_scaling[5] = 5;
        cost_scaling[6] = 5;
        cost_scaling[7] = 5;
        cost_scaling[8] = 5;
        cost_scaling[9] = 5;
        cost_scaling[10] = 5;
        cost_scaling[11] = 5;
        cost_scaling[12] = 5;
        cost_scaling[13] = 5;
        cost_scaling[14] = 5;
        cost_scaling[15] = 5;
        cost_scaling[16] = 5;
        cost_scaling[17] = 5;
        cost_scaling[18] = 5;
        cost_scaling[19] = 5;
        cost_scaling[20] = 5;
        cost_scaling[21] = 5;
        cost_scaling[22] = 5;
        cost_scaling[23] = 5;
        cost_scaling[24] = 5;
        cost_scaling[25] = 5;
        cost_scaling[26] = 5;
        cost_scaling[27] = 5;
        cost_scaling[28] = 5;
        cost_scaling[29] = 5;
        cost_scaling[30] = 5;
        cost_scaling[31] = 5;
        cost_scaling[32] = 5;
        cost_scaling[33] = 5;
        cost_scaling[34] = 5;
        cost_scaling[35] = 5;
        cost_scaling[36] = 5;
        cost_scaling[37] = 5;
        cost_scaling[38] = 5;
        cost_scaling[39] = 5;
        cost_scaling[40] = 5;
        cost_scaling[41] = 5;
        cost_scaling[42] = 5;
        cost_scaling[43] = 5;
        cost_scaling[44] = 5;
        cost_scaling[45] = 5;
        cost_scaling[46] = 5;
        cost_scaling[47] = 5;
        cost_scaling[48] = 5;
        cost_scaling[49] = 5;
        cost_scaling[50] = 5;
        cost_scaling[51] = 5;
        cost_scaling[52] = 5;
        cost_scaling[53] = 5;
        cost_scaling[54] = 5;
        cost_scaling[55] = 5;
        cost_scaling[56] = 5;
        cost_scaling[57] = 5;
        cost_scaling[58] = 5;
        cost_scaling[59] = 5;
        cost_scaling[60] = 5;
        cost_scaling[61] = 5;
        cost_scaling[62] = 5;
        cost_scaling[63] = 5;
        cost_scaling[64] = 5;
        cost_scaling[65] = 5;
        cost_scaling[66] = 5;
        cost_scaling[67] = 5;
        cost_scaling[68] = 5;
        cost_scaling[69] = 5;
        cost_scaling[70] = 5;
        cost_scaling[71] = 5;
        cost_scaling[72] = 5;
        cost_scaling[73] = 5;
        cost_scaling[74] = 5;
        cost_scaling[75] = 5;
        cost_scaling[76] = 5;
        cost_scaling[77] = 5;
        cost_scaling[78] = 5;
        cost_scaling[79] = 5;
        cost_scaling[80] = 5;
        cost_scaling[81] = 5;
        cost_scaling[82] = 5;
        cost_scaling[83] = 5;
        cost_scaling[84] = 5;
        cost_scaling[85] = 5;
        cost_scaling[86] = 5;
        cost_scaling[87] = 5;
        cost_scaling[88] = 5;
        cost_scaling[89] = 5;
        cost_scaling[90] = 5;
        cost_scaling[91] = 5;
        cost_scaling[92] = 5;
        cost_scaling[93] = 5;
        cost_scaling[94] = 5;
        cost_scaling[95] = 5;
        cost_scaling[96] = 5;
        cost_scaling[97] = 5;
        cost_scaling[98] = 5;
        cost_scaling[99] = 5;
        cost_scaling[100] = 5;
        cost_scaling[101] = 5;
        cost_scaling[102] = 5;
        cost_scaling[103] = 5;
        cost_scaling[104] = 5;
        cost_scaling[105] = 5;
        cost_scaling[106] = 5;
        cost_scaling[107] = 5;
        cost_scaling[108] = 5;
        cost_scaling[109] = 5;
        cost_scaling[110] = 5;
        cost_scaling[111] = 5;
        cost_scaling[112] = 5;
        cost_scaling[113] = 5;
        cost_scaling[114] = 5;
        cost_scaling[115] = 5;
        cost_scaling[116] = 5;
        cost_scaling[117] = 5;
        cost_scaling[118] = 5;
        cost_scaling[119] = 5;
        cost_scaling[120] = 5;
        cost_scaling[121] = 5;
        cost_scaling[122] = 5;
        cost_scaling[123] = 5;
        cost_scaling[124] = 5;
        cost_scaling[125] = 5;
        cost_scaling[126] = 5;
        cost_scaling[127] = 5;
        cost_scaling[128] = 5;
        cost_scaling[129] = 5;
        cost_scaling[130] = 5;
        cost_scaling[131] = 5;
        cost_scaling[132] = 5;
        cost_scaling[133] = 5;
        cost_scaling[134] = 5;
        cost_scaling[135] = 5;
        cost_scaling[136] = 5;
        cost_scaling[137] = 5;
        cost_scaling[138] = 5;
        cost_scaling[139] = 5;
        cost_scaling[140] = 5;
        cost_scaling[141] = 5;
        cost_scaling[142] = 5;
        cost_scaling[143] = 5;
        cost_scaling[144] = 5;
        cost_scaling[145] = 5;
        cost_scaling[146] = 5;
        cost_scaling[147] = 5;
        cost_scaling[148] = 5;
        cost_scaling[149] = 5;
        cost_scaling[150] = 5;
        cost_scaling[151] = 5;
        cost_scaling[152] = 5;
        cost_scaling[153] = 5;
        cost_scaling[154] = 5;
        cost_scaling[155] = 5;
        cost_scaling[156] = 5;
        cost_scaling[157] = 5;
        cost_scaling[158] = 5;
        cost_scaling[159] = 5;
        cost_scaling[160] = 5;
        cost_scaling[161] = 5;
        cost_scaling[162] = 5;
        cost_scaling[163] = 5;
        cost_scaling[164] = 5;
        cost_scaling[165] = 5;
        cost_scaling[166] = 5;
        cost_scaling[167] = 5;
        cost_scaling[168] = 5;
        cost_scaling[169] = 5;
        cost_scaling[170] = 5;
        cost_scaling[171] = 5;
        cost_scaling[172] = 5;
        cost_scaling[173] = 5;
        cost_scaling[174] = 5;
        cost_scaling[175] = 5;
        cost_scaling[176] = 5;
        cost_scaling[177] = 5;
        cost_scaling[178] = 5;
        cost_scaling[179] = 5;
        cost_scaling[180] = 5;
        cost_scaling[181] = 5;
        cost_scaling[182] = 5;
        cost_scaling[183] = 5;
        cost_scaling[184] = 5;
        cost_scaling[185] = 5;
        cost_scaling[186] = 5;
        cost_scaling[187] = 5;
        cost_scaling[188] = 5;
        cost_scaling[189] = 5;
        cost_scaling[190] = 5;
        cost_scaling[191] = 5;
        cost_scaling[192] = 5;
        cost_scaling[193] = 5;
        cost_scaling[194] = 5;
        cost_scaling[195] = 5;
        cost_scaling[196] = 5;
        cost_scaling[197] = 5;
        cost_scaling[198] = 5;
        cost_scaling[199] = 5;
        cost_scaling[200] = 5;
        cost_scaling[201] = 5;
        cost_scaling[202] = 5;
        cost_scaling[203] = 5;
        cost_scaling[204] = 5;
        cost_scaling[205] = 5;
        cost_scaling[206] = 5;
        cost_scaling[207] = 5;
        cost_scaling[208] = 5;
        cost_scaling[209] = 5;
        cost_scaling[210] = 5;
        cost_scaling[211] = 5;
        cost_scaling[212] = 5;
        cost_scaling[213] = 5;
        cost_scaling[214] = 5;
        cost_scaling[215] = 5;
        cost_scaling[216] = 5;
        cost_scaling[217] = 5;
        cost_scaling[218] = 5;
        cost_scaling[219] = 5;
        cost_scaling[220] = 5;
        cost_scaling[221] = 5;
        cost_scaling[222] = 5;
        cost_scaling[223] = 5;
        cost_scaling[224] = 5;
        cost_scaling[225] = 5;
        cost_scaling[226] = 5;
        cost_scaling[227] = 5;
        cost_scaling[228] = 5;
        cost_scaling[229] = 5;
        cost_scaling[230] = 5;
        cost_scaling[231] = 5;
        cost_scaling[232] = 5;
        cost_scaling[233] = 5;
        cost_scaling[234] = 5;
        cost_scaling[235] = 5;
        cost_scaling[236] = 5;
        cost_scaling[237] = 5;
        cost_scaling[238] = 5;
        cost_scaling[239] = 5;
        cost_scaling[240] = 5;
        cost_scaling[241] = 5;
        cost_scaling[242] = 5;
        cost_scaling[243] = 5;
        cost_scaling[244] = 5;
        cost_scaling[245] = 5;
        cost_scaling[246] = 5;
        cost_scaling[247] = 5;
        cost_scaling[248] = 5;
        cost_scaling[249] = 5;
        cost_scaling[250] = 5;
        cost_scaling[251] = 5;
        cost_scaling[252] = 5;
        cost_scaling[253] = 5;
        cost_scaling[254] = 5;
        cost_scaling[255] = 5;
        cost_scaling[256] = 5;
        cost_scaling[257] = 5;
        cost_scaling[258] = 5;
        cost_scaling[259] = 5;
        cost_scaling[260] = 5;
        cost_scaling[261] = 5;
        cost_scaling[262] = 5;
        cost_scaling[263] = 5;
        cost_scaling[264] = 5;
        cost_scaling[265] = 5;
        cost_scaling[266] = 5;
        cost_scaling[267] = 5;
        cost_scaling[268] = 5;
        cost_scaling[269] = 5;
        cost_scaling[270] = 5;
        cost_scaling[271] = 5;
        cost_scaling[272] = 5;
        cost_scaling[273] = 5;
        cost_scaling[274] = 5;
        cost_scaling[275] = 5;
        cost_scaling[276] = 5;
        cost_scaling[277] = 5;
        cost_scaling[278] = 5;
        cost_scaling[279] = 5;
        cost_scaling[280] = 5;
        cost_scaling[281] = 5;
        cost_scaling[282] = 5;
        cost_scaling[283] = 5;
        cost_scaling[284] = 5;
        cost_scaling[285] = 5;
        cost_scaling[286] = 5;
        cost_scaling[287] = 5;
        cost_scaling[288] = 5;
        cost_scaling[289] = 5;
        cost_scaling[290] = 5;
        cost_scaling[291] = 5;
        cost_scaling[292] = 5;
        cost_scaling[293] = 5;
        cost_scaling[294] = 5;
        cost_scaling[295] = 5;
        cost_scaling[296] = 5;
        cost_scaling[297] = 5;
        cost_scaling[298] = 5;
        cost_scaling[299] = 5;
        cost_scaling[300] = 5;
        cost_scaling[301] = 5;
        cost_scaling[302] = 5;
        cost_scaling[303] = 5;
        cost_scaling[304] = 5;
        cost_scaling[305] = 5;
        cost_scaling[306] = 5;
        cost_scaling[307] = 5;
        cost_scaling[308] = 5;
        cost_scaling[309] = 5;
        cost_scaling[310] = 5;
        cost_scaling[311] = 5;
        cost_scaling[312] = 5;
        cost_scaling[313] = 5;
        cost_scaling[314] = 5;
        cost_scaling[315] = 5;
        cost_scaling[316] = 5;
        cost_scaling[317] = 5;
        cost_scaling[318] = 5;
        cost_scaling[319] = 5;
        cost_scaling[320] = 5;
        cost_scaling[321] = 5;
        cost_scaling[322] = 5;
        cost_scaling[323] = 5;
        cost_scaling[324] = 5;
        cost_scaling[325] = 5;
        cost_scaling[326] = 5;
        cost_scaling[327] = 5;
        cost_scaling[328] = 5;
        cost_scaling[329] = 5;
        cost_scaling[330] = 5;
        cost_scaling[331] = 5;
        cost_scaling[332] = 5;
        cost_scaling[333] = 5;
        cost_scaling[334] = 5;
        cost_scaling[335] = 5;
        cost_scaling[336] = 5;
        cost_scaling[337] = 5;
        cost_scaling[338] = 5;
        cost_scaling[339] = 5;
        cost_scaling[340] = 5;
        cost_scaling[341] = 5;
        cost_scaling[342] = 5;
        cost_scaling[343] = 5;
        cost_scaling[344] = 5;
        cost_scaling[345] = 5;
        cost_scaling[346] = 5;
        cost_scaling[347] = 5;
        cost_scaling[348] = 5;
        cost_scaling[349] = 5;
        cost_scaling[350] = 5;
        cost_scaling[351] = 5;
        cost_scaling[352] = 5;
        cost_scaling[353] = 5;
        cost_scaling[354] = 5;
        cost_scaling[355] = 5;
        cost_scaling[356] = 5;
        cost_scaling[357] = 5;
        cost_scaling[358] = 5;
        cost_scaling[359] = 5;
        cost_scaling[360] = 5;
        cost_scaling[361] = 5;
        cost_scaling[362] = 5;
        cost_scaling[363] = 5;
        cost_scaling[364] = 5;
        cost_scaling[365] = 5;
        cost_scaling[366] = 5;
        cost_scaling[367] = 5;
        cost_scaling[368] = 5;
        cost_scaling[369] = 5;
        cost_scaling[370] = 5;
        cost_scaling[371] = 5;
        cost_scaling[372] = 5;
        cost_scaling[373] = 5;
        cost_scaling[374] = 5;
        cost_scaling[375] = 5;
        cost_scaling[376] = 5;
        cost_scaling[377] = 5;
        cost_scaling[378] = 5;
        cost_scaling[379] = 5;
        cost_scaling[380] = 5;
        cost_scaling[381] = 5;
        cost_scaling[382] = 5;
        cost_scaling[383] = 5;
        cost_scaling[384] = 5;
        cost_scaling[385] = 5;
        cost_scaling[386] = 5;
        cost_scaling[387] = 5;
        cost_scaling[388] = 5;
        cost_scaling[389] = 5;
        cost_scaling[390] = 5;
        cost_scaling[391] = 5;
        cost_scaling[392] = 5;
        cost_scaling[393] = 5;
        cost_scaling[394] = 5;
        cost_scaling[395] = 5;
        cost_scaling[396] = 5;
        cost_scaling[397] = 5;
        cost_scaling[398] = 5;
        cost_scaling[399] = 5;
        cost_scaling[400] = 5;
        cost_scaling[401] = 5;
        cost_scaling[402] = 5;
        cost_scaling[403] = 5;
        cost_scaling[404] = 5;
        cost_scaling[405] = 5;
        cost_scaling[406] = 5;
        cost_scaling[407] = 5;
        cost_scaling[408] = 5;
        cost_scaling[409] = 5;
        cost_scaling[410] = 5;
        cost_scaling[411] = 5;
        cost_scaling[412] = 5;
        cost_scaling[413] = 5;
        cost_scaling[414] = 5;
        cost_scaling[415] = 5;
        cost_scaling[416] = 5;
        cost_scaling[417] = 5;
        cost_scaling[418] = 5;
        cost_scaling[419] = 5;
        cost_scaling[420] = 5;
        cost_scaling[421] = 5;
        cost_scaling[422] = 5;
        cost_scaling[423] = 5;
        cost_scaling[424] = 5;
        cost_scaling[425] = 5;
        cost_scaling[426] = 5;
        cost_scaling[427] = 5;
        cost_scaling[428] = 5;
        cost_scaling[429] = 5;
        cost_scaling[430] = 5;
        cost_scaling[431] = 5;
        cost_scaling[432] = 5;
        cost_scaling[433] = 5;
        cost_scaling[434] = 5;
        cost_scaling[435] = 5;
        cost_scaling[436] = 5;
        cost_scaling[437] = 5;
        cost_scaling[438] = 5;
        cost_scaling[439] = 5;
        cost_scaling[440] = 5;
        cost_scaling[441] = 5;
        cost_scaling[442] = 5;
        cost_scaling[443] = 5;
        cost_scaling[444] = 5;
        cost_scaling[445] = 5;
        cost_scaling[446] = 5;
        cost_scaling[447] = 5;
        cost_scaling[448] = 5;
        cost_scaling[449] = 5;
        cost_scaling[450] = 5;
        cost_scaling[451] = 5;
        cost_scaling[452] = 5;
        cost_scaling[453] = 5;
        cost_scaling[454] = 5;
        cost_scaling[455] = 5;
        cost_scaling[456] = 5;
        cost_scaling[457] = 5;
        cost_scaling[458] = 5;
        cost_scaling[459] = 5;
        cost_scaling[460] = 5;
        cost_scaling[461] = 5;
        cost_scaling[462] = 5;
        cost_scaling[463] = 5;
        cost_scaling[464] = 5;
        cost_scaling[465] = 5;
        cost_scaling[466] = 5;
        cost_scaling[467] = 5;
        cost_scaling[468] = 5;
        cost_scaling[469] = 5;
        cost_scaling[470] = 5;
        cost_scaling[471] = 5;
        cost_scaling[472] = 5;
        cost_scaling[473] = 5;
        cost_scaling[474] = 5;
        cost_scaling[475] = 5;
        cost_scaling[476] = 5;
        cost_scaling[477] = 5;
        cost_scaling[478] = 5;
        cost_scaling[479] = 5;
        cost_scaling[480] = 5;
        cost_scaling[481] = 5;
        cost_scaling[482] = 5;
        cost_scaling[483] = 5;
        cost_scaling[484] = 5;
        cost_scaling[485] = 5;
        cost_scaling[486] = 5;
        cost_scaling[487] = 5;
        cost_scaling[488] = 5;
        cost_scaling[489] = 5;
        cost_scaling[490] = 5;
        cost_scaling[491] = 5;
        cost_scaling[492] = 5;
        cost_scaling[493] = 5;
        cost_scaling[494] = 5;
        cost_scaling[495] = 5;
        cost_scaling[496] = 5;
        cost_scaling[497] = 5;
        cost_scaling[498] = 5;
        cost_scaling[499] = 5;
        cost_scaling[500] = 5;
        cost_scaling[501] = 5;
        cost_scaling[502] = 5;
        cost_scaling[503] = 5;
        cost_scaling[504] = 5;
        cost_scaling[505] = 5;
        cost_scaling[506] = 5;
        cost_scaling[507] = 5;
        cost_scaling[508] = 5;
        cost_scaling[509] = 5;
        cost_scaling[510] = 5;
        cost_scaling[511] = 5;
        cost_scaling[512] = 5;
        cost_scaling[513] = 5;
        cost_scaling[514] = 5;
        cost_scaling[515] = 5;
        cost_scaling[516] = 5;
        cost_scaling[517] = 5;
        cost_scaling[518] = 5;
        cost_scaling[519] = 5;
        cost_scaling[520] = 5;
        cost_scaling[521] = 5;
        cost_scaling[522] = 5;
        cost_scaling[523] = 5;
        cost_scaling[524] = 5;
        cost_scaling[525] = 5;
        cost_scaling[526] = 5;
        cost_scaling[527] = 5;
        cost_scaling[528] = 5;
        cost_scaling[529] = 5;
        cost_scaling[530] = 5;
        cost_scaling[531] = 5;
        cost_scaling[532] = 5;
        cost_scaling[533] = 5;
        cost_scaling[534] = 5;
        cost_scaling[535] = 5;
        cost_scaling[536] = 5;
        cost_scaling[537] = 5;
        cost_scaling[538] = 5;
        cost_scaling[539] = 5;
        cost_scaling[540] = 5;
        cost_scaling[541] = 5;
        cost_scaling[542] = 5;
        cost_scaling[543] = 5;
        cost_scaling[544] = 5;
        cost_scaling[545] = 5;
        cost_scaling[546] = 5;
        cost_scaling[547] = 5;
        cost_scaling[548] = 5;
        cost_scaling[549] = 5;
        cost_scaling[550] = 5;
        cost_scaling[551] = 5;
        cost_scaling[552] = 5;
        cost_scaling[553] = 5;
        cost_scaling[554] = 5;
        cost_scaling[555] = 5;
        cost_scaling[556] = 5;
        cost_scaling[557] = 5;
        cost_scaling[558] = 5;
        cost_scaling[559] = 5;
        cost_scaling[560] = 5;
        cost_scaling[561] = 5;
        cost_scaling[562] = 5;
        cost_scaling[563] = 5;
        cost_scaling[564] = 5;
        cost_scaling[565] = 5;
        cost_scaling[566] = 5;
        cost_scaling[567] = 5;
        cost_scaling[568] = 5;
        cost_scaling[569] = 5;
        cost_scaling[570] = 5;
        cost_scaling[571] = 5;
        cost_scaling[572] = 5;
        cost_scaling[573] = 5;
        cost_scaling[574] = 5;
        cost_scaling[575] = 5;
        cost_scaling[576] = 5;
        cost_scaling[577] = 5;
        cost_scaling[578] = 5;
        cost_scaling[579] = 5;
        cost_scaling[580] = 5;
        cost_scaling[581] = 5;
        cost_scaling[582] = 5;
        cost_scaling[583] = 5;
        cost_scaling[584] = 5;
        cost_scaling[585] = 5;
        cost_scaling[586] = 5;
        cost_scaling[587] = 5;
        cost_scaling[588] = 5;
        cost_scaling[589] = 5;
        cost_scaling[590] = 5;
        cost_scaling[591] = 5;
        cost_scaling[592] = 5;
        cost_scaling[593] = 5;
        cost_scaling[594] = 5;
        cost_scaling[595] = 5;
        cost_scaling[596] = 5;
        cost_scaling[597] = 5;
        cost_scaling[598] = 5;
        cost_scaling[599] = 5;
        cost_scaling[600] = 5;
        cost_scaling[601] = 5;
        cost_scaling[602] = 5;
        cost_scaling[603] = 5;
        cost_scaling[604] = 5;
        cost_scaling[605] = 5;
        cost_scaling[606] = 5;
        cost_scaling[607] = 5;
        cost_scaling[608] = 5;
        cost_scaling[609] = 5;
        cost_scaling[610] = 5;
        cost_scaling[611] = 5;
        cost_scaling[612] = 5;
        cost_scaling[613] = 5;
        cost_scaling[614] = 5;
        cost_scaling[615] = 5;
        cost_scaling[616] = 5;
        cost_scaling[617] = 5;
        cost_scaling[618] = 5;
        cost_scaling[619] = 5;
        cost_scaling[620] = 5;
        cost_scaling[621] = 5;
        cost_scaling[622] = 5;
        cost_scaling[623] = 5;
        cost_scaling[624] = 5;
        cost_scaling[625] = 5;
        cost_scaling[626] = 5;
        cost_scaling[627] = 5;
        cost_scaling[628] = 5;
        cost_scaling[629] = 5;
        cost_scaling[630] = 5;
        cost_scaling[631] = 5;
        cost_scaling[632] = 5;
        cost_scaling[633] = 5;
        cost_scaling[634] = 5;
        cost_scaling[635] = 5;
        cost_scaling[636] = 5;
        cost_scaling[637] = 5;
        cost_scaling[638] = 5;
        cost_scaling[639] = 5;
        cost_scaling[640] = 5;
        cost_scaling[641] = 5;
        cost_scaling[642] = 5;
        cost_scaling[643] = 5;
        cost_scaling[644] = 5;
        cost_scaling[645] = 5;
        cost_scaling[646] = 5;
        cost_scaling[647] = 5;
        cost_scaling[648] = 5;
        cost_scaling[649] = 5;
        cost_scaling[650] = 5;
        cost_scaling[651] = 5;
        cost_scaling[652] = 5;
        cost_scaling[653] = 5;
        cost_scaling[654] = 5;
        cost_scaling[655] = 5;
        cost_scaling[656] = 5;
        cost_scaling[657] = 5;
        cost_scaling[658] = 5;
        cost_scaling[659] = 5;
        cost_scaling[660] = 5;
        cost_scaling[661] = 5;
        cost_scaling[662] = 5;
        cost_scaling[663] = 5;
        cost_scaling[664] = 5;
        cost_scaling[665] = 5;
        cost_scaling[666] = 5;
        cost_scaling[667] = 5;
        cost_scaling[668] = 5;
        cost_scaling[669] = 5;
        cost_scaling[670] = 5;
        cost_scaling[671] = 5;
        cost_scaling[672] = 5;
        cost_scaling[673] = 5;
        cost_scaling[674] = 5;
        cost_scaling[675] = 5;
        cost_scaling[676] = 5;
        cost_scaling[677] = 5;
        cost_scaling[678] = 5;
        cost_scaling[679] = 5;
        cost_scaling[680] = 5;
        cost_scaling[681] = 5;
        cost_scaling[682] = 5;
        cost_scaling[683] = 5;
        cost_scaling[684] = 5;
        cost_scaling[685] = 5;
        cost_scaling[686] = 5;
        cost_scaling[687] = 5;
        cost_scaling[688] = 5;
        cost_scaling[689] = 5;
        cost_scaling[690] = 5;
        cost_scaling[691] = 5;
        cost_scaling[692] = 5;
        cost_scaling[693] = 5;
        cost_scaling[694] = 5;
        cost_scaling[695] = 5;
        cost_scaling[696] = 5;
        cost_scaling[697] = 5;
        cost_scaling[698] = 5;
        cost_scaling[699] = 5;
        cost_scaling[700] = 5;
        cost_scaling[701] = 5;
        cost_scaling[702] = 5;
        cost_scaling[703] = 5;
        cost_scaling[704] = 5;
        cost_scaling[705] = 5;
        cost_scaling[706] = 5;
        cost_scaling[707] = 5;
        cost_scaling[708] = 5;
        cost_scaling[709] = 5;
        cost_scaling[710] = 5;
        cost_scaling[711] = 5;
        cost_scaling[712] = 5;
        cost_scaling[713] = 5;
        cost_scaling[714] = 5;
        cost_scaling[715] = 5;
        cost_scaling[716] = 5;
        cost_scaling[717] = 5;
        cost_scaling[718] = 5;
        cost_scaling[719] = 5;
        cost_scaling[720] = 5;
        cost_scaling[721] = 5;
        cost_scaling[722] = 5;
        cost_scaling[723] = 5;
        cost_scaling[724] = 5;
        cost_scaling[725] = 5;
        cost_scaling[726] = 5;
        cost_scaling[727] = 5;
        cost_scaling[728] = 5;
        cost_scaling[729] = 5;
        cost_scaling[730] = 5;
        cost_scaling[731] = 5;
        cost_scaling[732] = 5;
        cost_scaling[733] = 5;
        cost_scaling[734] = 5;
        cost_scaling[735] = 5;
        cost_scaling[736] = 5;
        cost_scaling[737] = 5;
        cost_scaling[738] = 5;
        cost_scaling[739] = 5;
        cost_scaling[740] = 5;
        cost_scaling[741] = 5;
        cost_scaling[742] = 5;
        cost_scaling[743] = 5;
        cost_scaling[744] = 5;
        cost_scaling[745] = 5;
        cost_scaling[746] = 5;
        cost_scaling[747] = 5;
        cost_scaling[748] = 5;
        cost_scaling[749] = 5;
        cost_scaling[750] = 5;
        cost_scaling[751] = 5;
        cost_scaling[752] = 5;
        cost_scaling[753] = 5;
        cost_scaling[754] = 5;
        cost_scaling[755] = 5;
        cost_scaling[756] = 5;
        cost_scaling[757] = 5;
        cost_scaling[758] = 5;
        cost_scaling[759] = 5;
        cost_scaling[760] = 5;
        cost_scaling[761] = 5;
        cost_scaling[762] = 5;
        cost_scaling[763] = 5;
        cost_scaling[764] = 5;
        cost_scaling[765] = 5;
        cost_scaling[766] = 5;
        cost_scaling[767] = 5;
        cost_scaling[768] = 5;
        cost_scaling[769] = 5;
        cost_scaling[770] = 5;
        cost_scaling[771] = 5;
        cost_scaling[772] = 5;
        cost_scaling[773] = 5;
        cost_scaling[774] = 5;
        cost_scaling[775] = 5;
        cost_scaling[776] = 5;
        cost_scaling[777] = 5;
        cost_scaling[778] = 5;
        cost_scaling[779] = 5;
        cost_scaling[780] = 5;
        cost_scaling[781] = 5;
        cost_scaling[782] = 5;
        cost_scaling[783] = 5;
        cost_scaling[784] = 5;
        cost_scaling[785] = 5;
        cost_scaling[786] = 5;
        cost_scaling[787] = 5;
        cost_scaling[788] = 5;
        cost_scaling[789] = 5;
        cost_scaling[790] = 5;
        cost_scaling[791] = 5;
        cost_scaling[792] = 5;
        cost_scaling[793] = 5;
        cost_scaling[794] = 5;
        cost_scaling[795] = 5;
        cost_scaling[796] = 5;
        cost_scaling[797] = 5;
        cost_scaling[798] = 5;
        cost_scaling[799] = 5;
        cost_scaling[800] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Cost ****/






    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(15 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);












    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = 2;
    ubu[0] = 6;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */



    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    lh[0] = -0.03;
    uh[0] = 0.03;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
    }
    free(luh);











    /* terminal constraints */




















}

// this function only sets external functions, numerical values are set in burning_zone_acados_create_setup_nlp_in_numerical_values
void burning_zone_acados_create_setup_nlp_in(burning_zone_solver_capsule* capsule, const int N)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &capsule->impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_z", &capsule->impl_dae_fun_jac_x_xdot_z[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_jac_x_xdot_u", &capsule->impl_dae_jac_x_xdot_u_z[i]);
        
    }

    /**** Cost ****/
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun", &capsule->ext_cost_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac", &capsule->ext_cost_0_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac_hess", &capsule->ext_cost_0_fun_jac_hess);
    
    
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i-1]);
        
        
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun", &capsule->ext_cost_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac", &capsule->ext_cost_e_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac_hess", &capsule->ext_cost_e_fun_jac_hess);
    
    

    /**** Constraints ****/

    // bounds for initial stage



    // set up nonlinear constraints for stage 1 to N-1
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        
        
    }



    /* terminal constraints */
}


static void burning_zone_acados_create_set_opts(burning_zone_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 800;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
    ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, N, "cost_numerical_hessian", &ext_cost_num_hess);
}


/**
 * Internal function for burning_zone_acados_create: step 7
 */
void burning_zone_acados_set_nlp_out(burning_zone_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for burning_zone_acados_create: step 9
 */
int burning_zone_acados_create_precompute(burning_zone_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int burning_zone_acados_create_with_discretization(burning_zone_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != BURNING_ZONE_N && !new_time_steps) {
        fprintf(stderr, "burning_zone_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, BURNING_ZONE_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    burning_zone_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = burning_zone_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    burning_zone_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    burning_zone_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    burning_zone_acados_create_setup_functions(capsule);
    burning_zone_acados_create_setup_nlp_in(capsule, N);
    burning_zone_acados_create_setup_nlp_in_numerical_values(capsule, N, new_time_steps);
    burning_zone_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = burning_zone_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int burning_zone_acados_update_qp_solver_cond_N(burning_zone_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from burning_zone_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = burning_zone_acados_create_precompute(capsule);
    return status;
}


int burning_zone_acados_reset(burning_zone_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar)
{

    // set initialization to all zeros
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    // sets primal and dual iterates to zero
    ocp_nlp_out_set_values_to_zero(nlp_config, nlp_dims, nlp_out);

    // TODO this should be implemented using blasfeo_dvecse
    double* buffer = calloc(NX+NZ, sizeof(double));
    for (int i=0; i<N; i++)
    {
            ocp_nlp_set(nlp_solver, i, "xdot_guess", buffer);
            ocp_nlp_set(nlp_solver, i, "z_guess", buffer);
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    if (reset_numerical_values)
    {
        // reset parameters to initial values
        burning_zone_acados_create_set_default_parameters(capsule);

        // reset numerical values in nlp_in
        burning_zone_acados_create_setup_nlp_in_numerical_values(capsule, N, NULL);
    }

    if (reset_solver_options)
    {
        // reset solver options to initial values
        burning_zone_acados_create_set_opts(capsule);
    }

    if (reset_x_to_x0_bar)
    {ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, 0, "lbx", buffer);
        for (int i=0; i<N+1; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        }
    }

    free(buffer);
    return 0;
}




int burning_zone_acados_update_params(burning_zone_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 5;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int burning_zone_acados_update_params_sparse(burning_zone_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int burning_zone_acados_set_p_global_and_precompute_dependencies(burning_zone_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, burning_zone_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int burning_zone_acados_solve(burning_zone_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int burning_zone_acados_setup_qp_matrices_and_factorize(burning_zone_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int burning_zone_acados_free(burning_zone_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->impl_dae_fun[i]);
        external_function_external_param_casadi_free(&capsule->impl_dae_fun_jac_x_xdot_z[i]);
        external_function_external_param_casadi_free(&capsule->impl_dae_jac_x_xdot_u_z[i]);
        
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_z);
    free(capsule->impl_dae_jac_x_xdot_u_z);
    

    // cost
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac_hess);
    
    
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->ext_cost_fun[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac_hess[i]);
        
        
    }
    free(capsule->ext_cost_fun);
    free(capsule->ext_cost_fun_jac);
    free(capsule->ext_cost_fun_jac_hess);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun_jac_hess);
    
    

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);



    return 0;
}


void burning_zone_acados_print_stats(burning_zone_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1296];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int burning_zone_acados_custom_update(burning_zone_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *burning_zone_acados_get_nlp_in(burning_zone_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *burning_zone_acados_get_nlp_out(burning_zone_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *burning_zone_acados_get_sens_out(burning_zone_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *burning_zone_acados_get_nlp_solver(burning_zone_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *burning_zone_acados_get_nlp_config(burning_zone_solver_capsule* capsule) { return capsule->nlp_config; }
void *burning_zone_acados_get_nlp_opts(burning_zone_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *burning_zone_acados_get_nlp_dims(burning_zone_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *burning_zone_acados_get_nlp_plan(burning_zone_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
