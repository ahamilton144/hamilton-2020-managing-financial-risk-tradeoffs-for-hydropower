
/* main.cpp - Multi-objective optimization of financial risk management for a hydropower producer

  Adapted by Andrew Hamilton from Lake Problem DPS, Sep 2017. Last revised Jan 2020.
  University of North Carolina at Chapel Hill
  andrew.hamilton@unc.edu

  Lake Problem DPS {
    Riddhi Singh, May, 2014
    The Pennsylvania State University
    rus197@psu.edu

    Adapted by Tori Ward, July 2014
    Cornell University
    vlw27@cornell.edu

    Adapted by Jonathan Herman and David Hadka, Sept-Dec 2014
    Cornell University and The Pennsylvania State University

    Adapted by Julianne Quinn, July 2015 as DPS problem
    Cornell University
    jdq8@cornell.edu
  }


*/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sstream>
#include <ctime>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include "./../misc/boostutil.h"
#include "./../misc/borg/borgms.h"
#include "./../misc/borg/moeaframework.h"

#define NUM_YEARS 20                   //20yr sims
#define NUM_SAMPLES 50000
#define NUM_LINES_STOCHASTIC_INPUT 999999    //Input file samp.txt has 1M rows
#define NUM_VARIABLES_STOCHASTIC_INPUT 3            //3 cols in input: swe,revenue,payoutCfd
#define INDEX_STOCHASTIC_REVENUE 1   
#define INDEX_STOCHASTIC_SNOW_PAYOUT 2    
#define MEAN_REVENUE 127.80086602479503     // mean revenue in absense of any financial risk mgmt. Make sure this is consistent with current input synthetic_data.txt revenue column.
#define MIN_SLOPE_CFD 0.05          // if contract slope dv < $0.05M/inch, act as if 0.
#define MIN_MAX_FUND 0.05               // if max fund dv < $0.05M, act as if 0.
#define NORMALIZE_SLOPE_CFD 4.0
#define NORMALIZE_FUND 250.0
#define BUFFER_MAX_SIZE 5000
#define EPS 0.0000000000001
#define NUM_OBJECTIVES 2
#define EPS_OBJ1 0.075
#define EPS_OBJ2 0.225
#define NUM_CONSTRAINTS 1
#define EPS_CONS1 0.05
#define NUM_DV  2
#define NUM_PARAM 6         // cost_fraction, discount_rate, delta_interest_fund, delta_interest_debt, lambda, lambda_prem_shift
#define NUM_PARAM_SAMPLES 151  // number of LHC samples of financial parameters in LHC file, including baseline. 
#define BORG_RUN_TYPE 2		// 0: single run no borg; 1: borg run, serial; 2: borg parallel for cluster;
#define SENSITIVITY_ANALYSIS 0	// 0 for baseline financial params (SFPUC October 2016 estimate), 1 for sensitivity analysis

namespace ublas = boost::numeric::ublas;
namespace tools = boost::math::tools;
namespace accumulator = boost::accumulators;
using namespace std;

double policyCashflowPostWithdrawal(const double f_fund_balance, const double f_power_price_index, const double f_cash_in);

double stochastic_input[NUM_LINES_STOCHASTIC_INPUT][NUM_VARIABLES_STOCHASTIC_INPUT];                   // Stochastic variables
double param_LHC_sample[NUM_PARAM][NUM_PARAM_SAMPLES];
#if (BORG_RUN_TYPE == 0)
double problem_dv[NUM_DV];
double problem_objs[NUM_OBJECTIVES];
double problem_constraints[NUM_CONSTRAINTS];
double pareto[7][100];
double N_pareto;
#endif


ublas::vector<double> annualized_cashflow(NUM_SAMPLES);     // annualized cash flow objective 
ublas::vector<double> debt_steal(NUM_SAMPLES);                      // no-steal constraint on debt

ublas::vector<double> revenue(NUM_YEARS);                          // state variables
ublas::vector<double> unit_payout_cfd(NUM_YEARS);      // unit payout (assuming slope $1M/inch) for capped contract for differences (cfd)
ublas::vector<double> discount_factor(NUM_YEARS);

ublas::vector<double> cashflow(NUM_YEARS);                     // decisions
ublas::vector<double> fund_balance(NUM_YEARS + 1);
ublas::vector<double> debt(NUM_YEARS + 1);
ublas::vector<double> fund_withdrawal(NUM_YEARS);

double cost_fraction;   // params to be looped over with LHC sample, doing borg each time. Declare globally.
double mean_net_revenue;
double discount_rate;
double interest_fund;
double interest_debt;
double lambda_prem_shift;
int LHC_set;
int NFE;
int NFE_counter = 0;
unsigned int seed_borg;
unsigned int seed_sample;      // use same seed for each function evaluation, so always comparing same simulations. should be less noisy.
int lines_to_use[NUM_SAMPLES];


typedef accumulator::accumulator_set<double, accumulator::stats<accumulator::tag::tail_quantile<accumulator::right> > > accumulator_t;

// problem for borg search
void portfolioProblem(double *problem_dv, double *problem_objs, double *problem_constraints) {
    // initialize variables

    zero(annualized_cashflow);
    zero(debt_steal);
    zero(revenue);
    zero(discount_factor);
    zero(unit_payout_cfd);

//    printf("%d\n", NFE_counter);
    NFE_counter += 1;

    double max_fund = problem_dv[0];
    if (max_fund < MIN_MAX_FUND){
        max_fund = 0.0;
    }
    double slope_cfd = problem_dv[1];
    if (slope_cfd < MIN_SLOPE_CFD){
        slope_cfd = 0.0;
    }
    double total_payout_cfd;

    double discount_normalization;      // discounting normalization, 1/sum_(discount_factor)
    double cash_in;
    accumulator_t debt_q95(accumulator::tag::tail<accumulator::right>::cache_size = NUM_SAMPLES);    // accumulator object for calculating upper 95th quantile of debt

    // create discounting factor
    for (int i = 0; i < NUM_YEARS; i++){
        discount_factor(i) = pow(discount_rate, i+1);
        discount_normalization += discount_factor(i);
    }
    discount_normalization = 1.0 / discount_normalization;

    // run revenue model simulation
    for (int s = 0; s < NUM_SAMPLES; s++) {
        // randomly generated revenues
        int index = lines_to_use[s];

//        printf("\n\nSample %d  %d\n", s, index);

        // get the random revenue & cfd payout from the States of the world file
        // for each sample, get 20 years from SOW file
        for (int i = 0; i < NUM_YEARS; i++) {
            revenue(i) = (stochastic_input[index + i][INDEX_STOCHASTIC_REVENUE] - MEAN_REVENUE * cost_fraction);
            unit_payout_cfd(i) = stochastic_input[index + i][INDEX_STOCHASTIC_SNOW_PAYOUT];
//            printf("%f  %f \n", revenue(i), unit_payout_cfd(i));
        }

        // initialize reserve fund, withdrawal, cashflow
        zero(fund_balance);
        zero(fund_withdrawal);
        zero(cashflow);
        zero(debt);

        //calculate new revenues, reserve fund balance, objectives
        for (int i = 0; i < NUM_YEARS; i++) {

            total_payout_cfd = slope_cfd * (unit_payout_cfd(i) - lambda_prem_shift);
            cash_in = revenue(i) + total_payout_cfd - debt(i) * interest_debt;

            cashflow(i) = policyCashflowPostWithdrawal(fund_balance(i) * interest_fund, cash_in, max_fund);
            fund_withdrawal(i) = cashflow(i) - cash_in;
            fund_balance(i + 1) = fund_balance(i) * interest_fund - fund_withdrawal(i);
            if (cashflow(i) < 0.0){
                debt(i + 1) = -cashflow(i);
                cashflow(i) = 0.0;
            }else{
                debt(i + 1) = 0.0;
            }

            annualized_cashflow(s) += cashflow(i) * discount_factor(i);
//            printf("%d %f %f %f %f\n", i, revenue(i), total_payout_cfd, cashflow(i), debt(i));
//            printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", revenue(i), unit_payout_cfd(i), slope_cfd, snow_put_strike, total_payout_cfd,
//                   cash_in, cashflow(i), fund_withdrawal(i), debt(i+1), fund_balance(i+1));

        }
        annualized_cashflow(s) = discount_normalization *
                                             (annualized_cashflow(s) +
                                                     ((fund_balance(NUM_YEARS) * interest_fund * discount_factor(0)) -
                                                             (debt(NUM_YEARS) * interest_debt * discount_factor(0))) *
                                                     discount_factor(NUM_YEARS - 1));       //annualized adjusted revenue objective
        debt_q95(vmax(debt));                                                               //q95(max(debt)) constraint
        debt_steal(s) = debt(NUM_YEARS) - debt(NUM_YEARS - 1);
//        printf("%f %f \n", annualized_cashflow(s), vmax(debt));
    }



    // aggregate objectives
    problem_objs[0] = -1 * vsum(annualized_cashflow) / NUM_SAMPLES; // max: average annualized cashflow, across sam
    problem_objs[1] = accumulator::quantile(debt_q95, accumulator::quantile_probability = 0.95);    //minimize 95th percentile of max debt

    // check constraints
    problem_constraints[0] = max(0.0, (vsum(debt_steal) / NUM_SAMPLES) - EPS_CONS1);

    annualized_cashflow.clear();
    debt_steal.clear();
    fund_balance.clear();
    debt.clear();
    revenue.clear();
    fund_withdrawal.clear();
    cashflow.clear();

}


// calculate cash flow after withdrawal, using fund balance and cash in as inputs, and constrained by max_fund.
double policyCashflowPostWithdrawal(const double f_fund_balance, const double f_cash_in, const double f_max_fund) {
    double cashflow = 0;
    if (f_cash_in < 0.0){
        if (f_fund_balance < EPS){
            cashflow = f_cash_in;
        }else{
            cashflow = min(f_cash_in + f_fund_balance, 0.0);
        }
    }else{
        if (f_fund_balance > (f_max_fund - EPS)){
            cashflow = f_cash_in + (f_fund_balance - f_max_fund);
        }else{
            cashflow = max(f_cash_in - (f_max_fund - f_fund_balance), 0.0);
        }
    }
    return cashflow;
}



int main(int argc, char* argv[]) {


    clock_t begin = clock();

    // get stochastic inputs
    for (int i = 0; i < NUM_LINES_STOCHASTIC_INPUT; i++) {
        for (int j = 0; j < NUM_VARIABLES_STOCHASTIC_INPUT; j++) {
            stochastic_input[i][j] = 0.0;
        }
    }

    FILE *myfile;
    myfile = fopen("./../../data/generated_inputs/synthetic_data.txt", "r");

    int linenum = 0;
    char testbuffer[BUFFER_MAX_SIZE];

    if (myfile == NULL) {
        perror("Error opening file");
    } else {
        char buffer[BUFFER_MAX_SIZE];
        fgets(buffer, BUFFER_MAX_SIZE, myfile);       // eat header line
        while (fgets(buffer, BUFFER_MAX_SIZE, myfile) != NULL) {
            linenum++;
            if (buffer[0] != '#') {
                char *pStart = testbuffer;
                char *pEnd;
                for (int i = 0; i < BUFFER_MAX_SIZE; i++) {
                    testbuffer[i] = buffer[i];
                }
                for (int cols = 0; cols < NUM_VARIABLES_STOCHASTIC_INPUT; cols++) {
                    stochastic_input[linenum - 1][cols] = strtod(pStart, &pEnd);
                    pStart = pEnd;
//                    printf("%f ",stochastic_input[linenum-1][cols]);
                }
//                printf("\n");
            }
        }

    }
    fclose(myfile);

    // read in LHC dv
    FILE *myfile2;
    myfile2 = fopen("./../../data/generated_inputs/param_LHC_sample_withLamPremShift.txt", "r");
    char testbuffer2[BUFFER_MAX_SIZE];
    linenum = 0;

    if (myfile2 == NULL) {
        perror("Error opening file");
    } else {
        char buffer2[BUFFER_MAX_SIZE];
        fgets(buffer2, BUFFER_MAX_SIZE, myfile2);       // eat header line
        while (fgets(buffer2, BUFFER_MAX_SIZE, myfile2) != NULL) {
            linenum++;
            if (buffer2[0] != '#') {
                char *pStart2 = testbuffer2;
                char *pEnd2;
                for (int i = 0; i < BUFFER_MAX_SIZE; i++) {
                    testbuffer2[i] = buffer2[i];
                }
                for (int i = 0; i < NUM_PARAM; i++) {
                    param_LHC_sample[i][linenum - 1] = strtod(pStart2, &pEnd2);
                    pStart2 = pEnd2;
//                    if (linenum == 1){
//                        printf("%f  %d\n",param_LHC_sample[i][linenum-1], i);
//                    }
                }
            }
        }
    }
//    printf("\n");
    fclose(myfile2);

#if BORG_RUN_TYPE > 0
    // setting random seeds
    seed_borg = atoi(argv[1]);
    seed_sample = atoi(argv[2]);      // use same seed for sample each time, so always comparing same simulations
    NFE = atoi(argv[3]);
#else
    seed_sample = atoi(argv[1]);
#endif

    srand(seed_sample);
    for (int s = 0; s < NUM_SAMPLES; s++) {
        lines_to_use[s] = rand() % (NUM_LINES_STOCHASTIC_INPUT - NUM_YEARS - 1) + 1;
//        printf("%d\n", lines_to_use[s]);
    }

#if BORG_RUN_TYPE == 2
    // interface with Borg-MS
    BORG_Algorithm_ms_startup(&argc, &argv);
    BORG_Algorithm_ms_max_evaluations(NFE);
    BORG_Algorithm_output_frequency(NFE / 50);
#endif

#if BORG_RUN_TYPE == 0      // get dv and params from argv, not borg, and just run once
    LHC_set = atoi(argv[2]);

    // read pareto set in
    FILE *myfile4;
    myfile4 = fopen(argv[3], "r");
    char testbuffer4[BUFFER_MAX_SIZE];
    linenum = 0;

    if (myfile4 == NULL) {
        perror("Error opening file");
    } else {
        char buffer4[BUFFER_MAX_SIZE];
//        fgets(buffer4, BUFFER_MAX_SIZE, myfile4);       // eat header line
        while (fgets(buffer4, BUFFER_MAX_SIZE, myfile4) != NULL) {
            linenum++;
            if (buffer4[0] != '#') {
                char *pStart4 = testbuffer4;
                char *pEnd4;
                for (int i = 0; i < BUFFER_MAX_SIZE; i++) {
                    testbuffer4[i] = buffer4[i];
                }
                for (int i = 0; i < 5; i++) {
                    pareto[i][linenum - 1] = strtod(pStart4, &pEnd4);
                    pStart4 = pEnd4;
                }
            }
        }
    }
//    printf("\n");
    fclose(myfile4);
    N_pareto = linenum;

	// open file for writing results
    ofstream retest_write;
    retest_write.open(argv[4]);

    for (int i = 0; i < N_pareto; i++){
        // decision variables
        problem_dv[0] = pareto[0][i];
        problem_dv[1] = pareto[1][i];

        // params from LHC sample
        cost_fraction = param_LHC_sample[0][LHC_set];             // fraction of MEAN_REVENUE that is must-meet costs
        double delta = param_LHC_sample[1][LHC_set];              // discount rate, as %/yr
        double Delta_interest_fund = param_LHC_sample[2][LHC_set];    // interest rate on reserve funds, as %/yr, markdown below delta (all negative)
        double Delta_interest_debt = param_LHC_sample[3][LHC_set];    // interest rate charged on debt, as %/yr, markup above delta (all positive)
        lambda_prem_shift = param_LHC_sample[5][LHC_set];         // shift in cfd premium to apply, based on lambda parameter (relative to based premiums from lambda=0.25)

        // calculated params for LHC sensitivity analysis, used in portfolioProblem
        mean_net_revenue = MEAN_REVENUE * (1. - cost_fraction);
        discount_rate = 1. / (delta / 100. + 1.);
        interest_fund = (Delta_interest_fund + delta) / 100. + 1.;
        interest_debt = (Delta_interest_debt + delta) / 100. + 1.;

        // run problem with given dv
        portfolioProblem(problem_dv, problem_objs, problem_constraints);

        retest_write << std::fixed << std::setprecision(10) << pareto[0][i] << '\t' << pareto[1][i] << '\t' << pareto[2][i] << '\t' << pareto[3][i] << '\t' << pareto[4][i] << '\t';
        retest_write << problem_objs[0] << '\t' << problem_objs[1] << '\t' << problem_constraints[0] << '\n';

    }
    retest_write.close();


#elif BORG_RUN_TYPE > 0
    // loop over uncertain parameters in LHC sample, borg each time

#if (SENSITIVITY_ANALYSIS == 0)
    int p_min = NUM_PARAM_SAMPLES - 1;
    int p_max = NUM_PARAM_SAMPLES;
#else
    int p_min = 0;
    int p_max = NUM_PARAM_SAMPLES - 1;
#endif
    for (int p = p_min; p < p_max; ++p) {
        LHC_set = p;
        // params from LHC sample
        cost_fraction = param_LHC_sample[0][p];             // fraction of MEAN_REVENUE that is must-meet costs
        double delta = param_LHC_sample[1][p];              // discount rate, as %/yr
        double Delta_interest_fund = param_LHC_sample[2][p];    // interest rate on reserve funds, as %/yr, markdown below delta (all negative)
        double Delta_interest_debt = param_LHC_sample[3][p];    // interest rate charged on debt, as %/yr, markup above delta (all positive)
        lambda_prem_shift = param_LHC_sample[5][p];         // shift in cfd premium to apply, based on lambda parameter (relative to based premiums from lambda=0.25)


        // calculated params for LHC sensitivity analysis, used in portfolioProblem
        mean_net_revenue = MEAN_REVENUE * (1. - cost_fraction);
        discount_rate = 1. / (delta / 100. + 1.);
        interest_fund = (Delta_interest_fund + delta) / 100. + 1.;
        interest_debt = (Delta_interest_debt + delta) / 100. + 1.;


//        printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", cost_fraction, delta, Delta_interest_fund, Delta_interest_debt, discount_rate, interest_fund, interest_debt, lambda_prem_shift);


#if (BORG_RUN_TYPE == 1) | (BORG_RUN_TYPE == 2)
        // Define the problem with decisions, objectives, constraints and the evaluation function
        BORG_Problem problem = BORG_Problem_create(NUM_DV, NUM_OBJECTIVES, NUM_CONSTRAINTS, portfolioProblem);

        // Set all the parameter bounds and epsilons
        BORG_Problem_set_bounds(problem, 0, 0, NORMALIZE_FUND);            //bounds for reserve max size
        BORG_Problem_set_bounds(problem, 1, 0, NORMALIZE_SLOPE_CFD);            //bounds for index contract slope

        BORG_Problem_set_epsilon(problem, 0, EPS_OBJ1); // avg_annualized_cashflow (units $M, so 0.01=$10,000)
        BORG_Problem_set_epsilon(problem, 1, EPS_OBJ2); // q95_max_debt (units $M, so 0.01=$10,000)

#if BORG_RUN_TYPE == 1

        //This is set up to run only one seed at a time
        char output_filename[256];
        FILE *output_file = NULL;
#if (SENSITIVITY_ANALYSIS == 0)
        sprintf(output_filename, "./../../data/optimization_output/baseline/sets/param%d_seedS%d_seedB%d.set", p, seed_sample, seed_borg);
#else
        sprintf(output_filename, "./../../data/optimization_output/sensitivity/sets/param%d_seedS%d_seedB%d.set", p, seed_sample, seed_borg);
#endif
        BORG_Random_seed(seed_borg);
        BORG_Archive result = BORG_Algorithm_run(problem, NFE); // this actually runs the optimization

        //If this is the master node, print out the final archive
        if (result != NULL) {
            output_file = fopen(output_filename, "w");
            if (!output_file) {
                BORG_Debug("Unable to open final output file\n");
            }
            BORG_Archive_print(result, output_file);
            BORG_Archive_destroy(result);
            fclose(output_file);
        }

        BORG_Problem_destroy(problem);

#elif BORG_RUN_TYPE == 2
        //This is set up to run only one seed at a time
        char output_filename[256];
        char runtime[256];
        FILE *output_file = NULL;
#if (SENSITIVITY_ANALYSIS == 0)
        sprintf(output_filename, "./../../data/optimization_output/baseline/sets/param%d_seedS%d_seedB%d.set", p, seed_sample, seed_borg);
        sprintf(runtime, "./../../data/optimization_output/baseline/runtime/param%d_seedS%d_seedB%d.runtime", p, seed_sample, seed_borg);
#else
        sprintf(output_filename, "./../../data/optimization_output/sensitivity/sets/param%d_seedS%d_seedB%d.set", p, seed_sample, seed_borg);
        sprintf(runtime, "./../../data/optimization_output/sensitivity/runtime/param%d_seedS%d_seedB%d.runtime", p, seed_sample, seed_borg);
#endif

        BORG_Algorithm_output_runtime(runtime);

        BORG_Random_seed(seed_borg);
        BORG_Archive result = BORG_Algorithm_ms_run(problem); // this actually runs the optimization

        //If this is the master node, print out the final archive
        if (result != NULL) {
            output_file = fopen(output_filename, "w");
            if (!output_file) {
                BORG_Debug("Unable to open final output file\n");
            }
            BORG_Archive_print(result, output_file);
            BORG_Archive_destroy(result);
            fclose(output_file);
        }

        BORG_Problem_destroy(problem);

#endif

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC / 60.0;
//        printf("%d\t%f\n", p, elapsed_secs);
    }
#endif

#if (BORG_RUN_TYPE==2)
    BORG_Algorithm_ms_shutdown();
#endif

#endif
    return EXIT_SUCCESS;

}
