//
// Created by federicosilvestri on 3/26/24.
//

#include "iostream"
#include "fstream"
#include "json.hpp"
#include "glpk.h"
#include "vector"

using json = nlohmann::json;
using namespace  std;

inline int nextPow2(double x) {
    if (x == 0)  {
        return 0;
    }
    return (int)std::pow(2, std::floor(std::log2(std::abs(x))));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "You must pass the file path (one for input=output file)" << std::endl;
        exit(1);
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Cannot open th file" << std::endl;
        return 1;
    }

    json data;
    file >> data;
    file.close();

    if (data.empty()) {
        std::cerr << "Empty JSON file." << std::endl;
        return 1;
    }

    int highest_sub = int(data["highest_sub"]);
    int min_bit_per_sub = int(data["min_bit_per_sub"]);
    int max_bit_per_sub = int(data["max_bit_per_sub"]);
    int bit_budget = int(data["bit_budget"]);
    float explained_variance = float(data["explained_variance"]);
    auto exp_var_per_sub = std::vector<float>();
    auto cum_sum_exp_per_sub = std::vector<float>();


    for (auto t : data["exp_var_per_sub"]["data"]) {
        exp_var_per_sub.push_back(t);
    }
    for (auto t : data["cumsum_exp_per_sub"]["data"]) {
        cum_sum_exp_per_sub.push_back(t);
    }


    // done loading


    /* Solve ILP to assign bits to dimensions
    c_i = variance explained in i dimension
    x_i = (integer) bits allocated in i dimensions
    maximize Sum c_i * x_i */
    glp_prob *lp = glp_create_prob();
    glp_iocp parm; glp_init_iocp(&parm);
    parm.presolve = GLP_ON;
    glp_set_obj_dir(lp, GLP_MAX);
    int glp_rows = highest_sub,
    glp_cols = highest_sub;
    glp_add_rows(lp, glp_rows);
    glp_add_cols(lp, glp_cols);

    int lastMatIdx = 1, rowCounter = 1;
    std::vector<int> rowIndices(glp_rows * glp_cols + 1, 0), colIndices(glp_rows * glp_cols + 1, 0);
    std::vector<double> numVal(glp_rows * glp_cols + 1, 0);

    // Function
    for (int i=0; i<glp_cols; i++) {
        glp_set_obj_coef(lp, i+1, exp_var_per_sub[i]);
    }

    // set integer constraint
    for (int d=1; d<=glp_cols; d++) {
        glp_set_col_kind(lp, d, GLP_IV);
    }

    /* CONSTRAINTS */
    /* 1. Bit allocation per dimension 0 or more
          x_i >= 0
          x_i <= 8
    */
    for (int i=1; i<=glp_cols; i++) {
        double lb = 0;
        if (cum_sum_exp_per_sub[i-1] <= explained_variance) {
            lb = min_bit_per_sub;
        }
        glp_set_col_bnds(lp, i, GLP_DB, lb, max_bit_per_sub);
    }

    // sum(x_i) = budget
    // 1 1 1 1 1 = budget
    glp_set_row_bnds(lp, rowCounter, GLP_FX, bit_budget, 0.0);
    for (int d = 1; d <= glp_cols; d++) {
        rowIndices[lastMatIdx] = rowCounter;
        colIndices[lastMatIdx] = d;
        numVal[lastMatIdx] = 1.0;
        lastMatIdx++;
    }
    rowCounter++;

    /* Force use of at least X% of the variance explained with a uniform single
      bit allocation scheme of size BitBudget, the most basic scheme

      if BitBudget>length(VarExplainedPerDim)
      UniformAllocVarExplained = CumSumVarExplainedPerDim(BitBudget);

      VarExplainedXPercentage = PercentVarExplained*UniformAllocVarExplained;

      Force to use dimensions such that VarExplainedXPercentage is satisfied
      for maximization problem we would set x_i >= 1, now we have - x_i <= -1
      HERE WE CONSIDER AT LEAST 2 BITS PER DIMENSION */
    for (int i=0; i<glp_cols-1; i++) {
        int k = nextPow2(exp_var_per_sub[i] / exp_var_per_sub[i+1]);
        // int k = nextPow2(varExplainedPerSubs(0) / varExplainedPerSubs(i+1));
        if (std::isnan(k) || k <= 0) {
            k = 0;
        }
        glp_set_row_bnds(lp, rowCounter, GLP_UP, 0.0, k);

        for (int j=0; j<glp_cols; j++) {
            rowIndices[lastMatIdx] = rowCounter;
            colIndices[lastMatIdx] = j+1;
            if (i == j) {
                numVal[lastMatIdx] = 1;
            } else if (i+1 == j) {
                numVal[lastMatIdx] = -1;
            }
            lastMatIdx++;
        }

        rowCounter++;
    }

    glp_load_matrix(lp, lastMatIdx-1, rowIndices.data(), colIndices.data(), numVal.data());

//    if (true) {
//        std::cout << "glp matrix: " << std::endl;
//        for (int i=0; i<glp_rows; i++) {
//            for (int j=0; j<glp_cols; j++)  {
//                std::cout << numVal.at(i*glp_cols + j + 1) << ", ";
//            }
//            int bnds = glp_get_row_type(lp, i+1);
//            if (bnds == GLP_FR) {
//                std::cout << ": FR";
//            } else if (bnds == GLP_LO) {
//                std::cout << "> " << glp_get_row_lb(lp, i+1);
//            } else if (bnds == GLP_UP) {
//                std::cout << "< " << glp_get_row_ub(lp, i+1);
//            } else if (bnds == GLP_DB) {
//                std::cout << ": DB " << glp_get_row_lb(lp, i+1) << " " << glp_get_row_ub(lp, i+1);
//            } else if (bnds == GLP_FX) {
//                std::cout << "= " << glp_get_row_lb(lp, i+1);
//            }
//            std::cout << std::endl;
//        }
//    }


    int ret = glp_intopt(lp, &parm);
    if (ret != 0) {
        std::cout << "glp solver failed: " << ret << std::endl;
        exit(1);
    }

    std::cout << "glp objective value: " << glp_mip_obj_val(lp) << std::endl;
    json out_data;
    auto out_vec = vector<int>();
    for (auto i = 1; i <= glp_cols; i++) {
        out_vec.push_back(glp_mip_col_val(lp, i));
    }
    
    out_data["bit_alloc"] = out_vec;
    std::ofstream out_file(argv[1]);
    out_file << out_data.dump(4);
    out_file.close();
    
    return 0;
}

